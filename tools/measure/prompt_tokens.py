"""initial_prompt のトークン数と、Whisper に届く語の数を測る (#4 の U1・U2)。

faster-whisper 1.2.1 (whisperx 3.8.5 が継承する) は、initial_prompt を `" " + prompt.strip()` で
トークン化し、後ろの `448 // 2 - 1 = 223` トークンだけを使う (`get_prompt`)。ここでは
キットの `glossary.build_initial_prompt` が作る文字列を同じ形でトークン化し、後ろ 223 トークンに
欠けずに入る語 (項目) を、トークンの文字の位置 (offsets) で数える。

    # U1: 合成の辞書 (語の長さ 4 通り × 項目数)
    python tools/measure/prompt_tokens.py synthetic
    # U2: 手元の実辞書。数だけを出す (語・パス・プロジェクト名は出さない)
    python tools/measure/prompt_tokens.py real --global-dir ~/voice_memos/glossary --projects-glob "C:/Projects/*"

トークナイザは HF キャッシュの Systran/faster-whisper-large-v3 の tokenizer.json (ダウンロードしない)。
--verify (既定) は、faster-whisper の `WhisperModel.get_prompt` を実際に通した prompt の長さと中身でも確かめる
(数を 2 通りで確かめる)。import しただけでは、重い import・外部への接続・プロセスの起動をしない。
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kit  # noqa: E402

_kit.use_kit_scripts()
import glossary  # noqa: E402  (キットの scripts/glossary.py。標準ライブラリだけ)

WHISPER_MAX_LENGTH = 448
KEEP = WHISPER_MAX_LENGTH // 2 - 1  # 223
PREFIX = "以下は本音声で言及される固有名詞の一覧です: "
SEPARATOR = "、"


def whisper_prompt_text(entries: list[dict]) -> str:
    """whisperx が tokenizer.encode に渡す文字列 (`" " + initial_prompt.strip()`)。prompt が空なら ""。"""
    prompt = glossary.build_initial_prompt(entries)
    return (" " + prompt.strip()) if prompt else ""


def item_spans(text: str, canonicals: list[str]) -> list[tuple[int, int]]:
    """whisper_prompt_text の中での各 canonical の文字の範囲 [start, end)。"""
    pos = 1 + len(PREFIX)
    spans = []
    for c in canonicals:
        if text[pos:pos + len(c)] != c:
            raise ValueError("build_initial_prompt の形が想定と違う (PREFIX・区切りを見直す)")
        spans.append((pos, pos + len(c)))
        pos += len(c) + len(SEPARATOR)
    return spans


def analyze(entries: list[dict], encode, keep: int = KEEP) -> dict:
    """entries の initial_prompt のトークン数と、後ろ keep トークンに欠けずに入る項目を数える。

    encode(text) -> (ids, offsets)。項目が届くのは、切り捨てた最後のトークンの終わりの位置より後ろから
    項目が始まるとき (1 字が 2 トークンに分かれて境目がその字にかかるときは、届かない側に数える)。
    戻り値の reach_flags は項目ごとの真偽 (語そのものは持たない)。
    """
    canonicals = [e["canonical"] for e in entries if e.get("canonical")]
    text = whisper_prompt_text(entries)
    if not text:
        return {"items": 0, "chars": 0, "tokens": 0, "kept": 0, "dropped": 0, "reach": 0, "miss": 0,
                "prefix_reaches": False, "prefix_tokens": 0, "reach_flags": []}
    ids, offsets = encode(text)
    n = len(ids)
    dropped = max(0, n - keep)
    cut = offsets[dropped - 1][1] if dropped else 0
    spans = item_spans(text, canonicals)
    flags = [(s >= cut) for s, _ in spans] if dropped else [True] * len(spans)
    prefix_end = 1 + len(PREFIX)
    prefix_tokens = sum(1 for (s, e) in offsets if e <= prefix_end)
    return {
        "items": len(canonicals),
        "chars": len(text),
        "tokens": n,
        "kept": n - dropped,
        "dropped": dropped,
        "reach": sum(flags),
        "miss": len(flags) - sum(flags),
        "prefix_reaches": dropped == 0,
        "prefix_tokens": prefix_tokens,
        "reach_flags": flags,
    }


def verify_with_get_prompt(text: str, hf_tok) -> dict:
    """faster-whisper の get_prompt を実際に通して、残るトークンが「後ろ 223」と一致するかを確かめる。"""
    from types import SimpleNamespace

    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import WhisperModel

    tok = Tokenizer(hf_tok, True, task="transcribe", language="ja")
    ids = tok.encode(text)
    stub = SimpleNamespace(max_length=WHISPER_MAX_LENGTH)
    prompt = WhisperModel.get_prompt(stub, tok, ids, without_timestamps=True)
    tail = len(tok.sot_sequence) + 1  # sot・言語・task + no_timestamps
    head = 1 if ids else 0  # sot_prev
    kept = len(prompt) - head - tail
    match = kept == min(len(ids), KEEP) and (kept == 0 or prompt[head:head + kept] == ids[-kept:])
    return {"prompt_len": len(prompt), "kept_by_get_prompt": kept, "match": bool(match)}


def prefix_standalone_tokens(encode) -> int:
    ids, _ = encode(" " + PREFIX.strip())
    return len(ids)


# --- U1: 合成の辞書 ---------------------------------------------------------

def synthetic_rows(encode, profiles, counts, seed: int = 0, hf_tok=None) -> list[dict]:
    import synth

    rows = []
    for prof in profiles:
        for n in counts:
            entries = synth.mixed_entries(n, seed) if prof == "mixed" else synth.synth_entries(prof, n, seed)
            r = analyze(entries, encode)
            r.pop("reach_flags")
            row = {"profile": prof, **r,
                   "avg_canonical_chars": round(sum(len(e["canonical"]) for e in entries) / max(1, n), 2)}
            if hf_tok is not None:
                v = verify_with_get_prompt(whisper_prompt_text(entries), hf_tok)
                row["verify_kept"] = v["kept_by_get_prompt"]
                row["verify_match"] = v["match"] and v["kept_by_get_prompt"] == r["kept"]
            rows.append(row)
    return rows


def max_items_all_reach(encode, profile: str, seed: int = 0, limit: int = 300) -> int:
    """前置きの文も含めて全部が後ろ 223 トークンに入る、最大の項目数。"""
    import synth

    best = 0
    for n in range(1, limit + 1):
        entries = synth.mixed_entries(n, seed) if profile == "mixed" else synth.synth_entries(profile, n, seed)
        if analyze(entries, encode)["dropped"] == 0:
            best = n
        else:
            break
    return best


# --- U2: 手元の実辞書 (数だけ) ------------------------------------------------

def _quiet(fn, *a, **k):
    """キットの関数が出すパス・ファイル名の print を捨てる (出力に私的な名前を出さない)。"""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = fn(*a, **k)
    return res, buf.getvalue().count("WARN")


def _layer_entries(d: Path | None, label: str) -> tuple[list[dict], int]:
    entries, warns = _quiet(glossary.load_glossary, d)
    for e in entries:
        e["_layer"] = label
    return entries, warns


def real_rows(encode, global_dir: Path | None, project_roots: list[Path], hf_tok=None) -> list[dict]:
    """実辞書の数だけを返す。語・パス・プロジェクト名・ファイル名は戻り値に入れない。

    マージはキットの load_merged_glossary と同じ順 (グローバル → extra → プロジェクト)。
    プロジェクトの extra_glossary_dirs は、キットの load_config・resolve_paths で解決する。
    """
    import transcribe  # キットの scripts/transcribe.py (標準ライブラリと glossary・proofread・engine だけ)

    rows = []
    g_entries, g_warn = _layer_entries(global_dir, "global") if global_dir else ([], 0)

    def row(label: str, layers: list[tuple[str, list[dict]]], warns: int) -> dict:
        merged = glossary.merge_glossaries(*[l for _, l in layers if l]) if any(l for _, l in layers) else []
        r = analyze(merged, encode)
        flags = r.pop("reach_flags")
        by_layer: dict[str, dict] = {}
        for e, f in zip([e for e in merged if e.get("canonical")], flags):
            b = by_layer.setdefault(e.get("_layer", "?"), {"items": 0, "reach": 0})
            b["items"] += 1
            b["reach"] += int(f)
        return {
            "label": label,
            "layer_items": {name: len(l) for name, l in layers},
            "merged_items": r["items"],
            "tokens": r["tokens"],
            "kept": r["kept"],
            "dropped": r["dropped"],
            "reach": r["reach"],
            "miss": r["miss"],
            "prefix_reaches": r["prefix_reaches"],
            "reach_by_layer": by_layer,
            "avg_canonical_chars": round(
                sum(len(e["canonical"]) for e in merged if e.get("canonical")) / max(1, r["items"]), 2),
            "parse_warnings": warns,
            **(_verify_fields(merged, r, hf_tok) if hf_tok is not None and merged else {}),
        }

    if g_entries:
        rows.append(row("G", [("global", g_entries)], g_warn))
    i = 0
    for root in sorted(project_roots, key=lambda p: str(p).lower()):
        warns = g_warn
        try:
            (cfg, _w) = _quiet(transcribe.load_config, root)
        except (ValueError, OSError):  # 壊れた config.json (JSONDecodeError は ValueError)。数えて既定の設定で続ける
            warns += 1
            cfg = json.loads((transcribe.SKILL_DIR / "config.default.json").read_text(encoding="utf-8-sig"))
        (paths, _w2) = _quiet(transcribe.resolve_paths, cfg, root)
        extra_layers = []
        for d in paths.get("extra_glossary_dirs", []):
            ents, w = _layer_entries(d, "extra")
            extra_layers.append(ents)
            warns += w
        proj, w = _layer_entries(paths.get("glossary"), "project")
        warns += w
        if not proj and not any(extra_layers):
            continue
        i += 1
        layers = [("global", g_entries)] + [("extra", e) for e in extra_layers] + [("project", proj)]
        # 同じ名前の層 (extra が複数) は数を足す
        counts: dict[str, int] = {}
        for name, l in layers:
            counts[name] = counts.get(name, 0) + len(l)
        r = row(f"P{i}", layers, warns)
        r["layer_items"] = counts
        rows.append(r)
    return rows


def _verify_fields(entries: list[dict], r: dict, hf_tok) -> dict:
    v = verify_with_get_prompt(whisper_prompt_text(entries), hf_tok)
    return {"verify_kept": v["kept_by_get_prompt"], "verify_match": v["match"] and v["kept_by_get_prompt"] == r["kept"]}


def find_project_roots(pattern: str) -> list[Path]:
    """pattern (例: C:/Projects/*) のうち、voice_memos/glossary か voice_memos/config.json を持つもの。"""
    out = []
    for p in glob.glob(str(Path(pattern).expanduser())):
        root = Path(p)
        vm = root / "voice_memos"
        if (vm / "glossary").is_dir() or (vm / "config.json").is_file():
            out.append(root)
    return out


# --- 出力 -------------------------------------------------------------------

def _table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(
            json.dumps(r.get(c), ensure_ascii=False) if isinstance(r.get(c), (dict, list)) else str(r.get(c))
            for c in cols) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="initial_prompt のトークン数と届く語の数 (#4 U1・U2)")
    ap.add_argument("--tokenizer", type=Path, default=None, help="tokenizer.json (省略時は HF キャッシュ)")
    ap.add_argument("--no-verify", action="store_true", help="get_prompt での確かめを省く")
    ap.add_argument("--json", type=Path, default=None, help="結果の JSON の書き出し先")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synthetic", help="U1: 合成の辞書")
    s.add_argument("--profiles", default="S,M,L,E,mixed")
    s.add_argument("--counts", default="1,5,10,20,30,40,50,60,78,100")
    s.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("real", help="U2: 手元の実辞書 (数だけ)")
    r.add_argument("--global-dir", type=Path, default=None)
    r.add_argument("--projects-glob", default=None, help='例: "C:/Projects/*"')
    args = ap.parse_args(argv)
    _kit.utf8_stdout()

    tok_path = args.tokenizer or _kit.find_tokenizer_json()
    hf_tok = _kit.load_hf_tokenizer(tok_path)
    encode = _kit.hf_encoder(hf_tok)
    verify_tok = None if args.no_verify else hf_tok
    print(f"tokenizer: {_kit.DEFAULT_MODEL_REPO} tokenizer.json (keep={KEEP})")
    print(f"prefix tokens: standalone={prefix_standalone_tokens(encode)}")

    out: dict = {"keep": KEEP}
    if args.cmd == "synthetic":
        profiles = [p for p in args.profiles.split(",") if p]
        counts = [int(c) for c in args.counts.split(",") if c]
        rows = synthetic_rows(encode, profiles, counts, args.seed, verify_tok)
        print(_table(rows, ["profile", "items", "avg_canonical_chars", "chars", "tokens", "kept", "dropped",
                            "reach", "miss", "prefix_reaches", "prefix_tokens"]
                     + (["verify_kept", "verify_match"] if verify_tok else [])))
        maxes = {p: max_items_all_reach(encode, p, args.seed) for p in profiles}
        print("max items (prefix + all items within the last 223 tokens): "
              + ", ".join(f"{p}={n}" for p, n in maxes.items()))
        out.update({"rows": rows, "max_items_all_reach": maxes})
    else:
        roots = find_project_roots(args.projects_glob) if args.projects_glob else []
        g = args.global_dir.expanduser() if args.global_dir else None
        rows = real_rows(encode, g, roots, verify_tok)
        print(f"projects scanned: {len(roots)} (with glossary or config), rows: {len(rows)}")
        print(_table(rows, ["label", "layer_items", "merged_items", "avg_canonical_chars", "tokens", "kept",
                            "dropped", "reach", "miss", "prefix_reaches", "reach_by_layer", "parse_warnings"]
                     + (["verify_kept", "verify_match"] if verify_tok else [])))
        out.update({"rows": rows})
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
