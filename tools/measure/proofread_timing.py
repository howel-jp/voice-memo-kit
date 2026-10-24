"""校閲 (Claude CLI) の所要時間と本文の字数の関係を、合成の文で測る (#4 の U4)。

キットの `proofread.proofread_body` をそのまま呼ぶ。変えるのは `subprocess.run` の timeout だけ
(既定 1,800 秒。900 秒を超えるときの実の時間を知るため。キットと同じにするなら --timeout 900)。
CLI が返す JSON (duration_ms、usage など) は、キットが result 以外を捨てる前に (`_call_claude_json` が
書かせる stdout のファイルから) 写し取る。送る本文は synth.py の合成の文だけ。

    python tools/measure/proofread_timing.py --call-claude --sizes 1000,4000,8000 --out-dir <dir> --label run1

--call-claude を付けたときだけ CLI を呼ぶ (1 サイズにつき proofread_body を 1 回。3,000 字を超える本文は
キットが分けて校閲するので、CLI の呼び出しは 2 回以上になる。呼び出しごとの記録は calls)。
import しただけでは何もしない。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kit  # noqa: E402

_kit.use_kit_scripts()


_BODY_MARK = "【文字起こし本文】\n"


def make_run_wrapper(real_run, timeout: float, record: dict):
    """subprocess.run の代わり。timeout だけを替え、CLI の stdout (JSON のファイル) を写し取る。

    呼び出しごとに record["calls"] に足す (input_chars: そのまとまりの字数、wall_sec、raw_json)。
    record["flags"] は最初の呼び出しのもの。record["wall_sec"]・record["raw_json"] は最後の呼び出しのもの。
    """

    def run(cmd, *args, **kwargs):
        kwargs["timeout"] = timeout
        record.setdefault("flags", [c for c in cmd[1:] if isinstance(c, str) and c.startswith("--")])
        call: dict = {}
        record.setdefault("calls", []).append(call)
        fin_name = getattr(kwargs.get("stdin"), "name", None)
        if fin_name:
            try:  # 別に開いて読む (子に渡すファイルの読み位置は動かさない)
                call["input_chars"] = len(Path(fin_name).read_text(encoding="utf-8").split(_BODY_MARK, 1)[-1])
            except OSError:
                pass
        t0 = time.perf_counter()
        try:
            cp = real_run(cmd, *args, **kwargs)
        finally:
            record["wall_sec"] = call["wall_sec"] = round(time.perf_counter() - t0, 2)
        fout = kwargs.get("stdout")
        name = getattr(fout, "name", None)
        if name:
            try:
                fout.flush()
            except Exception:
                pass
            try:
                record["raw_json"] = call["raw_json"] = Path(name).read_text(encoding="utf-8")
            except OSError:
                pass
        return cp

    return run


_SUM_KEYS = ("duration_ms", "duration_api_ms", "input_tokens", "cache_creation_input_tokens",
             "cache_read_input_tokens", "output_tokens", "result_chars")


def merge_meta(metas: list[dict]) -> dict:
    """呼び出しが 2 回以上のときの、行の列の値 (時間とトークンは合計、num_turns は最大、is_error はどれか)。"""
    if len(metas) == 1:
        return metas[0]
    out: dict = {}
    for k in _SUM_KEYS:
        vals = [m.get(k) for m in metas if m.get(k) is not None]
        out[k] = sum(vals) if vals else None
    turns = [m.get("num_turns") for m in metas if isinstance(m.get("num_turns"), int)]
    out["num_turns"] = max(turns) if turns else None
    out["is_error"] = any(bool(m.get("is_error")) for m in metas)
    last = metas[-1] if metas else {}
    out["subtype"], out["stop_reason"] = last.get("subtype"), last.get("stop_reason")
    if "error_result_head" in last:
        out["error_result_head"] = last["error_result_head"]
    return out


def cli_meta(raw_json: str | None) -> dict:
    """CLI の JSON から、時間とトークンの数だけを取り出す (result の本文は入れない)。

    is_error のときだけ、result の先頭 300 字 (CLI のエラーの文) を error_result_head に入れる
    (2026-10-24、12,638 字で num_turns 4・出力 128,001 トークンの is_error を観測したが、文を残していなかった)。
    """
    if not raw_json:
        return {}
    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return {"json_error": True}
    usage = data.get("usage") or {}
    extra = {"error_result_head": (data.get("result") or "")[:300]} if data.get("is_error") else {}
    return {
        **extra,
        "duration_ms": data.get("duration_ms"),
        "duration_api_ms": data.get("duration_api_ms"),
        "num_turns": data.get("num_turns"),
        "is_error": data.get("is_error"),
        "subtype": data.get("subtype"),
        "stop_reason": data.get("stop_reason"),
        "input_tokens": usage.get("input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "result_chars": len(data.get("result") or ""),
    }


def _ends_cleanly(text: str) -> bool:
    return text.rstrip()[-1:] in ("。", "？", "！", "?", "!", "」", "）")


def coverage(body: str, out: str) -> dict:
    """校閲後の段落が、入力のどの段落に当たるか (最も似た段落の番号) から、欠けた範囲を見る。

    比の安全ガード (60%) を通っても、入力の前の方や途中が丸ごと無いことがある (2026-10-24 に観測)。

    input_paras_not_matched は「どの出力段落からも最も似た段落として選ばれなかった入力段落の数」。
    欠けと、段落の結合 (2 段落が 1 つの出力段落になる) を区別しない。欠けた位置は first・last で読む
    (PR #7 のレビュー 1 回目の任意 3)。
    """
    if not out:
        return {}

    def grams(t: str) -> set[str]:
        return {t[i:i + 2] for i in range(len(t) - 1)}

    def sim(a: set[str], b: set[str]) -> float:
        return len(a & b) / max(1, len(a | b))  # 2 字の組の Jaccard

    inp = [grams(p) for p in body.split("\n\n")]
    outp = [grams(p) for p in out.split("\n\n")]
    matched = [max(range(len(inp)), key=lambda i: sim(inp[i], o)) for o in outp]
    covered = set(matched)
    return {
        "first_output_matches_input_para": matched[0],
        "last_output_matches_input_para": matched[-1],
        "input_paras_not_matched": len(inp) - len(covered),
    }


def measure_one(body: str, canonicals: list[str], model: str, timeout: float, real_run=None) -> tuple[dict, str]:
    """1 回の校閲を測る。戻り値は (数だけの記録, 校閲後の本文 (合成の文) か "")。"""
    import proofread

    record: dict = {}
    real_run = real_run or subprocess.run
    status, out, err = "ok", "", ""
    t0 = time.perf_counter()
    with mock.patch.object(proofread.subprocess, "run", make_run_wrapper(real_run, timeout, record)):
        try:
            out = proofread.proofread_body(body, canonicals, model=model)
        except proofread.ProofreadRejected as e:
            status, err = "rejected", type(e).__name__
        except subprocess.TimeoutExpired as e:
            status, err = "timeout", type(e).__name__
        except Exception as e:  # noqa: BLE001
            status, err = "error", type(e).__name__
    total = round(time.perf_counter() - t0, 2)
    calls = [{"input_chars": c.get("input_chars"), "wall_sec": c.get("wall_sec"), **cli_meta(c.get("raw_json"))}
             for c in record.get("calls", [])]
    # 呼び出しが 1 回のときは、今までの列と同じ値 (#4 の数を読み直せるように)
    meta = cli_meta(record.get("raw_json")) if len(calls) <= 1 else merge_meta(calls)
    wall_cli = record.get("wall_sec") if len(calls) <= 1 else round(sum(c["wall_sec"] or 0 for c in calls), 2)
    row = {
        "status": status,
        "error": err,
        "model": model,
        "timeout": timeout,
        "input_chars": len(body),
        "input_paragraphs": body.count("\n\n") + 1,
        "output_chars": len(out),
        "output_paragraphs": (out.count("\n\n") + 1) if out else 0,
        "ratio": round(len(out) / max(1, len(body)), 3) if out else None,
        "returned_input_unchanged": out == body,
        "output_ends_cleanly": _ends_cleanly(out) if out else None,
        "wall_sec_cli": wall_cli,
        "wall_sec_total": total,
        "flags": record.get("flags"),
        **coverage(body, out),
        **meta,
        "cli_calls": len(calls),
        "calls": calls,
    }
    return row, out


def default_model() -> str:
    cfg = json.loads((_kit.KIT_ROOT / "config.default.json").read_text(encoding="utf-8-sig"))
    return cfg.get("proofread", {}).get("model", "claude-sonnet-4-6")


def main(argv: list[str] | None = None) -> int:
    import synth

    ap = argparse.ArgumentParser(description="校閲の時間と字数 (#4 U4)。合成の文だけを送る")
    ap.add_argument("--call-claude", action="store_true", help="これを付けたときだけ Claude CLI を呼ぶ")
    ap.add_argument("--sizes", required=True, help="本文の字数 (カンマ区切り)。1 つにつき 1 回呼ぶ")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="省略時は config.default.json の proofread.model")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--out-dir", type=Path, required=True, help="JSONL と合成の入出力の置き場 (repo の外)")
    ap.add_argument("--label", default="run")
    args = ap.parse_args(argv)
    _kit.utf8_stdout()

    sizes = [int(s) for s in args.sizes.split(",") if s]
    model = args.model or default_model()
    names = synth.memo_names(args.seed)
    if not args.call_claude:
        for n in sizes:
            body = synth.synth_memo_text(n, args.seed, names)
            print(f"[dry-run] size={n} input_chars={len(body)} (CLI は呼ばない。--call-claude で呼ぶ)")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log = args.out_dir / f"proofread_timing_{args.label}.jsonl"
    for n in sizes:
        body = synth.synth_memo_text(n, args.seed, names)
        stem = f"{args.label}_{n}"
        (args.out_dir / f"{stem}_in.txt").write_text(body, encoding="utf-8")
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{started}] size={n} input_chars={len(body)} model={model} timeout={args.timeout} ...", flush=True)
        row, out = measure_one(body, names, model, args.timeout)
        row.update({"label": args.label, "size": n, "started": started})
        if out:
            (args.out_dir / f"{stem}_out.txt").write_text(out, encoding="utf-8")
        with log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    print(f"log: {log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
