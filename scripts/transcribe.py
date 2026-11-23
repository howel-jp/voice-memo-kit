"""ボイスメモ文字起こし CLI（ローカル WhisperX → Claude が読める Markdown）。

プロジェクトに依存しない共通ツール。**カレントディレクトリをプロジェクトルート**とみなし、
その配下の voice_memos/ を入出力に使う（--project で明示指定も可）。

    python transcribe.py                       # <project>/voice_memos/inbox の音声を全部処理
    python transcribe.py path/to/memo.m4a      # 単一ファイル
    python transcribe.py path/to/folder        # フォルダ内の音声を全部
    python transcribe.py --timestamps          # 各段落頭に [mm:ss] を付与
    python transcribe.py --no-glossary         # 固有名詞辞書を使わない
    python transcribe.py --no-proofread        # Claude 校閲をスキップ
    python transcribe.py --keep                # 処理後に音声を inbox に残す
    python transcribe.py --device cpu          # GPU 不調時の CPU フォールバック
    python transcribe.py --no-topic            # 話題 (topic) を付けない (config で入れたとき)
    python transcribe.py --no-on-complete      # 終わりの知らせ (on_complete) を 1 回だけ止める
    python transcribe.py --list                # 対象の音声と置き場を出すだけ (何もしない)

設定: config.default.json（キット既定）に <project>/voice_memos/config.json を重ねて上書き。

2 層構造（プロジェクト層 + グローバル層 ~/voice_memos）:
- 音声: <project>/voice_memos/inbox → ~/voice_memos/inbox の順に両方を処理（プロジェクト優先）
- 辞書: ~/voice_memos/glossary → paths.extra_glossary_dirs（明示追加）→ <project>/voice_memos/glossary
  の順にマージ（同 canonical は後の層＝プロジェクトが優先）
- 出力: <project>/voice_memos/transcripts/<録音日>_<元ファイル名>.md
  （config の output_root の候補がこの PC にあれば、その根の下。どれも無ければ根を作らずに既定の置き場。#14）
- 退避: 音声が見つかった層の processed/（プロジェクト層は、output_root があればその根の下）
- 終わりの知らせ: config の on_complete.command があれば、md を書いて音声を退かした後に、
  md の絶対パスを最後の引数にして、シェルを通さずに起こす（1 md に 1 回。#16）
--no-global でグローバル層を無視。
"""
from __future__ import annotations

import argparse
import datetime
import json
import locale
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Windows cp932 で encode できない文字でのクラッシュを防ぐ
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from engine import load_prompt_encoder, transcribe_audio
from glossary import (
    PROMPT_MAX_TOKENS,
    WHISPER_MAX_LENGTH,
    PromptPlan,
    apply_replacements_to_segments,
    build_initial_prompt,  # noqa: F401（0.5.0 までの名前を transcribe からも引けるように残す）
    load_glossary,
    merge_glossaries,
    plan_initial_prompt,
)
from proofread import TOPIC_MAX_CHARS, describe_failure, make_topic, proofread_body

SKILL_DIR = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".mp4"}

# 出力 md の <details> の見出し（SKILL.md §3 に同じ文を書く）。
# 校閲前: 辞書の置き換え (B) の後・校閲 (C) の前の文。本文が校閲後のときだけ付く。
# 置き換え前: Whisper の出力そのまま (B の前)。置き換えが 1 箇所以上あったときだけ付く
#   （0 箇所なら、本文か校閲前の節と同じ文になるため）。
PRE_PROOFREAD_SUMMARY = "校閲前（辞書の置き換え後）"
PRE_REPLACEMENT_SUMMARY = "置き換え前（Whisper の出力）"
# frontmatter の proofread_error_detail の長さの上限（改行は除く）
ERROR_DETAIL_MAX_CHARS = 200
# 1 チャンクの出力（数え直し）がこれを超えたら [WARN]。上限は 224（prompt を使い切ると 221。#4 の U3）。
# 値は #3 の保留（1 チャンク 224 トークンの頭打ち）の再開の条件と同じ。
CHUNK_OUTPUT_WARN_TOKENS = 200
# 置き場の根の候補が、この PC にどれも無いときの行（SKILL.md §2・README に同じ文を書く。#14）
OUTPUT_ROOT_MISSING_WARN = ("  [WARN] 置き場の根（output_root.candidates）が、この PC にどれもありません。"
                            "根は作らずに、既定の置き場に出します")
# --out-dir を渡したときは md はそこに出るので、既定に戻るのは音声の退避先だけ（README に同じ文。PR #15 のレビュー 2 回目の任意 1）
OUTPUT_ROOT_MISSING_WARN_OUT_DIR = ("  [WARN] 置き場の根（output_root.candidates）が、この PC にどれもありません。"
                                    "根は作らずに、音声の退避先は既定の置き場にします（md は --out-dir に出します）")
# 展開できずに残った環境変数（$VAR・${VAR}・%VAR%）。残った候補は飛ばす
_UNEXPANDED_VAR_RE = re.compile(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_][A-Za-z0-9_]*%")

# 終わりの知らせ（on_complete。#16）。行の決まった部分は README・SKILL.md に同じ文を書く（tests/test_on_complete.py の DocsTest）
ON_COMPLETE_DEFAULT_TIMEOUT_SEC = 120
# 上限の秒の上限（1 日）。Windows の待ちはミリ秒の 32 ビット（約 49.7 日）で、超えると回り込む（PR #17 のレビュー 1 回目の任意 3）
ON_COMPLETE_MAX_TIMEOUT_SEC = 86400
ON_COMPLETE_FAILED = "終わりの知らせのコマンドが失敗しました"
ON_COMPLETE_TIMED_OUT = "秒で終わらないため止めました"  # 前に「終わりの知らせのコマンドが <上限> 」が付く
ON_COMPLETE_CANNOT_START = "終わりの知らせのコマンドを起こせません"
ON_COMPLETE_DONE_NOTE = "文字起こしは済んでいます"
# 行の頭。成功の行は OK_HEAD、それ以外（失敗・時間切れ・起こせない）は WARN_HEAD で始まる。
# ほかのスキルは、成否をこの頭で読んでいる（0.8.0 から）。ほかのスキルとの約束（tests/test_kit_api.py。#18）
ON_COMPLETE_OK_HEAD = "[E] 終わりの知らせ: exit 0"
ON_COMPLETE_WARN_HEAD = "  [WARN] "
ON_COMPLETE_EXE_ONLY = ("起こすのは実行ファイル（.exe）だけです。cmd・.bat・.cmd は md のパスの & % ^ をコマンドとして読みうるため。"
                        "node・python・powershell -File などを最初に書きます")
ON_COMPLETE_EXE_ABSOLUTE = ("実行ファイル（.exe）の絶対パスを書きます。cmd・.bat・.cmd は md のパスの & % ^ を"
                            "コマンドとして読みうるため")
# 出力の要点（最後の空でない行）の長さの上限と、要点を探すために読む出力の末尾のバイト数
ON_COMPLETE_TAIL_MAX_CHARS = 200
ON_COMPLETE_READ_TAIL_BYTES = 64 * 1024
# Windows で起こすのは実行ファイル（.exe・.com。拡張子なしは CreateProcess が .exe を足す）だけ。
# cmd と .bat・.cmd などは、md のパスの & % ^（音声の名前から来る）をコマンドとして読みうるので起こさない（PR #17 のレビュー 1 回目の推奨 3・任意 1）
_SAFE_EXTS = ("", ".exe", ".com")
_SHELL_NAMES = ("cmd", "cmd.exe")
# 端末の色などの印（ANSI の CSI 列）。出力の要点から外す
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(project_root: Path) -> dict:
    # utf-8-sig: メモ帳や PowerShell 5.1 が付ける BOM を許容する
    cfg = json.loads((SKILL_DIR / "config.default.json").read_text(encoding="utf-8-sig"))
    local = project_root / "voice_memos" / "config.json"
    if local.exists():
        cfg = _deep_merge(cfg, json.loads(local.read_text(encoding="utf-8-sig")))
        print(f"config: {local}")
    return cfg


def _expand_candidate(raw, project_root: Path) -> Path | None:
    """置き場の根の候補 1 つを、この PC のパスにする。環境変数（$VAR・${VAR}・%VAR%）と ~ を展開する。

    展開できない環境変数が残るもの・空のもの・文字列でないものは None（その候補は飛ばす）。
    参照する環境変数のどれかが空のものも飛ばす（`${EMPTY}/sub` が `/sub` = ドライブの根の下を指さないように）。
    相対パスはプロジェクトルート基準。
    """
    if not isinstance(raw, str):
        return None
    for m in _UNEXPANDED_VAR_RE.finditer(raw):
        name = m.group(0).strip("${}%")
        if not os.environ.get(name, "").strip():  # 無い・空の変数（無いものは下の確かめでも飛ぶ）
            return None
    s = os.path.expandvars(raw).strip()
    if not s or _UNEXPANDED_VAR_RE.search(s):
        return None
    path = Path(s).expanduser()
    return path if path.is_absolute() else project_root / path


def _under_root(root: Path, rel, key: str) -> Path:
    """根の下の相対パス（output_root.transcripts・processed）。根の外を指すもの（絶対パス・..）は断る。"""
    rel_path = Path(rel) if isinstance(rel, str) and rel.strip() else None
    if rel_path is None or rel_path.is_absolute() or rel_path.drive or rel_path.root or ".." in rel_path.parts:
        raise ValueError(f"output_root.{key} は、根の下の相対パスにします（空・絶対パス・.. は使えません）")
    return root / rel_path


def resolve_output_root(cfg: dict, project_root: Path) -> tuple[Path | None, bool]:
    """置き場の根（config の output_root.candidates のうち、この PC にある最初のフォルダ）。

    返す値: (根, 候補はあるのにどれも無いか)。根は作らない（無いフォルダは候補として飛ばす。#14）。
    """
    out = cfg.get("output_root") or {}
    candidates = out.get("candidates") or []
    if isinstance(candidates, str):
        candidates = [candidates]
    for raw in candidates:
        path = _expand_candidate(raw, project_root)
        if path is not None and path.is_dir():
            return path, False
    return None, bool(candidates)


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    """プロジェクト層とグローバル層（~/voice_memos）のパスを解決する。

    - glossary: マージ順（後ほど優先）= グローバル <global_root>/glossary
      → paths.extra_glossary_dirs（明示追加。プロジェクト共有の辞書など）
      → プロジェクト paths.glossary（既定 voice_memos/glossary、最優先）。
    - inbox: プロジェクト → グローバルの順に両方を走査。
    - transcripts: プロジェクト側（呼び出し元の作業場所に集約）。output_root の根があれば、その下の output_root.transcripts。
    - processed: 音声が見つかった層の processed/ に退避。プロジェクト層は、根があればその下の output_root.processed。
    - output_root: 使う根（無ければ None）。output_root_missing: 候補はあるのに、この PC にどれも無い
      （そのときは根を作らずに既定の置き場。呼び出し側が [WARN] を出す）。
    output_root の transcripts・processed が根の外を指すときは ValueError（根があってもなくても確かめる）。
    """
    p = cfg["paths"]
    glossary = project_root / p["glossary"]
    extra = [Path(d).expanduser() for d in p.get("extra_glossary_dirs", []) or []]
    extra = [d if d.is_absolute() else project_root / d for d in extra]

    global_root: Path | None = None
    raw = cfg.get("global_root")
    if raw:
        global_root = Path(raw).expanduser()
    g = cfg.get("global_paths", {})

    transcripts = project_root / p["transcripts"]
    processed = project_root / p["processed"]
    out = cfg.get("output_root") or {}
    root, missing = resolve_output_root(cfg, project_root)
    if out.get("candidates"):
        check = root or project_root  # 根が無い PC でも、設定の誤りは同じように知らせる
        under = (_under_root(check, out.get("transcripts"), "transcripts"),
                 _under_root(check, out.get("processed"), "processed"))
        if root is not None:
            transcripts, processed = under
    return {
        "glossary": glossary,
        "extra_glossary_dirs": extra,
        "inbox": project_root / p["inbox"],
        "transcripts": transcripts,
        "processed": processed,
        "global_root": global_root,
        "global_glossary": (global_root / g["glossary"]) if global_root and g.get("glossary") else None,
        "global_inbox": (global_root / g["inbox"]) if global_root and g.get("inbox") else None,
        "global_processed": (global_root / g["processed"]) if global_root and g.get("processed") else None,
        "output_root": root,
        "output_root_missing": missing,
    }


def _audio_in(dir_path: Path | None) -> list[Path]:
    if dir_path is None or not dir_path.is_dir():
        return []
    return sorted(f for f in dir_path.iterdir() if f.suffix.lower() in AUDIO_EXTS)


def collect_inputs(arg_input: str | None, paths: dict, use_global: bool) -> list[tuple[Path, Path]]:
    """処理対象を (音声, 退避先 processed/) の組で返す。

    引数なし: プロジェクト inbox → グローバル inbox の順（プロジェクト優先）。
    引数あり: 指定ファイル/フォルダ。退避先は、グローバル inbox 配下ならグローバル、それ以外はプロジェクト。
    """
    def processed_for(audio: Path) -> Path:
        gi = paths.get("global_inbox")
        if gi and paths.get("global_processed") and gi in audio.resolve().parents:
            return paths["global_processed"]
        return paths["processed"]

    if arg_input:
        target = Path(arg_input)
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        files = _audio_in(target) if target.is_dir() else [target]
        return [(f, processed_for(f)) for f in files]

    items = [(f, paths["processed"]) for f in _audio_in(paths["inbox"])]
    if use_global:
        items += [(f, paths["global_processed"]) for f in _audio_in(paths.get("global_inbox"))]
    return items


def format_body(segments: list, gap_sec: float, with_timestamps: bool) -> str:
    """セグメントを段落化。無音 gap > gap_sec で段落改行。日本語は空白なし連結。"""
    paragraphs: list[tuple[float, str]] = []
    cur: list[str] = []
    cur_start: float | None = None
    prev_end: float | None = None

    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        if prev_end is not None and (start - prev_end) > gap_sec and cur:
            paragraphs.append((cur_start or 0.0, "".join(cur)))
            cur = []
            cur_start = None
        if cur_start is None:
            cur_start = start
        cur.append(text)
        prev_end = end

    if cur:
        paragraphs.append((cur_start or 0.0, "".join(cur)))

    lines = []
    for st, body in paragraphs:
        if with_timestamps:
            lines.append(f"[{int(st // 60):02d}:{int(st % 60):02d}] {body}")
        else:
            lines.append(body)
    return "\n\n".join(lines)


def fmt_duration(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def details_section(summary: str, text: str) -> str:
    """出力 md の折りたたみの節（本文の後ろに続ける）。"""
    return f"\n\n<details>\n<summary>{summary}</summary>\n\n{text}\n\n</details>\n"


def one_line(text: str, limit: int = ERROR_DETAIL_MAX_CHARS) -> str:
    """frontmatter に書く 1 行にする（空白と改行を 1 つの空白にまとめ、limit 字で切る）。"""
    return " ".join(str(text).split())[:limit]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        cand = path.parent / f"{path.stem}_{i}{path.suffix}"
        if not cand.exists():
            return cand
        i += 1


def write_atomic(path: Path, text: str) -> None:
    """`.` で始まる一時ファイルに書いてから名前を変える（ほかの道具が書きかけの md を読まない。#14）。

    書き方（エンコーディング・改行）は write_text と同じ。前の回に残った同じ名前の一時ファイルは上書きする。
    書けなかったとき（容量・権限など）は、一時ファイルを消してから例外を送り出す（置き場に残さない）。
    """
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def load_merged_glossary(paths: dict, use_global: bool) -> list[dict]:
    """グローバル → extra_glossary_dirs → プロジェクトの順でマージ（後ほど優先）。"""
    sources: list[tuple[str, Path | None]] = []
    if use_global and paths.get("global_glossary"):
        sources.append(("global", paths["global_glossary"]))
    for d in paths.get("extra_glossary_dirs", []):
        sources.append(("extra", d))
    sources.append(("project", paths.get("glossary")))

    layers: list[list[dict]] = []
    for label, d in sources:
        entries = load_glossary(d)
        if entries:
            layers.append(entries)
            print(f"[A] glossary {label:7s} {len(entries):3d} entries  {d}")
        elif d is not None and label != "global":
            print(f"[A] glossary {label:7s}   0 entries  {d}（無いか空）")
    merged = merge_glossaries(*layers) if layers else []
    if merged:
        print(f"[A] glossary merged {len(merged)} entries（同じ canonical は後の層が優先）")
    else:
        print("[A] glossary なし（辞書ディレクトリが無いか空）")
    return merged


def prompt_log_lines(plan: PromptPlan, note: str = "") -> list[str]:
    """initial_prompt の長さと、Whisper に届く語・届かない語のコンソールの行（#1 の C1）。辞書の語が無ければ []。

    note は、トークン数を数えられないときの理由（engine.load_prompt_encoder の 2 つ目の値）。
    """
    if not plan.words:
        return []
    lim = PROMPT_MAX_TOKENS
    if plan.tokens is None:
        return [f"  [WARN] initial_prompt のトークン数を数えられません（{note or '理由は不明'}）。"
                f"Whisper が使うのは後ろ {lim} トークンまでで、届く語の数は分かりません"]
    n_reach, n_miss = len(plan.reached or []), len(plan.missed or [])
    marked = f"（うち印 {plan.marked_reached}/{plan.marked}）" if plan.marked else ""
    lines: list[str] = []
    if plan.tokens > lim:  # 印の無い辞書で、0.5.0 と同じ文字列を渡す（Whisper が後ろ 223 だけを使う）
        lines.append(f"[A] initial_prompt {plan.tokens} トークン（Whisper が使うのは後ろ {lim}）、"
                     f"届く {n_reach}/{plan.words} 語{marked}、届かない {n_miss} 語")
        if n_miss:
            lines.append(f"  [WARN] initial_prompt が {lim} トークンを超えるため、前置きの文と {n_miss} 語は Whisper に"
                         "届きません（置き換えと校閲には使います）。"
                         '届けたい語には辞書で "priority": true を付けます（README）')
        else:
            lines.append(f"  [WARN] initial_prompt が {lim} トークンを超えるため、前置きの文の前の方は Whisper に届きません"
                         "（語は全部届きます）")
    elif plan.selected:
        lines.append(f"[A] initial_prompt {plan.tokens}/{lim} トークン（印の語を優先）、"
                     f"届く {n_reach}/{plan.words} 語{marked}、届かない {n_miss} 語")
        if n_miss:
            warn = (f"  [WARN] initial_prompt が {lim} トークンに収まらないため、{n_miss} 語は Whisper に予告しません"
                    "（置き換えと校閲には使います）")
            if (plan.marked_reached or 0) < plan.marked:
                # 「N 語も」は N が 1 のとき「1 語も入らない (= どれも入らない)」と読めるので「は」にする (PR #11 のレビュー 2 回目)
                warn += f"。印の語のうち {plan.marked - (plan.marked_reached or 0)} 語は入りません"
            lines.append(warn)
    else:
        lines.append(f"[A] initial_prompt {plan.tokens}/{lim} トークン、届く {n_reach}/{plan.words} 語{marked}")
    if n_miss:
        # 印の語なのに入らなかった語には「（印）」を付ける（印が多すぎるとき。SKILL.md §5）
        marks = plan.missed_marks or [False] * n_miss
        lines.append("  届かない語: " + "、".join(w + ("（印）" if m else "") for w, m in zip(plan.missed or [], marks)))
    return lines


def chunk_output_cap(prompt_tokens: int) -> int:
    """1 チャンクの出力の上限: min(448 - len(prompt) + 1, 448 // 2)（#3 §2 の 3b、#4 の U3）。

    len(prompt) は sot_prev + 使う prompt（後ろ 223 まで）+ sot・言語・task + no_timestamps。prompt が無ければ 4。
    """
    used = min(max(prompt_tokens, 0), PROMPT_MAX_TOKENS)
    prompt_len = (used + 5) if used else 4
    return min(WHISPER_MAX_LENGTH - prompt_len + 1, WHISPER_MAX_LENGTH // 2)


def chunk_log_lines(segments: list[dict], encode, prompt_tokens: int) -> list[str]:
    """チャンク（whisperx の 1 セグメント = VAD の 1 チャンク）ごとの出力のトークン数のコンソールの行。

    セグメントの文を同じトークナイザで数え直した数なので、Whisper が出したトークンの数とは少し違うことがある。
    """
    if not segments:
        return []
    counts = [len(encode(seg.get("text") or "")[0]) if (seg.get("text") or "") else 0 for seg in segments]
    top, cap = max(counts), chunk_output_cap(prompt_tokens)
    lines = [f"[A] チャンクの出力（数え直し）: 最大 {top} トークン / {len(counts)} チャンク（1 チャンクの上限 {cap}）"]
    over = sum(1 for c in counts if c > CHUNK_OUTPUT_WARN_TOKENS)
    if over:
        lines.append(f"  [WARN] 1 チャンクの出力が {CHUNK_OUTPUT_WARN_TOKENS} トークンを超えました（{over} チャンク、最大 {top}）。"
                     f"上限 {cap} に当たると、そのチャンクの末尾は黙って切れます（数は文を数え直した目安）")
    return lines


def process_one(
    audio: Path,
    *,
    cfg: dict,
    processed_dir: Path,
    out_dir: Path,
    entries: list[dict],
    with_timestamps: bool,
    keep: bool,
    device: str,
    use_proofread: bool,
    use_topic: bool = False,
) -> Path:
    print(f"\n=== {audio.name} ===")
    w = cfg["whisper"]
    gap_sec = float(cfg["formatting"]["paragraph_gap_sec"])

    # A: glossary を initial_prompt で予告。Whisper が使うのは後ろ 223 トークンまで（#1）。
    # prompt はモデルを読むときに渡すので、トークン数はその前に、モデルと同じ tokenizer.json で数える。
    # 印（"priority": true）の無い辞書は build_initial_prompt と同じ文字列（0.5.0 と同じ）で、変わるのはログだけ。
    encode, encode_note = load_prompt_encoder(w["model"])
    plan = plan_initial_prompt(entries, encode) if entries else None
    for line in prompt_log_lines(plan, encode_note) if plan else []:
        print(line)
    initial_prompt = plan.text if plan else ""

    result = transcribe_audio(
        audio,
        model_name=w["model"],
        compute_type=w["compute_type"],
        language=w["language"],
        batch_size=int(w["batch_size"]),
        chunk_size=int(w["chunk_size"]),
        device=device,
        initial_prompt=initial_prompt,
    )
    if encode is not None:  # チャンクの出力の数（#3 の保留: 1 チャンク 224 トークンの頭打ち）
        for line in chunk_log_lines(result["segments"], encode, (plan.tokens or 0) if plan else 0):
            print(line)

    # 置き換え (B) の前の文（Whisper の出力そのまま）。置き換えが 1 箇所以上あれば md の節に残す。
    # apply_replacements_to_segments は segments を書き換えず、新しい list を返す。
    pre_replacement_body = format_body(result["segments"], gap_sec, with_timestamps)

    # B: wrong_variants → canonical 置換
    replacement_count = 0
    if entries:
        result["segments"], log = apply_replacements_to_segments(result["segments"], entries)
        replacement_count = sum(r.get("count", 0) for r in log)
        print(f"[B] glossary 置換 {replacement_count} 箇所" if log else "[B] glossary 置換なし")

    raw_display_body = format_body(result["segments"], gap_sec, with_timestamps)
    plain_body = format_body(result["segments"], gap_sec, with_timestamps=False)

    # C: Claude による文脈校閲。失敗しても文字起こしは失わない。
    # proofread はモデル名 = 本文が校閲後。それ以外は false で、本文は校閲前（辞書の置き換え後）。
    proofread_status = "false"
    proofread_error: tuple[str, str] | None = None  # (理由の語, 1 行の補足)。校閲を試みて使わなかったときだけ
    main_body = raw_display_body
    raw_section = ""
    if use_proofread and not plain_body.strip():
        print("[C] 文字起こしが空のため、校閲しません")
    elif use_proofread:
        canonicals = [e["canonical"] for e in entries if e.get("canonical")]
        model = cfg.get("proofread", {}).get("model", "claude-sonnet-4-6")
        try:
            print(f"[C] Claude 校閲中（{model}）...")
            main_body = proofread_body(plain_body, canonicals, model=model)
            proofread_status = model
            raw_section = details_section(PRE_PROOFREAD_SUMMARY, raw_display_body)
            print("[C] 校閲完了")
        except Exception as e:
            code, detail = describe_failure(e)
            proofread_error = (code, one_line(detail))
            print(f"  [WARN] 校閲に失敗（{code}: {proofread_error[1]}）、校閲前の文字起こしを使用します: {e}")
    if replacement_count > 0:
        raw_section += details_section(PRE_REPLACEMENT_SUMMARY, pre_replacement_body)

    # D: 話題（設定で入れたときだけ。#14）。校閲後の本文（校閲していなければ文字起こし）を Claude が読み、1 行にする。
    # 失敗しても文字起こしは失わない（topic を書かず、理由を topic_error に残す）。
    topic: str | None = None
    topic_error: tuple[str, str] | None = None
    if use_topic:
        topic_text = main_body if proofread_status != "false" else plain_body
        if not topic_text.strip():
            print("[D] 文字起こしが空のため、話題を付けません")
        else:
            tcfg = cfg.get("topic", {})
            model = tcfg.get("model") or cfg.get("proofread", {}).get("model", "claude-sonnet-4-6")
            try:
                print(f"[D] 話題を付けています（{model}）...")
                topic = make_topic(topic_text, model=model, max_chars=int(tcfg.get("max_chars") or TOPIC_MAX_CHARS))
                print(f"[D] 話題: {topic}")
            except Exception as e:
                code, detail = describe_failure(e)
                topic_error = (code, one_line(detail))
                print(f"  [WARN] 話題を付けられません（{code}: {topic_error[1]}）。topic を書かずに保存します")

    # 録音の時刻は音声の更新時刻（今までの「録音日」と同じ出どころ）。recorded_at は時差つき（#14）
    recorded = datetime.datetime.fromtimestamp(audio.stat().st_mtime).astimezone()
    date = recorded.strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = unique_path(out_dir / f"{date}_{audio.stem}.md")

    # source と topic は JSON の文字列の形で書く（YAML の二重引用符の文字列としても読める）。
    # 音声の名前の `#`・`[`・`&` などで、ほかの道具が前付けを読み違えないため（0.7.0。#14）
    frontmatter = (
        "---\n"
        f"date: {date}\n"
        f"recorded_at: {recorded.isoformat(timespec='seconds')}\n"
        f"source: {json.dumps(audio.name, ensure_ascii=False)}\n"
        f"duration: {fmt_duration(result['duration'])}\n"
        f"engine: {result['engine']}\n"
        f"segments: {len(result['segments'])}\n"
        f"glossary_replacements: {replacement_count}\n"
        f"proofread: {proofread_status}\n"
    )
    if proofread_error:
        # detail は JSON の文字列の形（YAML の二重引用符の文字列としても読める）
        frontmatter += (
            f"proofread_error: {proofread_error[0]}\n"
            f"proofread_error_detail: {json.dumps(proofread_error[1], ensure_ascii=False)}\n"
        )
    if topic:
        frontmatter += f"topic: {json.dumps(topic, ensure_ascii=False)}\n"
    if topic_error:
        frontmatter += (
            f"topic_error: {topic_error[0]}\n"
            f"topic_error_detail: {json.dumps(topic_error[1], ensure_ascii=False)}\n"
        )
    frontmatter += "---\n\n"
    write_atomic(out_path, frontmatter + f"# ボイスメモ {date}\n\n" + main_body + "\n" + raw_section)
    print(f"  -> {out_path}")

    if not keep:
        try:
            processed_dir.mkdir(parents=True, exist_ok=True)
            dest = unique_path(processed_dir / audio.name)
            shutil.move(str(audio), str(dest))
            print(f"  moved: {dest}")
        except Exception as e:
            print(f"  [WARN] processed への移動に失敗: {e}")

    return out_path


def on_complete_settings(cfg: dict) -> tuple[list[str], float] | None:
    """config の on_complete を確かめて (command, 上限の秒) を返す。command が無いか空なら None（起こさない）。

    形の誤りは ValueError（main が「設定の誤り」として、文字起こしの前に止める。output_root と同じ。#16）。
    """
    oc = cfg.get("on_complete")
    if oc is None:
        return None
    if not isinstance(oc, dict):
        raise ValueError('on_complete は {"command": [...], "timeout_sec": 秒} の形にします')
    command = oc.get("command")
    if command is None or command == []:
        return None
    if not isinstance(command, list) or not all(isinstance(a, str) and a.strip() for a in command):
        raise ValueError("on_complete.command は、空でない文字列の配列にします（シェルを通さないので、1 つの文字列には書けません）")
    timeout = oc.get("timeout_sec", ON_COMPLETE_DEFAULT_TIMEOUT_SEC)
    # 比べるだけで確かめる（NaN・Infinity・大きすぎる整数も、float にせずに誤りになる）
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not (0 < timeout <= ON_COMPLETE_MAX_TIMEOUT_SEC):
        raise ValueError(f"on_complete.timeout_sec は {ON_COMPLETE_MAX_TIMEOUT_SEC} までの正の数（秒）にします")
    return list(command), float(timeout)


def _find_on_path(name: str) -> str | None:
    """区切りの無い名前を、PATH のフォルダだけで引く（PR #17 のレビュー 1 回目の推奨 2）。

    Windows の CreateProcess は、PATH より先に、起こした側の exe のフォルダとカレントを探すので、それに任せない。
    Windows では、PATH のフォルダごとに、名前のまま（拡張子があるとき）と PATHEXT の拡張子を足したものを試す
    （npm → npm.cmd のように見つかったものは、run_on_complete が理由を添えて断る）。相対のフォルダ（. など）は見ない。
    """
    if os.name != "nt":
        return shutil.which(name)
    # 拡張子は小文字で試す（大文字小文字を区別しない。行に出る名前が "node.EXE" にならないように）
    exts = [e.lower() for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(os.pathsep) if e]
    names = ([name] if os.path.splitext(name)[1] else []) + [name + e for e in exts]
    for d in os.environ.get("PATH", "").split(os.pathsep):
        d = d.strip().strip('"')
        if not d or not os.path.isabs(d):
            continue
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    return None


def on_complete_argv(command: list[str], md_path: Path, project_root: Path) -> list[str]:
    """起こす引数の列: command + [md の絶対パス]。

    command[0] が区切りを含む相対パスなら、プロジェクトルート基準にする。区切りの無い名前は PATH だけで引く
    （見つからなければ FileNotFoundError）。
    md のパスは os.path.abspath（ジャンクションをたどらない。--out-dir が相対パスでも絶対パスになる）。
    """
    exe = command[0]
    if "/" in exe or "\\" in exe:
        if not os.path.isabs(exe):
            exe = str(project_root / exe)
    else:
        found = _find_on_path(exe)
        if found is None:
            raise FileNotFoundError(f"PATH に見つかりません: {exe}")
        exe = found
    return [exe, *command[1:], os.path.abspath(md_path)]


def _unsafe_program(exe: str) -> bool:
    """Windows で起こさないもの: cmd と、実行ファイル（.exe・.com・拡張子なし）でないもの（.bat・.cmd など）。

    Windows は名前の末尾の空白と . を落として開くので、落としてから確かめる（"hook.cmd " も .cmd）。
    """
    if os.name != "nt":
        return False
    base = os.path.basename(exe.rstrip(" .")).lower()
    return base in _SHELL_NAMES or os.path.splitext(base)[1] not in _SAFE_EXTS


def _decode_output(data: bytes) -> str:
    """子の出力を読む。UTF-8 で読めなければ、Windows の既定のコードページ（日本語の Windows なら cp932）で読む。

    Windows では mbcs（ANSI コードページ）を使う。Python の UTF-8 モード（PYTHONUTF8=1）でも変わらない。
    """
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        enc = "mbcs" if os.name == "nt" else (locale.getpreferredencoding(False) or "utf-8")
        return data.decode(enc, errors="replace")


def _read_last_line(f) -> str:
    """一時ファイルに受けた出力の、最後の空でない行（1 行）。

    色などの印を外し、印字できない字を落とし（タブなどの空白は空白にする）、空白をまとめて切る。
    """
    size = f.seek(0, os.SEEK_END)
    start = max(0, size - ON_COMPLETE_READ_TAIL_BYTES)
    f.seek(start)
    data = f.read()
    if start and b"\n" in data:  # 途中から読んだときは、切れた最初の行を捨てる（UTF-8 の字の途中で切れることもある）
        data = data[data.index(b"\n") + 1:]
    # 行ごとに読む（UTF-8 の行とそうでない行が混じっても、最後の行を読み違えない。PR #17 のレビュー 1 回目の任意 2）
    for raw in reversed(data.splitlines()):
        line = _ANSI_CSI_RE.sub("", _decode_output(raw))
        line = "".join(ch if ch.isprintable() else (" " if ch.isspace() else "") for ch in line)
        if line.strip():
            return one_line(line, ON_COMPLETE_TAIL_MAX_CHARS)
    return ""


def _read_last_line_or_empty(f) -> str:
    """_read_last_line と同じ。読めなければ空（要点を付けない。標準出力と標準エラー出力を 1 つずつ受ける。
    PR #19 のレビュー 2 回目の任意 1）。"""
    try:
        return _read_last_line(f)
    except Exception:
        return ""


def _stop_process_tree(proc: subprocess.Popen) -> None:
    """起こしたプロセスとその子孫を止める。Windows は taskkill /T /F、止められなければ（Windows 以外も）kill()。

    例外を送り出さない（止められなくても戻る。run_on_complete の「例外を送り出さない」を守る。PR #19 のレビュー 2 回目の必須 1）。
    """
    if os.name == "nt":
        # PATH やカレントの taskkill.exe を拾わないように、System32 のものを名指す
        taskkill = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")
        try:
            subprocess.run([taskkill, "/T", "/F", "/PID", str(proc.pid)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        except Exception:
            pass
    try:
        if proc.poll() is None:
            proc.kill()
    except Exception:
        pass
    try:
        proc.wait(timeout=30)
    except Exception:  # 止まらない（TimeoutExpired）・待てない（OSError など）
        pass


@dataclass(frozen=True)
class OnCompleteResult:
    """終わりの知らせのコマンドを 1 回走らせた結果（run_on_complete_result の戻り。#18）。

    - line: ランチャーに出す 1 行（run_on_complete の戻りと同じ）。成功は ON_COMPLETE_OK_HEAD で始まり、
      それ以外（失敗・時間切れ・起こせない）は ON_COMPLETE_WARN_HEAD で始まる
    - started: 起こせたか（起こした後にキットの側で例外が出ても True。False なら、コマンドは走っていない）
    - timed_out: 上限を過ぎて止めたか
    - exit_code: 終わりの値（起こせない・時間切れ・終わりを待つ途中でキットの側に例外が出たときは None）
    - ok: 通ったか（exit_code が 0）。成否は ok で読む。値そのものの真偽（bool(result)）は失敗でも真
      （__bool__ を定めると、`if result:` が ok と started のどちらの意味か読み違えやすいため、定めない）
    """
    line: str
    started: bool
    timed_out: bool
    exit_code: int | None

    @property
    def ok(self) -> bool:
        """コマンドが通ったか（終わりの値が 0）。"""
        return self.exit_code == 0


def run_on_complete_result(md_path: Path, command: list[str], timeout: float, project_root: Path) -> OnCompleteResult:
    """書いた md のパスを最後の引数にして command を起こし、成否を値で返す（#16・#18）。

    シェルを通さない。cwd はプロジェクトルート。stdin は空。stdout・stderr は一時ファイルに受ける
    （pipe だと、時間切れの後に孫が pipe を持っていたとき返らない。#4 の U5）。
    上限を過ぎたら、起こしたプロセスとその子孫を止める。何が起きても例外を送り出さない（文字起こしを失敗にしない）。
    """
    def cannot_start(reason: str) -> OnCompleteResult:
        line = (f"{ON_COMPLETE_WARN_HEAD}{ON_COMPLETE_CANNOT_START}（{one_line(reason, ON_COMPLETE_TAIL_MAX_CHARS)}）。"
                f"{ON_COMPLETE_DONE_NOTE}")
        return OnCompleteResult(line=line, started=False, timed_out=False, exit_code=None)

    # 起こした後に例外が出ても、started は True のまま（子は走った。PR #19 のレビュー 1 回目の推奨 1）
    proc: subprocess.Popen | None = None
    code: int | None = None
    timed_out = False
    tail_out = tail_err = ""
    try:
        argv = on_complete_argv(command, md_path, project_root)
        if _unsafe_program(argv[0]):
            found = os.path.basename(argv[0])
            bare = "/" not in command[0] and "\\" not in command[0]
            if bare and found.rstrip(" .").lower() not in _SHELL_NAMES:
                # 名前だけで書いて、PATH で .bat・.cmd が先に見つかったとき（例: python.bat の shim）。
                # 書いた名前そのものを勧めない（PR #17 のレビュー 2 回目の任意 1）
                return cannot_start(f"{command[0]} を PATH で引くと {found} になるため起こしません。"
                                    f"{ON_COMPLETE_EXE_ABSOLUTE}")
            return cannot_start(f"{found} は起こしません。{ON_COMPLETE_EXE_ONLY}")
        with tempfile.TemporaryFile() as out, tempfile.TemporaryFile() as err:
            try:
                proc = subprocess.Popen(argv, cwd=str(project_root), stdin=subprocess.DEVNULL, stdout=out, stderr=err)
            except OSError as e:  # 無いコマンド・権限など
                return cannot_start(f"{type(e).__name__}: {e}")
            try:
                code = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True  # 止める前に立てる（止める所で何が起きても、時間切れの行にする。PR #19 のレビュー 2 回目の必須 1）
                _stop_process_tree(proc)
            tail_out = _read_last_line_or_empty(out)
            tail_err = _read_last_line_or_empty(err)
    except Exception as e:  # 一時ファイルが作れない・閉じられないなど。文字起こしは失敗にしない
        if proc is None:
            return cannot_start(f"{type(e).__name__}: {e}")
        if code is None and not timed_out:
            # 起こした後、終わりを待つ途中の例外。子が残らないように止め、終わりの値なしの失敗として返す
            try:
                _stop_process_tree(proc)
            except Exception:  # 止められなくても行を返す（例外を送り出さない）
                pass
            line = (f"{ON_COMPLETE_WARN_HEAD}{ON_COMPLETE_FAILED}"
                    f"（{one_line(f'{type(e).__name__}: {e}', ON_COMPLETE_TAIL_MAX_CHARS)}）。{ON_COMPLETE_DONE_NOTE}")
            return OnCompleteResult(line=line, started=True, timed_out=False, exit_code=None)
        # 終わりの値（か時間切れ）は分かっている。読めた分の要点で、下の行にする

    if timed_out:
        tail = tail_err or tail_out
        line = (f"{ON_COMPLETE_WARN_HEAD}終わりの知らせのコマンドが {timeout:g} {ON_COMPLETE_TIMED_OUT}"
                + (f"（{tail}）" if tail else "") + f"。{ON_COMPLETE_DONE_NOTE}")
        return OnCompleteResult(line=line, started=True, timed_out=True, exit_code=None)
    if code != 0:
        tail = tail_err or tail_out
        line = (f"{ON_COMPLETE_WARN_HEAD}{ON_COMPLETE_FAILED}（exit {code}" + (f": {tail}" if tail else "")
                + f"）。{ON_COMPLETE_DONE_NOTE}")
        return OnCompleteResult(line=line, started=True, timed_out=False, exit_code=code)
    line = ON_COMPLETE_OK_HEAD + (f"（{tail_out}）" if tail_out else "")
    return OnCompleteResult(line=line, started=True, timed_out=False, exit_code=0)


def run_on_complete(md_path: Path, command: list[str], timeout: float, project_root: Path) -> str:
    """書いた md のパスを最後の引数にして command を起こし、ランチャーに出す 1 行を返す（#16）。

    動きは run_on_complete_result と同じで、その line を返す。成功の行は ON_COMPLETE_OK_HEAD で始まり、
    それ以外は ON_COMPLETE_WARN_HEAD で始まる（ほかのスキルとの約束。tests/test_kit_api.py。#18）。
    """
    return run_on_complete_result(md_path, command, timeout, project_root).line


def main() -> int:
    parser = argparse.ArgumentParser(description="ボイスメモ ローカル文字起こし")
    parser.add_argument("input", nargs="?", default=None,
                        help="音声ファイル or フォルダ（省略時は <project>/voice_memos/inbox）")
    parser.add_argument("--project", type=Path, default=None,
                        help="プロジェクトルート（省略時はカレントディレクトリ）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="出力先（省略時は <project>/voice_memos/transcripts。config の output_root の根があればその下）")
    parser.add_argument("--no-glossary", action="store_true", help="固有名詞辞書を使わない")
    parser.add_argument("--no-global", action="store_true",
                        help="グローバル層（~/voice_memos）の inbox・辞書を使わない")
    parser.add_argument("--timestamps", action="store_true", help="各段落頭に [mm:ss] を付与")
    parser.add_argument("--keep", action="store_true", help="処理後も音声を移動しない")
    parser.add_argument("--no-proofread", action="store_true", help="Claude 校閲をスキップ")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None,
                        help="推論デバイス（省略時は config の whisper.device、既定 cuda）")
    parser.add_argument("--no-topic", action="store_true", help="話題（topic）を付けない（config で入れたとき）")
    parser.add_argument("--no-on-complete", action="store_true",
                        help="終わりの知らせ（config の on_complete.command）を起こさない")
    parser.add_argument("--list", action="store_true",
                        help="対象の音声と置き場を出すだけ（文字起こし・移動・フォルダの作成をしない）")
    args = parser.parse_args()

    project_root = (args.project or Path.cwd()).resolve()
    cfg = load_config(project_root)
    device = args.device or cfg.get("whisper", {}).get("device", "cuda")
    try:
        paths = resolve_paths(cfg, project_root)
        on_complete = on_complete_settings(cfg)  # 形の誤りは、--no-on-complete・--list でも止める
    except ValueError as e:
        print(f"ERROR: 設定の誤り: {e}", file=sys.stderr)
        return 1
    out_dir = args.out_dir or paths["transcripts"]
    use_global = not args.no_global

    # 置き場（SKILL.md §1・§3 はこの行を読む。決め打ちの voice_memos/transcripts を見ない）。
    # 各行の = の後ろはパスだけ（注を付けない）。根を使ったときだけ output_root= の行を足す
    print(f"transcripts={out_dir}")
    print(f"processed  ={paths['processed']}")
    if paths.get("output_root"):
        print(f"output_root={paths['output_root']}")
    elif paths.get("output_root_missing"):
        print(OUTPUT_ROOT_MISSING_WARN_OUT_DIR if args.out_dir else OUTPUT_ROOT_MISSING_WARN)

    if args.no_on_complete:
        on_complete = None
    # 設定の行。--list でも出す（待つ時間を前もって見積もれる。PR #17 のレビュー 1 回目の推奨 1）
    on_complete_line = (f"[E] 終わりの知らせ: md を書くたびに走らせます（上限 {on_complete[1]:g} 秒）: "
                        f"{subprocess.list2cmdline(on_complete[0])}") if on_complete else ""

    inputs = collect_inputs(args.input, paths, use_global)
    if args.list:
        for audio, processed_dir in inputs:
            print(f"  {audio}  (退避: {processed_dir})")
        print(f"対象 {len(inputs)} 件")
        if on_complete_line:
            print(on_complete_line)
        if not inputs and not args.input:
            print(f"  音声の置き場: project {paths['inbox']}"
                  + (f" / global {paths['global_inbox']}" if use_global and paths.get("global_inbox") else ""))
        return 0
    if not inputs:
        if args.input:
            print(f"ERROR: 対象が見つかりません: {args.input}", file=sys.stderr)
        else:
            print("処理対象がありません。音声を置いてください:")
            print(f"  project: {paths['inbox']}")
            if use_global and paths.get("global_inbox"):
                print(f"  global : {paths['global_inbox']}")
        return 1
    missing = [f for f, _ in inputs if not f.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: ファイルが存在しません: {f}", file=sys.stderr)
        return 1

    use_proofread = bool(cfg.get("proofread", {}).get("enabled", True)) and not args.no_proofread
    use_topic = bool(cfg.get("topic", {}).get("enabled", False)) and not args.no_topic
    print(f"project={project_root}")
    if use_global and paths.get("global_root"):
        print(f"global ={paths['global_root']}")
    n_proj = sum(1 for _, d in inputs if d == paths["processed"])
    print(f"対象 {len(inputs)} 件（project {n_proj} / global {len(inputs) - n_proj}） / "
          f"device={device} / glossary={not args.no_glossary} / proofread={use_proofread} / topic={use_topic}")
    if on_complete_line:
        print(on_complete_line)

    entries = [] if args.no_glossary else load_merged_glossary(paths, use_global)

    done = []
    for audio, processed_dir in inputs:
        try:
            md_path = process_one(
                audio, cfg=cfg, processed_dir=processed_dir, out_dir=out_dir,
                entries=entries, with_timestamps=args.timestamps,
                keep=args.keep, device=device, use_proofread=use_proofread, use_topic=use_topic,
            )
        except Exception as e:
            print(f"[ERROR] {audio.name} の処理に失敗: {e}", file=sys.stderr)
            continue
        done.append(md_path)
        # md を最終の名前で書き、音声を退かした後（#16）。成否で文字起こしを失敗にしない
        if on_complete:
            print(run_on_complete(md_path, on_complete[0], on_complete[1], project_root))

    print(f"\n完了: {len(done)}/{len(inputs)} 件")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
