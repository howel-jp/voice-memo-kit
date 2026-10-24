"""手元の transcript (キットの出力 md) から、長さの数だけを出す (#4 の U3・U4 の見込み)。

本文・ファイル名・パスは出さない。出すのは、件数、長さ (秒)、字数、字数/分、セグメント数/分、
トークン数/分 (--tokens のとき) と、そこから見込む 45 分の字数だけ。

    python tools/measure/transcript_stats.py --root ~/voice_memos --root "C:/Projects/*/voice_memos/transcripts" --tokens

数える本文: 校閲前の `<details>` (キットが校閲に成功したときに付ける。置き換え後・校閲前の文) があればそれを、
無ければ本文 (校閲していない文か、校閲後の文) を数える。`[mm:ss] ` の印と空白・改行は数えない。
import しただけでは何もしない。
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kit  # noqa: E402

_TS_RE = re.compile(r"^\[\d{2,}:\d{2}\] ", re.MULTILINE)
_WS_RE = re.compile(r"\s+")
_DETAILS_RE = re.compile(r"<details>\s*<summary>校閲前[^<]*</summary>(.*?)</details>", re.DOTALL)


def parse_transcript(text: str) -> dict | None:
    """キットの md を読み、数えるための値を返す (本文そのものは呼び出し側に返さない用途で使う)。"""
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---", 4)
    if end < 0:
        return None
    fm: dict[str, str] = {}
    for line in text[4:end].splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            fm[k.strip()] = v.strip()
    dur = fm.get("duration", "")
    m = re.fullmatch(r"(\d+):(\d{2})", dur)
    if not m:
        return None
    seconds = int(m.group(1)) * 60 + int(m.group(2))
    body = text[end + 4:]
    d = _DETAILS_RE.search(body)
    if d:
        counted, source = d.group(1), "details"
    else:
        lines = body.strip().splitlines()
        if lines and lines[0].startswith("# "):
            lines = lines[1:]
        counted, source = "\n".join(lines), "body"
    counted = _TS_RE.sub("", counted)
    try:
        segments = int(fm.get("segments", ""))
    except ValueError:
        segments = None
    return {
        "seconds": seconds,
        "segments": segments,
        "proofread": fm.get("proofread", "false") != "false",
        "source": source,
        "text": counted,
    }


def count_chars(text: str) -> int:
    return len(_WS_RE.sub("", text))


def count_chars_bytes(text: str) -> int:
    """字数の 2 つ目の数え方: UTF-32 に直したバイト数 / 4 (空白は除く)。"""
    return len(_WS_RE.sub("", text).encode("utf-32-le")) // 4


def collect_files(roots: list[str]) -> list[Path]:
    files: set[Path] = set()
    for r in roots:
        for d in glob.glob(str(Path(r).expanduser())):
            p = Path(d)
            if p.is_file() and p.suffix == ".md":
                files.add(p.resolve())
            elif p.is_dir():
                files.update(f.resolve() for f in p.rglob("*.md"))
    return sorted(files, key=lambda p: str(p).lower())


def stats(files: list[Path], encode=None, minutes_for_estimate: float = 45.0) -> dict:
    """数だけの行と集計を返す。行のラベルは T1.. (ファイル名は入れない)。"""
    rows = []
    skipped = 0
    for i, f in enumerate(files, 1):
        try:
            parsed = parse_transcript(f.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError):
            parsed = None
        if not parsed or parsed["seconds"] <= 0:
            skipped += 1
            continue
        chars = count_chars(parsed["text"])
        minutes = parsed["seconds"] / 60.0
        row = {
            "label": f"T{len(rows) + 1}",
            "seconds": parsed["seconds"],
            "segments": parsed["segments"],
            "proofread": parsed["proofread"],
            "source": parsed["source"],
            "chars": chars,
            "chars_check": count_chars_bytes(parsed["text"]),
            "chars_per_min": round(chars / minutes, 1),
            "segments_per_min": round(parsed["segments"] / minutes, 2) if parsed["segments"] else None,
            "chars_per_segment": round(chars / parsed["segments"], 1) if parsed["segments"] else None,
        }
        if encode is not None:
            ids, _ = encode(parsed["text"])
            row["tokens"] = len(ids)
            row["tokens_per_30s"] = round(len(ids) / (parsed["seconds"] / 30.0), 1)
        rows.append(row)
    rates = [r["chars_per_min"] for r in rows]
    total_chars = sum(r["chars"] for r in rows)
    total_min = sum(r["seconds"] for r in rows) / 60.0
    summary = {
        "files": len(rows),
        "skipped": skipped,
        "total_minutes": round(total_min, 1),
        "chars_per_min_median": round(_kit.median(rates), 1),
        "chars_per_min_max": round(max(rates), 1) if rates else 0,
        "chars_per_min_weighted": round(total_chars / total_min, 1) if total_min else 0,
        "chars_per_segment_median": round(_kit.median(
            [r["chars_per_segment"] for r in rows if r["chars_per_segment"]]), 1),
        "segments_per_min_median": round(_kit.median(
            [r["segments_per_min"] for r in rows if r["segments_per_min"]]), 2),
    }
    summary[f"est_chars_{int(minutes_for_estimate)}min_median"] = round(
        summary["chars_per_min_median"] * minutes_for_estimate)
    summary[f"est_chars_{int(minutes_for_estimate)}min_max"] = round(
        summary["chars_per_min_max"] * minutes_for_estimate)
    if encode is not None and rows:
        summary["tokens_per_30s_median"] = round(_kit.median([r["tokens_per_30s"] for r in rows]), 1)
        summary["tokens_per_30s_max"] = max(r["tokens_per_30s"] for r in rows)
    return {"rows": rows, "summary": summary}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="transcript の長さの数だけを出す (#4)")
    ap.add_argument("--root", action="append", required=True,
                    help="md を探す場所 (glob 可。ディレクトリなら下を再帰で探す)。複数可")
    ap.add_argument("--tokens", action="store_true", help="Whisper のトークン数も数える")
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--minutes", type=float, default=45.0, help="字数を見込む長さ (分)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    _kit.utf8_stdout()

    encode = None
    if args.tokens:
        encode = _kit.hf_encoder(_kit.load_hf_tokenizer(args.tokenizer))
    res = stats(collect_files(args.root), encode, args.minutes)
    cols = ["label", "seconds", "segments", "proofread", "source", "chars", "chars_check",
            "chars_per_min", "segments_per_min", "chars_per_segment"] + (["tokens", "tokens_per_30s"] if encode else [])
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in res["rows"]:
        print("| " + " | ".join(str(r.get(c)) for c in cols) + " |")
    print("summary: " + json.dumps(res["summary"], ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
