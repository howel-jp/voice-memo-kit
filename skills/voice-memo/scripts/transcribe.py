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

設定: config.default.json（キット既定）に <project>/voice_memos/config.json を重ねて上書き。
辞書: paths.glossary（明示）→ paths.glossary_candidates のうち最初に存在するディレクトリ。

出力: <project>/voice_memos/transcripts/<録音日>_<元ファイル名>.md
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

# Windows cp932 で encode できない文字でのクラッシュを防ぐ
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from engine import transcribe_audio
from glossary import apply_replacements_to_segments, build_initial_prompt, load_glossary
from proofread import proofread_body

SKILL_DIR = Path(__file__).resolve().parents[1]
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".mp4"}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(project_root: Path) -> dict:
    cfg = json.loads((SKILL_DIR / "config.default.json").read_text(encoding="utf-8"))
    local = project_root / "voice_memos" / "config.json"
    if local.exists():
        cfg = _deep_merge(cfg, json.loads(local.read_text(encoding="utf-8")))
        print(f"config: {local}")
    return cfg


def resolve_paths(cfg: dict, project_root: Path) -> dict:
    p = cfg["paths"]
    glossary: Path | None = None
    if p.get("glossary"):
        glossary = project_root / p["glossary"]
    else:
        for cand in p.get("glossary_candidates", []):
            if (project_root / cand).is_dir():
                glossary = project_root / cand
                break
    return {
        "glossary": glossary,
        "inbox": project_root / p["inbox"],
        "transcripts": project_root / p["transcripts"],
        "processed": project_root / p["processed"],
    }


def collect_inputs(arg_input: str | None, inbox: Path) -> list[Path]:
    if arg_input:
        target = Path(arg_input)
        if not target.is_absolute():
            target = (Path.cwd() / target).resolve()
        if target.is_dir():
            return sorted(f for f in target.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        return [target]
    if not inbox.exists():
        return []
    return sorted(f for f in inbox.iterdir() if f.suffix.lower() in AUDIO_EXTS)


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


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        cand = path.parent / f"{path.stem}_{i}{path.suffix}"
        if not cand.exists():
            return cand
        i += 1


def process_one(
    audio: Path,
    *,
    cfg: dict,
    paths: dict,
    out_dir: Path,
    use_glossary: bool,
    with_timestamps: bool,
    keep: bool,
    device: str,
    use_proofread: bool,
) -> Path:
    print(f"\n=== {audio.name} ===")
    w = cfg["whisper"]
    gap_sec = float(cfg["formatting"]["paragraph_gap_sec"])

    # A: glossary を initial_prompt で予告
    entries: list[dict] = []
    initial_prompt = ""
    if use_glossary:
        entries = load_glossary(paths["glossary"])
        if entries:
            initial_prompt = build_initial_prompt(entries)
            print(f"[A] glossary {len(entries)} entries ({paths['glossary']})")
        else:
            print("[A] glossary なし（辞書ディレクトリが無いか空）")

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

    # B: wrong_variants → canonical 置換
    replacement_count = 0
    if entries:
        result["segments"], log = apply_replacements_to_segments(result["segments"], entries)
        replacement_count = sum(r.get("count", 0) for r in log)
        print(f"[B] glossary 置換 {replacement_count} 箇所" if log else "[B] glossary 置換なし")

    raw_display_body = format_body(result["segments"], gap_sec, with_timestamps)
    plain_body = format_body(result["segments"], gap_sec, with_timestamps=False)

    # C: Claude による文脈校閲。失敗しても文字起こしは失わない。
    proofread_status = "false"
    main_body = raw_display_body
    raw_section = ""
    if use_proofread:
        canonicals = [e["canonical"] for e in entries if e.get("canonical")]
        model = cfg.get("proofread", {}).get("model", "claude-sonnet-4-6")
        try:
            print(f"[C] Claude 校閲中（{model}）...")
            main_body = proofread_body(plain_body, canonicals, model=model)
            proofread_status = model
            raw_section = (
                "\n\n<details>\n<summary>校閲前（文字起こし生データ）</summary>\n\n"
                + raw_display_body
                + "\n\n</details>\n"
            )
            print("[C] 校閲完了")
        except Exception as e:
            print(f"  [WARN] 校閲に失敗、生の文字起こしを使用します: {e}")

    date = datetime.datetime.fromtimestamp(audio.stat().st_mtime).strftime("%Y-%m-%d")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = unique_path(out_dir / f"{date}_{audio.stem}.md")

    frontmatter = (
        "---\n"
        f"date: {date}\n"
        f"source: {audio.name}\n"
        f"duration: {fmt_duration(result['duration'])}\n"
        f"engine: {result['engine']}\n"
        f"segments: {len(result['segments'])}\n"
        f"glossary_replacements: {replacement_count}\n"
        f"proofread: {proofread_status}\n"
        "---\n\n"
    )
    out_path.write_text(
        frontmatter + f"# ボイスメモ {date}\n\n" + main_body + "\n" + raw_section,
        encoding="utf-8",
    )
    print(f"  -> {out_path}")

    if not keep:
        try:
            paths["processed"].mkdir(parents=True, exist_ok=True)
            dest = unique_path(paths["processed"] / audio.name)
            shutil.move(str(audio), str(dest))
            print(f"  moved: {dest}")
        except Exception as e:
            print(f"  [WARN] processed への移動に失敗: {e}")

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="ボイスメモ ローカル文字起こし")
    parser.add_argument("input", nargs="?", default=None,
                        help="音声ファイル or フォルダ（省略時は <project>/voice_memos/inbox）")
    parser.add_argument("--project", type=Path, default=None,
                        help="プロジェクトルート（省略時はカレントディレクトリ）")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="出力先（省略時は <project>/voice_memos/transcripts）")
    parser.add_argument("--no-glossary", action="store_true", help="固有名詞辞書を使わない")
    parser.add_argument("--timestamps", action="store_true", help="各段落頭に [mm:ss] を付与")
    parser.add_argument("--keep", action="store_true", help="処理後も音声を移動しない")
    parser.add_argument("--no-proofread", action="store_true", help="Claude 校閲をスキップ")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    project_root = (args.project or Path.cwd()).resolve()
    cfg = load_config(project_root)
    paths = resolve_paths(cfg, project_root)
    out_dir = args.out_dir or paths["transcripts"]

    inputs = collect_inputs(args.input, paths["inbox"])
    if not inputs:
        if args.input:
            print(f"ERROR: 対象が見つかりません: {args.input}", file=sys.stderr)
        else:
            print(f"処理対象がありません。音声を置いてください: {paths['inbox']}")
        return 1
    missing = [f for f in inputs if not f.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: ファイルが存在しません: {f}", file=sys.stderr)
        return 1

    use_proofread = bool(cfg.get("proofread", {}).get("enabled", True)) and not args.no_proofread
    print(f"project={project_root}")
    print(f"対象 {len(inputs)} 件 / device={args.device} / "
          f"glossary={not args.no_glossary} / proofread={use_proofread}")
    done = []
    for audio in inputs:
        try:
            done.append(process_one(
                audio, cfg=cfg, paths=paths, out_dir=out_dir,
                use_glossary=not args.no_glossary, with_timestamps=args.timestamps,
                keep=args.keep, device=args.device, use_proofread=use_proofread,
            ))
        except Exception as e:
            print(f"[ERROR] {audio.name} の処理に失敗: {e}", file=sys.stderr)

    print(f"\n完了: {len(done)}/{len(inputs)} 件")
    return 0 if done else 1


if __name__ == "__main__":
    sys.exit(main())
