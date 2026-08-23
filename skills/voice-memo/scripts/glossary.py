"""固有名詞辞書（glossary）ユーティリティ。

プロジェクトごとの JSON 辞書ディレクトリを受け取り、
- A. initial_prompt 用文字列を組み立てる（WhisperX に固有名詞を予告）
- B. wrong_variants → canonical の置換を segments に適用する
を提供する。辞書のデータはキットではなく各プロジェクト側に置く。

JSON の書式:
{
  "title": "<カテゴリ名>",
  "entries": [
    {"canonical": "<正式表記>", "wrong_variants": ["<誤認識パターン>", ...], "notes": "..."}
  ]
}
"""
from __future__ import annotations

import json
from pathlib import Path


def load_glossary(glossary_dir: Path | None) -> list[dict]:
    """ディレクトリ内の全 JSON を読み込み、フラットな entry リストを返す。"""
    entries: list[dict] = []
    if glossary_dir is None or not glossary_dir.exists():
        return entries
    for json_file in sorted(glossary_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"[glossary] WARN: {json_file.name} のパース失敗: {e}")
            continue
        for entry in data.get("entries", []):
            entry["source"] = json_file.stem
            entries.append(entry)
    return entries


def build_initial_prompt(entries: list[dict]) -> str:
    """WhisperX の initial_prompt 用文字列を組み立てる。

    命令文（「正確に書き起こしてください」等）は文字起こし結果に hallucination として
    混入する事故が観測されているため、中立的な体言止めにする。
    """
    canonicals = [e["canonical"] for e in entries if e.get("canonical")]
    if not canonicals:
        return ""
    return "以下は本音声で言及される固有名詞の一覧です: " + "、".join(canonicals) + "。"


def _collides_with_canonical(wrong: str, next_char: str, canonicals: set[str]) -> bool:
    """wrong の直後の文字を足すと、より長い canonical の接頭辞になる場合 True。

    短い wrong_variant が正しい語の部分文字列であるケースで、正しい語まで誤って
    置換してしまうのを防ぐ 1 文字先読みガード。保守側（置換しない側）に倒す。
    """
    if not next_char:
        return False
    probe = wrong + next_char
    return any(len(c) > len(wrong) and c.startswith(probe) for c in canonicals)


def guarded_str_replace(
    text: str, wrong: str, canonical: str, canonicals: set[str]
) -> tuple[str, int]:
    """部分文字列衝突ガード付きの文字列置換（全出現を走査）。"""
    out: list[str] = []
    i = 0
    n = len(text)
    cnt = 0
    while i < n:
        if text.startswith(wrong, i):
            nxt = text[i + len(wrong)] if i + len(wrong) < n else ""
            if _collides_with_canonical(wrong, nxt, canonicals):
                out.append(text[i])
                i += 1
                continue
            out.append(canonical)
            i += len(wrong)
            cnt += 1
        else:
            out.append(text[i])
            i += 1
    return "".join(out), cnt


def apply_replacements_to_segments(
    segments: list[dict], entries: list[dict]
) -> tuple[list[dict], list[dict]]:
    """segments の text に wrong_variants → canonical の置換を適用する。

    ボイスメモ用途では align を省略するため words 配列は扱わない（text のみ）。

    Returns:
        (置換後 segments, 適用ログ)
    """
    canonicals = {e["canonical"] for e in entries if e.get("canonical")}
    log: list[dict] = []
    new_segments: list[dict] = []
    for seg in segments:
        new_seg = dict(seg)
        for entry in entries:
            canonical = entry.get("canonical")
            if not canonical:
                continue
            for wrong in entry.get("wrong_variants", []) or []:
                if not wrong:
                    continue
                text = new_seg.get("text") or ""
                new_text, count = guarded_str_replace(text, wrong, canonical, canonicals)
                if count > 0:
                    new_seg["text"] = new_text
                    log.append({
                        "wrong": wrong,
                        "canonical": canonical,
                        "count": count,
                        "source": entry.get("source", ""),
                    })
        new_segments.append(new_seg)
    return new_segments, log
