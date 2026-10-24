"""固有名詞辞書（glossary）ユーティリティ。

プロジェクトごとの JSON 辞書ディレクトリを受け取り、
- A. initial_prompt 用文字列を組み立てる（WhisperX に固有名詞を予告）
- B. wrong_variants → canonical の置換を segments に適用する
を提供する。辞書のデータはキットではなく各プロジェクト側に置く。

JSON の書式:
{
  "title": "<カテゴリ名>",
  "entries": [
    {"canonical": "<正式表記>", "wrong_variants": ["<誤認識パターン>", ...], "notes": "...",
     "priority": true}
  ]
}
priority（省略可）: true の語は、initial_prompt が Whisper の上限に収まらないとき優先して届ける（#1。plan_initial_prompt）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, NamedTuple

# initial_prompt の形（build_initial_prompt の戻り値は 0.3.x から変えない）
PROMPT_PREFIX = "以下は本音声で言及される固有名詞の一覧です: "
PROMPT_SEPARATOR = "、"
PROMPT_END = "。"

# faster-whisper 1.2.1（whisperx 3.8.5 が使う）の WhisperModel.max_length。
# initial_prompt は " " + prompt.strip() でトークン化され（whisperx/asr.py）、get_prompt が後ろの
# max_length // 2 - 1 = 223 トークンだけを使う（faster_whisper/transcribe.py。#3 §2 の行 1）。
WHISPER_MAX_LENGTH = 448
PROMPT_MAX_TOKENS = WHISPER_MAX_LENGTH // 2 - 1

# encode(text) -> (トークンの id の list, 各トークンの文字の範囲 (start, end) の list)。特殊トークンは足さない。
Encoder = Callable[[str], "tuple[list[int], list[tuple[int, int]]]"]


def load_glossary(glossary_dir: Path | None) -> list[dict]:
    """ディレクトリ内の全 JSON を読み込み、フラットな entry リストを返す。"""
    entries: list[dict] = []
    if glossary_dir is None or not glossary_dir.exists():
        return entries
    for json_file in sorted(glossary_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8-sig"))  # BOM 許容
        except json.JSONDecodeError as e:
            print(f"[glossary] WARN: {json_file.name} のパース失敗: {e}")
            continue
        for entry in data.get("entries", []):
            entry["source"] = json_file.stem
            entries.append(entry)
    return entries


def merge_glossaries(*layers: list[dict]) -> list[dict]:
    """複数層の辞書をマージする。後の層ほど優先（同じ canonical は後勝ち、wrong_variants は和集合）。

    典型: merge_glossaries(global_entries, project_entries)
    """
    merged: dict[str, dict] = {}
    order: list[str] = []
    for layer in layers:
        for entry in layer:
            canonical = entry.get("canonical")
            if not canonical:
                continue
            if canonical in merged:
                prev = merged[canonical]
                variants = list(prev.get("wrong_variants", []) or [])
                for v in entry.get("wrong_variants", []) or []:
                    if v and v not in variants:
                        variants.append(v)
                new = dict(entry)
                new["wrong_variants"] = variants
                merged[canonical] = new
            else:
                merged[canonical] = dict(entry)
                order.append(canonical)
    return [merged[c] for c in order]


def build_initial_prompt(entries: list[dict]) -> str:
    """WhisperX の initial_prompt 用文字列を組み立てる。

    命令文（「正確に書き起こしてください」等）は文字起こし結果に hallucination として
    混入する事故が観測されているため、中立的な体言止めにする。
    """
    canonicals = [e["canonical"] for e in entries if e.get("canonical")]
    if not canonicals:
        return ""
    return PROMPT_PREFIX + PROMPT_SEPARATOR.join(canonicals) + PROMPT_END


# --- initial_prompt の長さと、届ける語の優先 (#1) ------------------------------

def is_priority(entry: dict) -> bool:
    """辞書の項目に、initial_prompt で優先する印（"priority": true）があるか。JSON の true のときだけ。"""
    return entry.get("priority") is True


class PromptPlan(NamedTuple):
    """Whisper に渡す initial_prompt と、何が届くかの数（plan_initial_prompt の戻り値）。"""

    text: str  # Whisper に渡す initial_prompt
    tokens: int | None  # whisperx がトークン化する " " + text.strip() のトークン数（数えられないとき None）
    words: int  # 辞書の語（canonical のある項目）の数
    reached: list[str] | None  # Whisper に届く語（辞書の順）。数えられないとき None
    missed: list[str] | None  # 届かない語（辞書の順）。数えられないとき None
    prefix_reached: bool | None  # 前置きの文が欠けずに届くか
    marked: int  # 印の語の数
    marked_reached: int | None  # 届く印の語の数
    selected: bool  # キットが語を選んだ（印の語があって、全部は収まらないとき）
    missed_marks: list[bool] | None = None  # missed の語ごとに、印の語か（missed と同じ順）


def _whisper_text(text: str) -> str:
    """whisperx が tokenizer.encode に渡す文字列（whisperx/asr.py の " " + initial_prompt.strip()）。"""
    return " " + text.strip() if text else ""


def _assemble(canonicals: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """build_initial_prompt と同じ形の文字列と、_whisper_text の中での各語の文字の範囲 [start, end)。"""
    if not canonicals:
        return "", []
    text = PROMPT_PREFIX + PROMPT_SEPARATOR.join(canonicals) + PROMPT_END
    pos = 1 + len(PROMPT_PREFIX)  # 先頭の " " の分
    spans = []
    for c in canonicals:
        spans.append((pos, pos + len(c)))
        pos += len(c) + len(PROMPT_SEPARATOR)
    return text, spans


def _reach(text: str, spans: list[tuple[int, int]], encode: Encoder, max_tokens: int) -> tuple[int, list[bool], bool]:
    """(トークン数, 語ごとに届くか, 前置きの文が届くか)。

    届く = 後ろ max_tokens トークンに欠けずに入る: 切り捨てた最後のトークンの終わりより後ろから始まる語
    （1 字が 2 トークンに割れて境目がその字にかかる語は届かない側。tools/measure/prompt_tokens.py の analyze と同じ）。
    """
    if not text:
        return 0, [], False
    ids, offsets = encode(_whisper_text(text))
    n = len(ids)
    dropped = max(0, n - max_tokens)
    if not dropped:
        return n, [True] * len(spans), True
    cut = offsets[dropped - 1][1]
    return n, [s >= cut for s, _ in spans], False


def count_prompt_tokens(text: str, encode: Encoder) -> int:
    """whisperx が initial_prompt をトークン化したときの数（" " + text.strip()。空なら 0）。"""
    return len(encode(_whisper_text(text))[0]) if text else 0


def plan_initial_prompt(
    entries: list[dict], encode: Encoder | None = None, *, max_tokens: int = PROMPT_MAX_TOKENS
) -> PromptPlan:
    """Whisper に渡す initial_prompt を決め、どの語が届くかを数える（#1）。

    規則:
      1. 前置きの文と全部の語が max_tokens に収まる → build_initial_prompt と同じ文字列。
      2. 収まらず、印（"priority": true）の語が無い → build_initial_prompt と同じ文字列（0.5.0 と同じ。
         Whisper が後ろ max_tokens だけを使う）。届く語を数えるだけ。
      3. 収まらず、印の語がある → 前置きの文を入れ、優先の高い順に 1 語ずつ、合わせて max_tokens に入るなら入れる
         （入らない語は飛ばして次へ）。優先の順は 印の語 → 印の無い語、同じ印のうちでは辞書の後ろの方が先。
         並べるのは優先の低い順（印の無い語、印の語。それぞれの中は辞書の順）で、優先の高い語が後ろに来る。
         どの語も 1 語で前置きの文と合わせて max_tokens を超えるとき（語 1 つが 200 トークンほどの、実の辞書ではまず無い
         長さ）は、1 語も入らず、文字列は空（Whisper に prompt を渡さない。ログは「届く 0/N 語」）。
    encode が None（トークン数を数えられない）なら、全部の語を優先の低い順に並べる（印が無ければ
    build_initial_prompt と同じ文字列）。届く語は分からない（None）。
    """
    items = [e for e in entries if e.get("canonical")]
    canon = [e["canonical"] for e in items]
    marks = [is_priority(e) for e in items]
    n_marked = sum(marks)
    order = [i for i, m in enumerate(marks) if not m] + [i for i, m in enumerate(marks) if m]

    if encode is None:
        text, _ = _assemble([canon[i] for i in order])
        return PromptPlan(text, None, len(items), None, None, None, n_marked, None, False)

    text, spans = _assemble(canon)  # = build_initial_prompt(entries)
    tokens, flags, prefix_ok = _reach(text, spans, encode, max_tokens)
    selected = False
    if tokens > max_tokens and n_marked:
        chosen: set[int] = set()
        for i in reversed(order):  # 優先の高い順
            trial, _ = _assemble([canon[j] for j in order if j in chosen or j == i])
            if count_prompt_tokens(trial, encode) <= max_tokens:
                chosen.add(i)
        picked = [j for j in order if j in chosen]
        text, sub_spans = _assemble([canon[j] for j in picked])
        tokens, sub_flags, prefix_ok = _reach(text, sub_spans, encode, max_tokens)
        reached_idx = {j for j, ok in zip(picked, sub_flags) if ok}
        flags = [i in reached_idx for i in range(len(items))]
        selected = True
    return PromptPlan(
        text=text,
        tokens=tokens,
        words=len(items),
        reached=[c for c, ok in zip(canon, flags) if ok],
        missed=[c for c, ok in zip(canon, flags) if not ok],
        prefix_reached=prefix_ok,
        marked=n_marked,
        marked_reached=sum(1 for m, ok in zip(marks, flags) if m and ok),
        selected=selected,
        missed_marks=[m for m, ok in zip(marks, flags) if not ok],
    )


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
