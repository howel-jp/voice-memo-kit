"""Claude Code CLI によるボイスメモ校閲（文脈考慮の誤り修正＋軽い整え）。

content_pipeline/scripts/02b_proofread.py の CLI 呼び出しパターンを流用。
Claude Max サブスクリプションの容量内で動作（ANTHROPIC_API_KEY 不要・追加課金なし）。

メモは短く単独話者なので、segment 差分マージのような複雑さは不要。
本文テキストを渡し、校閲後テキストを受け取るシンプルな方式。
CHUNK_CHARS（3,000 字）以下の本文は丸ごと 1 回で渡す（0.4.0 と同じ）。それより長い本文は、
段落と文の境目で大きさのそろったまとまりに分け、順に校閲してつなぐ（#2。#4 の U4 で、45 分相当を
1 回で校閲すると、前の約 4 割が欠けるか is_error で終わった）。

校閲後テキストは、確かめ（check_result: 空・num_turns・長さの比・段落の数。分けたときは末尾も）を
まとまりごとに通ったときだけ返す。通らなければ ProofreadRejected の子などを送出し、呼び出し側は校閲前の本文を使う。
失敗の理由は describe_failure で (理由の語, 1 行の補足) にして md に残す（#8）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from glob import glob
from pathlib import Path
from typing import NamedTuple

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class ClaudeCLINotFound(FileNotFoundError):
    """Claude Code CLI のバイナリが見つからない（FileNotFoundError の子なので、今の except はそのまま効く）。"""


class ClaudeCLIError(RuntimeError):
    """Claude Code CLI が 0 以外で終わった・is_error を返した・出力が JSON として読めない。

    detail は md に書く 1 行（数と CLI の語だけ。本文・stderr は入れない）。
    """

    def __init__(self, message: str, detail: str):
        super().__init__(message)
        self.detail = detail


class ClaudeCLIResultError(ClaudeCLIError):
    """CLI は終わったが、JSON の is_error が真だった（#4 の U4 run2 の形）。

    分けて校閲するときは、やり直しの対象にする（終了コードのエラーはやり直さない）。
    """


def find_claude_bin() -> Path:
    """Claude Code CLI バイナリを探索（env override → VS Code 系拡張 → PATH）。"""
    env_path = os.environ.get("CLAUDE_CODE_BIN")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p
    user_home = Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
    patterns = [
        user_home / ".vscode" / "extensions"
            / "anthropic.claude-code-*-win32-x64"
            / "resources" / "native-binary" / "claude.exe",
        user_home / ".vscode-insiders" / "extensions"
            / "anthropic.claude-code-*-win32-x64"
            / "resources" / "native-binary" / "claude.exe",
        user_home / ".cursor" / "extensions"
            / "anthropic.claude-code-*-win32-x64"
            / "resources" / "native-binary" / "claude.exe",
    ]
    for pat in patterns:
        matches = sorted(glob(str(pat)))
        if matches:
            return Path(matches[-1])
    for name in ("claude", "claude.exe", "claude.cmd"):
        found = shutil.which(name)
        if found:
            return Path(found)
    raise ClaudeCLINotFound(
        "Claude Code CLI バイナリが見つかりません。\n"
        "  対処1: VS Code 拡張 'Anthropic.claude-code' をインストール\n"
        "  対処2: 環境変数 CLAUDE_CODE_BIN にバイナリパスを指定"
    )


def load_system_prompt() -> str:
    path = PROMPTS_DIR / "memo_proofread.md"
    if not path.exists():
        raise FileNotFoundError(f"校閲プロンプトが見つかりません: {path}")
    return path.read_text(encoding="utf-8")


def _call_claude_json(system_prompt: str, user_message: str, model: str) -> dict:
    """Claude Code CLI を呼んで、CLI の JSON（result・num_turns など）を返す。

    Windows の PIPE 詰まり回避のため stdin/stdout/stderr を tempfile 経由にする
    （02b_proofread.py と同じ手法）。ANTHROPIC_API_KEY は除去して Max を強制。
    stderr も pipe にしない: Python 3.10 の Windows の subprocess.run は、時間切れで子を止めた後、
    pipe が閉じるまで timeout なしで待つので、孫が pipe を持っていると返らない（#4 の U5）。
    """
    bin_path = find_claude_bin()
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)
    env.pop("ANTHROPIC_AUTH_TOKEN", None)

    cmd = [
        str(bin_path),
        "--print",
        "--model", model,
        "--output-format", "json",
        "--append-system-prompt", system_prompt,
        "--tools", "",
    ]

    stdin_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".txt", delete=False
    )
    stdin_tmp.write(user_message)
    stdin_path = Path(stdin_tmp.name)
    stdin_tmp.close()

    stdout_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    )
    stdout_path = Path(stdout_tmp.name)
    stdout_tmp.close()

    stderr_tmp = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".err", delete=False
    )
    stderr_path = Path(stderr_tmp.name)
    stderr_tmp.close()

    try:
        with open(stdin_path, "r", encoding="utf-8") as fin, \
             open(stdout_path, "w", encoding="utf-8") as fout, \
             open(stderr_path, "w", encoding="utf-8") as ferr:
            cp = subprocess.run(
                cmd, stdin=fin, stdout=fout, stderr=ferr,
                text=True, encoding="utf-8", env=env, timeout=900,
            )
        if cp.returncode != 0:
            err = stderr_path.read_text(encoding="utf-8", errors="replace")
            raise ClaudeCLIError(
                f"Claude Code CLI failed (exit {cp.returncode}):\n{err}",
                detail=f"exit {cp.returncode}",
            )
        raw = stdout_path.read_text(encoding="utf-8")
    finally:
        # 孫が開いたままのファイルは消せないことがある（そのときは残る。#4 の U5）
        for p in (stdin_path, stdout_path, stderr_path):
            try:
                p.unlink()
            except OSError:
                pass

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ClaudeCLIError(
            f"Claude Code CLI の出力が JSON として読めません: {e}", detail="出力が JSON として読めない"
        ) from e
    if not isinstance(data, dict):
        raise ClaudeCLIError(
            "Claude Code CLI の出力が JSON のオブジェクトではありません", detail="出力が JSON のオブジェクトでない"
        )
    if data.get("is_error"):
        head = str(data.get("result") or "")[:300]
        parts = [f"{k} {data[k]}" for k in ("subtype", "num_turns") if data.get(k) is not None]
        raise ClaudeCLIResultError(
            f"Claude error (subtype={data.get('subtype')}, num_turns={data.get('num_turns')}): {head}",
            detail="is_error" + (f" ({', '.join(parts)})" if parts else ""),
        )
    return data


def _call_claude(system_prompt: str, user_message: str, model: str) -> str:
    """Claude Code CLI を呼んで result テキストを返す（呼び方と戻り値は 0.3.x と同じ）。"""
    return _call_claude_json(system_prompt, user_message, model).get("result", "") or ""


_TRAILING_NOTE_RE = re.compile(
    r"\n\s*(?:---+|\*\*\*+)\s*\n\s*(?:修正|校閲|変更|注[:：記]|補足|※)[\s\S]*$"
)


def _normalize_newlines(text: str) -> str:
    """CRLF と CR を LF にそろえる。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _strip_wrapping(text: str) -> str:
    """前置き・コードフェンス・末尾の校閲メモを保険で除去する。

    モデルが規律に反して本文の後ろに「---\\n修正3点：…」のような注記を付けることがある
    （2026-08-25 に観測）。区切り線の後に修正/校閲/注記で始まるブロックが続く場合は本文から落とす。
    改行は LF にそろえる（CRLF の返りで段落の数を数え違えず、md に \\r を残さないため）。
    """
    t = _normalize_newlines(text).strip()
    if t.startswith("```"):
        lines = [ln for ln in t.split("\n") if not ln.strip().startswith("```")]
        t = "\n".join(lines).strip()
    t = _TRAILING_NOTE_RE.sub("", t).strip()
    return t


class ProofreadRejected(RuntimeError):
    """校閲結果が安全基準を満たさないときに送出。呼び出し側は校閲前の本文にフォールバックする。

    キットが送出するのは下の子（reason が md の proofread_error の語）。detail は md に書く 1 行（数だけ）。
    """

    reason = "error"

    def __init__(self, message: str, detail: str = ""):
        super().__init__(message)
        self.detail = detail or self.reason


class ProofreadEmpty(ProofreadRejected):
    """前置き・コードフェンス・末尾の注記を除いた校閲結果が空（#3 §2 の 6a）。"""

    reason = "empty"


class ProofreadTooShort(ProofreadRejected):
    """校閲後が入力の min_length_ratio を下回る（要約・大幅な欠落の疑い）。"""

    reason = "too_short"


class ProofreadIncomplete(ProofreadRejected):
    """校閲結果の一部が欠けた疑い（CLI が 1 回の応答で返していない、段落が減った、末尾が文の終わりでない）。

    #4 の U4 で、12,638 字の本文が num_turns 3・段落 72 → 45・比 0.602 で返った（前の約 4 割が無い）。
    """

    reason = "incomplete"


# 校閲後の文字数が入力のこの割合を下回ったら「要約・欠落の疑い」として棄却する。
# レベルB（フィラー除去＋軽い整え）なら 2〜3 割減が上限の目安。
MIN_LENGTH_RATIO = 0.6

_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

# 文の終わり: 。！？!? （後ろに閉じ括弧が続いてもよい）
SENTENCE_END_CHARS = "。！？!?"
CLOSING_CHARS = "」』）)】"


def count_paragraphs(text: str) -> int:
    """空行（空白だけの行を含む）で区切った段落の数（transcribe.format_body は段落を空行でつなぐ）。

    CRLF・CR は LF にそろえてから数える。
    """
    return len([p for p in _PARAGRAPH_SPLIT_RE.split(_normalize_newlines(text).strip()) if p.strip()])


def ends_with_sentence_end(text: str) -> bool:
    """末尾 (空白を除く) が文の終わり (。！？!? の後に閉じ括弧が続いてもよい) か。"""
    t = text.rstrip()
    while t and t[-1] in CLOSING_CHARS:
        t = t[:-1]
    return bool(t) and t[-1] in SENTENCE_END_CHARS


def _as_int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def check_result(
    body: str,
    cleaned: str,
    *,
    num_turns=None,
    min_length_ratio: float = MIN_LENGTH_RATIO,
    check_end: bool = False,
) -> None:
    """校閲結果 cleaned を、入力 body の校閲後として使ってよいかを確かめる。使えなければ ProofreadRejected の子を送出。

    順: 空 → num_turns（1 でない = CLI が続きの turn に入った。result は最後の turn の文だけ）
    → 長さの比 → 段落の数（減ったら欠けの疑い。増えるのは長文を分けただけなので通す）
    → 末尾（check_end のときだけ。入力が文の終わりで終わるのに、校閲後がそうでない = 切れたか、注記が足された）。
    num_turns が JSON に無い（None）ときは、num_turns では判定しない。
    """
    if not cleaned:
        raise ProofreadEmpty("校閲の結果が空です。校閲前の文字起こしを使います", detail="空の結果")
    if num_turns is not None:
        turns = _as_int(num_turns)
        if turns != 1:
            raise ProofreadIncomplete(
                f"CLI が 1 回の応答で返していません（num_turns {num_turns}）。"
                "一部が欠けた疑いがあるため校閲前の文字起こしを使います",
                detail=f"num_turns {turns if turns is not None else '?'}",
            )
    ratio = len(cleaned) / max(1, len(body))
    if ratio < min_length_ratio:
        raise ProofreadTooShort(
            f"校閲後が短すぎます（{len(cleaned)}/{len(body)} 文字 = {ratio:.0%}、下限 {min_length_ratio:.0%}）。"
            "要約・欠落の疑いがあるため校閲前の文字起こしを使います",
            detail=f"{len(cleaned)}/{len(body)} 文字 = {ratio:.0%}、下限 {min_length_ratio:.0%}",
        )
    n_in, n_out = count_paragraphs(body), count_paragraphs(cleaned)
    if n_out < n_in:
        raise ProofreadIncomplete(
            f"校閲後の段落が減っています（{n_in} → {n_out}）。一部が欠けた疑いがあるため校閲前の文字起こしを使います",
            detail=f"段落 {n_in} → {n_out}",
        )
    if check_end and ends_with_sentence_end(body) and not ends_with_sentence_end(cleaned):
        raise ProofreadIncomplete(
            "校閲後の末尾が文の終わりではありません（入力は文の終わりで終わる）。"
            "切れたか注記が足された疑いがあるため校閲前の文字起こしを使います",
            detail="末尾が文の終わりでない",
        )


# --- 長い本文を分ける (#2) ---------------------------------------------------

# この字数以下の本文は分けない（1 回で校閲する。0.4.0 と同じ）。
# 根拠: #4 の U4。1 回で返った 3 点の出力トークンは字数の 3.5〜4.2 倍で、3,000 字なら読みの上限 32,000 の 3 分の 1 程度。
CHUNK_CHARS = 3000

# 境目の種類（小さいほど良い）。Chunk.kind は、そのまとまりの前の境目の種類（最初のまとまりは 0）。
KIND_PARAGRAPH_AFTER_SENTENCE = 1  # 段落の境目で、前の段落が文の終わりで終わる
KIND_SENTENCE = 2  # 段落の中の文の終わり
KIND_PARAGRAPH = 3  # 段落の境目（前が文の終わりでない。文の途中の無音のことがある）
KIND_SOFT = 4  # 「、」・空白
KIND_HARD = 5  # どれも無い（目安の位置で切る）

_PARAGRAPH_SEP_RE = re.compile(r"\n\s*\n")
_SENTENCE_END_RE = re.compile(r"[。！？!?]+[」』）)】]*")
_SOFT_BREAK_RE = re.compile(r"[、，,]|\s")


class Chunk(NamedTuple):
    """分けた本文の 1 まとまり。sep + text を順につなぐと、元の本文と 1 字も違わない。"""

    text: str
    sep: str  # 前のまとまりとの間にあった文字列（段落の境目なら空行。段落の中なら ""）
    kind: int  # 前の境目の種類（KIND_*。最初のまとまりは 0）


def _cut_points(text: str) -> list[tuple[int, int, int]]:
    """切ってよい位置の一覧 (位置, 種類, 区切りの長さ)。まとまりは text[..位置]、次は 位置 + 区切りの長さ から。"""
    points: list[tuple[int, int, int]] = []
    para_start = 0
    seps = list(_PARAGRAPH_SEP_RE.finditer(text))
    spans = [(m.start(), m.end()) for m in seps] + [(len(text), len(text))]
    for sep_start, sep_end in spans:
        para = text[para_start:sep_start]
        last = len(para.rstrip())
        for regex, kind in ((_SENTENCE_END_RE, KIND_SENTENCE), (_SOFT_BREAK_RE, KIND_SOFT)):
            for m in regex.finditer(para):
                if m.end() < last:  # 段落の終わり (の空白) では切らない。そこは段落の境目
                    points.append((para_start + m.end(), kind, 0))
        if sep_start < len(text):
            kind = KIND_PARAGRAPH_AFTER_SENTENCE if ends_with_sentence_end(para) else KIND_PARAGRAPH
            points.append((sep_start, kind, sep_end - sep_start))
        para_start = sep_end
    points.sort()
    return points


def _pick(points: list[tuple[int, int, int]], lo: float, hi: float, ideal: float):
    """lo〜hi の中で、種類のいちばん良い位置 (同じ種類なら ideal に近い位置) を返す。無ければ None。"""
    best = None
    for p in points:
        if p[0] < lo:
            continue
        if p[0] > hi:
            break
        key = (p[1], abs(p[0] - ideal))
        if best is None or key < best[0]:
            best = (key, p)
    return best[1] if best else None


def split_into_chunks(text: str, max_chars: int = CHUNK_CHARS) -> list[Chunk]:
    """本文を max_chars 以下のまとまりに分ける。max_chars 以下なら分けない (1 つ)。

    まとまりの数は ⌈字数 / max_chars⌉ で、大きさをそろえる (目安の位置 = 残りの字数 / 残りのまとまりの数)。
    境目は、目安の ±max_chars/4 (まとまりが max_chars を超えない範囲) の中で、種類の良い順 (KIND_*)、
    同じ種類なら目安に近い所。まとまりの数が増えない位置 (残りが、残りのまとまりの数に収まる位置) を先に探し、
    無ければ数が増えるのを許す。その範囲に無ければ max_chars までの範囲で同じ順。どれも無ければ目安の位置で切る。
    """
    if max_chars < 1:
        raise ValueError("max_chars は 1 以上")
    if len(text) <= max_chars:
        return [Chunk(text, "", 0)]
    points = _cut_points(text)
    chunks: list[Chunk] = []
    start, sep, kind_before = 0, "", 0
    while len(text) - start > max_chars:
        n_rest = -(-(len(text) - start) // max_chars)
        ideal = start + (len(text) - start) / n_rest
        limit = start + max_chars
        need = len(text) - (n_rest - 1) * max_chars  # これより前で切ると、まとまりが 1 つ増える
        lo, hi = max(start + 1, ideal - max_chars / 4), min(ideal + max_chars / 4, limit)
        best = (_pick(points, max(lo, need), hi, ideal)
                or _pick(points, max(start + 1, need), limit, ideal)
                or _pick(points, lo, hi, ideal)
                or _pick(points, start + 1, limit, ideal))
        pos, kind, sep_len = best if best else (min(limit, max(start + 1, need, int(ideal))), KIND_HARD, 0)
        chunks.append(Chunk(text[start:pos], sep, kind_before))
        sep, kind_before, start = text[pos:pos + sep_len], kind, pos + sep_len
    chunks.append(Chunk(text[start:], sep, kind_before))
    return chunks


def _join(chunks: list[Chunk], outputs: list[str]) -> str:
    """校閲後のまとまりを、元の区切りでつなぐ (段落の境目は空行、段落の中は区切りなし)。"""
    out = outputs[0]
    for c, o in zip(chunks[1:], outputs[1:]):
        out += ("\n\n" if c.kind in (KIND_PARAGRAPH_AFTER_SENTENCE, KIND_PARAGRAPH) else "") + o
    return out


# --- 校閲 -------------------------------------------------------------------

def _user_message(body: str, canonicals: list[str]) -> str:
    glossary_line = "、".join(canonicals) if canonicals else "（指定なし）"
    return (
        "以下はボイスメモの文字起こしです。システムプロンプトの方針（レベルB: 誤り修正＋軽い整え、"
        "意味は変えない）に従って校閲し、**校閲後の本文テキストのみ**を返してください。\n\n"
        f"【固有名詞 canonical 一覧（最優先で守る）】\n{glossary_line}\n\n"
        "【文字起こし本文】\n"
        f"{body}"
    )


def _proofread_once(
    body: str, canonicals: list[str], *, system_prompt: str, model: str, min_length_ratio: float,
    check_end: bool = False,
) -> str:
    """1 回の呼び出しで校閲し、check_result を通った校閲後テキストを返す。"""
    data = _call_claude_json(system_prompt, _user_message(body, canonicals), model)
    cleaned = _strip_wrapping(str(data.get("result") or ""))
    check_result(body, cleaned, num_turns=data.get("num_turns"), min_length_ratio=min_length_ratio,
                 check_end=check_end)
    return cleaned


# やり直す失敗 (分けたときだけ)。時間切れ・終了コードのエラー・CLI が無い・JSON が読めないものはやり直さない。
_RETRYABLE = (ProofreadRejected, ClaudeCLIResultError)


def split_in_two(text: str) -> list[Chunk]:
    """やり直しのために、まとまりを半分の近くで 2 つ (以下) に分ける (境目の規則は split_into_chunks と同じ)。

    ふつうはどちらも 6 割以下。境目の少ない短い文で 3 つ以上になったときは、最初の境目だけを使う
    (やり直しの呼び出しを 2 回までにするため)。
    """
    parts = split_into_chunks(text, max(1, -(-len(text) * 3 // 5)))
    if len(parts) <= 2:
        return parts
    first, second = parts[0], parts[1]
    return [first, Chunk(text[len(first.text) + len(second.sep):], second.sep, second.kind)]


def _proofread_chunk(text: str, k: int, n: int, canonicals: list[str], **kw) -> str:
    """まとまり k/n を校閲する。ガードに当たるか is_error なら、2 つに分けて 1 回だけやり直す。"""
    t0 = time.perf_counter()
    print(f"[C] まとまり {k}/{n}（{len(text):,} 字）...", flush=True)
    try:
        out = _proofread_once(text, canonicals, check_end=True, **kw)
    except _RETRYABLE as e:
        code, detail = describe_failure(e)
        halves = split_in_two(text)
        print(f"[C] まとまり {k}/{n} が通らなかった（{code}: {detail}）。"
              f"{len(halves)} つに分けて 1 回だけやり直します", flush=True)
        out = _join(halves, [_proofread_once(h.text, canonicals, check_end=True, **kw) for h in halves])
    print(f"[C] まとまり {k}/{n} 完了（{time.perf_counter() - t0:.0f} 秒）", flush=True)
    return out


def proofread_body(
    body: str,
    canonicals: list[str],
    *,
    model: str = "claude-sonnet-4-6",
    min_length_ratio: float = MIN_LENGTH_RATIO,
    chunk_chars: int = CHUNK_CHARS,
) -> str:
    """本文を Claude で校閲して校閲後テキストを返す。

    呼び出し側は例外を捕捉し、失敗時は元の body を使う（文字起こしは失わない）。
    返すのは check_result を通った校閲後テキストだけ。空・一部の欠け・短すぎる結果は
    ProofreadRejected の子を送出する（意味を変えない規律の機械的なガード）。理由は describe_failure で読む。
    body が空白だけのときは CLI を呼ばずに body を返す（校閲していないので、呼び出し側は校閲済みとして扱わない）。

    chunk_chars を超える本文は split_into_chunks で分け、順に校閲してつなぐ。まとまりごとに check_result
    (末尾の確かめも) を当て、通らなければ 2 つに分けて 1 回だけやり直す。それでも通らないまとまりがあれば、
    残りは呼ばずに例外を送出する（型はそのまま。proofread_chunk = (k, n) を付ける）。
    """
    if not body.strip():
        return body
    system_prompt = load_system_prompt()
    kw = {"system_prompt": system_prompt, "model": model, "min_length_ratio": min_length_ratio}
    if len(body) <= chunk_chars:
        return _proofread_once(body, canonicals, **kw)

    chunks = split_into_chunks(body, chunk_chars)
    n = len(chunks)
    print(f"[C] {len(body):,} 字を {n} まとまりに分けて校閲します（1 まとまり {chunk_chars:,} 字まで）", flush=True)
    if any(c.kind == KIND_HARD for c in chunks):
        print("  [WARN] 文の終わり・段落の境目・読点の無い所で分けたまとまりがあります", flush=True)
    outputs: list[str] = []
    for k, c in enumerate(chunks, 1):
        try:
            outputs.append(_proofread_chunk(c.text, k, n, canonicals, **kw))
        except Exception as e:
            e.proofread_chunk = (k, n)
            if k < n:
                print(f"[C] まとまり {k}/{n} が通らなかったので、残りの {n - k} まとまりは校閲しません", flush=True)
            raise
    return _join(chunks, outputs)


def describe_failure(exc: BaseException) -> tuple[str, str]:
    """校閲の失敗を、md に書く (理由の語, 1 行の補足) にする。

    理由の語: timeout / cli_error / empty / too_short / incomplete / error。
    補足は数と理由の語だけ（本文・辞書の語・CLI の stderr・例外の文は入れない）。
    分けて校閲したまとまりの失敗なら、補足の前に「まとまり k/n: 」を付ける。
    """
    code, detail = _describe(exc)
    chunk = getattr(exc, "proofread_chunk", None)
    if chunk:
        detail = f"まとまり {chunk[0]}/{chunk[1]}: {detail}"
    return code, detail


def _describe(exc: BaseException) -> tuple[str, str]:
    if isinstance(exc, ProofreadRejected):
        return exc.reason, exc.detail
    if isinstance(exc, subprocess.TimeoutExpired):
        sec = exc.timeout
        return "timeout", (f"{sec:g} 秒で返らなかった" if isinstance(sec, (int, float)) else "時間切れ")
    if isinstance(exc, ClaudeCLIError):
        return "cli_error", exc.detail
    if isinstance(exc, ClaudeCLINotFound):
        return "cli_error", "Claude Code CLI が見つからない"
    return "error", type(exc).__name__
