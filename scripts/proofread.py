"""Claude Code CLI によるボイスメモ校閲（文脈考慮の誤り修正＋軽い整え）。

content_pipeline/scripts/02b_proofread.py の CLI 呼び出しパターンを流用。
Claude Max サブスクリプションの容量内で動作（ANTHROPIC_API_KEY 不要・追加課金なし）。

メモは短く単独話者なので、segment 差分マージのような複雑さは不要。
本文テキストを丸ごと渡し、校閲後テキストを受け取るシンプルな方式。

校閲後テキストは、確かめ（check_result: 空・num_turns・長さの比・段落の数）を通ったときだけ返す。
通らなければ ProofreadRejected の子を送出し、呼び出し側は校閲前の本文を使う。
失敗の理由は describe_failure で (理由の語, 1 行の補足) にして md に残す（#8）。
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

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

    Windows の PIPE 詰まり回避のため stdin/stdout を tempfile 経由にする
    （02b_proofread.py と同じ手法）。ANTHROPIC_API_KEY は除去して Max を強制。
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

    try:
        with open(stdin_path, "r", encoding="utf-8") as fin, \
             open(stdout_path, "w", encoding="utf-8") as fout:
            cp = subprocess.run(
                cmd, stdin=fin, stdout=fout, stderr=subprocess.PIPE,
                text=True, encoding="utf-8", env=env, timeout=900,
            )
        if cp.returncode != 0:
            raise ClaudeCLIError(
                f"Claude Code CLI failed (exit {cp.returncode}):\n{cp.stderr or ''}",
                detail=f"exit {cp.returncode}",
            )
        raw = stdout_path.read_text(encoding="utf-8")
    finally:
        for p in (stdin_path, stdout_path):
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
        raise ClaudeCLIError(
            f"Claude error (subtype={data.get('subtype')}, num_turns={data.get('num_turns')}): {head}",
            detail=f"is_error (subtype {data.get('subtype')}, num_turns {data.get('num_turns')})",
        )
    return data


def _call_claude(system_prompt: str, user_message: str, model: str) -> str:
    """Claude Code CLI を呼んで result テキストを返す（呼び方と戻り値は 0.3.x と同じ）。"""
    return _call_claude_json(system_prompt, user_message, model).get("result", "") or ""


_TRAILING_NOTE_RE = re.compile(
    r"\n\s*(?:---+|\*\*\*+)\s*\n\s*(?:修正|校閲|変更|注[:：記]|補足|※)[\s\S]*$"
)


def _strip_wrapping(text: str) -> str:
    """前置き・コードフェンス・末尾の校閲メモを保険で除去する。

    モデルが規律に反して本文の後ろに「---\\n修正3点：…」のような注記を付けることがある
    （2026-08-25 に観測）。区切り線の後に修正/校閲/注記で始まるブロックが続く場合は本文から落とす。
    """
    t = text.strip()
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
    """校閲結果の一部が欠けた疑い（CLI が 1 回の応答で返していない、段落が減った）。

    #4 の U4 で、12,638 字の本文が num_turns 3・段落 72 → 45・比 0.602 で返った（前の約 4 割が無い）。
    """

    reason = "incomplete"


# 校閲後の文字数が入力のこの割合を下回ったら「要約・欠落の疑い」として棄却する。
# レベルB（フィラー除去＋軽い整え）なら 2〜3 割減が上限の目安。
MIN_LENGTH_RATIO = 0.6

_PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t　]*\n")


def count_paragraphs(text: str) -> int:
    """空行で区切った段落の数（transcribe.format_body は段落を空行でつなぐ）。"""
    return len([p for p in _PARAGRAPH_SPLIT_RE.split(text.strip()) if p.strip()])


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
) -> None:
    """校閲結果 cleaned を、入力 body の校閲後として使ってよいかを確かめる。使えなければ ProofreadRejected の子を送出。

    順: 空 → num_turns（1 でない = CLI が続きの turn に入った。result は最後の turn の文だけ）
    → 長さの比 → 段落の数（減ったら欠けの疑い。増えるのは長文を分けただけなので通す）。
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


def proofread_body(
    body: str,
    canonicals: list[str],
    *,
    model: str = "claude-sonnet-4-6",
    min_length_ratio: float = MIN_LENGTH_RATIO,
) -> str:
    """本文を Claude で校閲して校閲後テキストを返す。

    呼び出し側は例外を捕捉し、失敗時は元の body を使う（文字起こしは失わない）。
    返すのは check_result を通った校閲後テキストだけ。空・一部の欠け・短すぎる結果は
    ProofreadRejected の子を送出する（意味を変えない規律の機械的なガード）。理由は describe_failure で読む。
    body が空白だけのときは CLI を呼ばずに body を返す（校閲していないので、呼び出し側は校閲済みとして扱わない）。
    """
    if not body.strip():
        return body
    system_prompt = load_system_prompt()
    glossary_line = "、".join(canonicals) if canonicals else "（指定なし）"
    user_message = (
        "以下はボイスメモの文字起こしです。システムプロンプトの方針（レベルB: 誤り修正＋軽い整え、"
        "意味は変えない）に従って校閲し、**校閲後の本文テキストのみ**を返してください。\n\n"
        f"【固有名詞 canonical 一覧（最優先で守る）】\n{glossary_line}\n\n"
        "【文字起こし本文】\n"
        f"{body}"
    )
    data = _call_claude_json(system_prompt, user_message, model)
    cleaned = _strip_wrapping(str(data.get("result") or ""))
    check_result(body, cleaned, num_turns=data.get("num_turns"), min_length_ratio=min_length_ratio)
    return cleaned


def describe_failure(exc: BaseException) -> tuple[str, str]:
    """校閲の失敗を、md に書く (理由の語, 1 行の補足) にする。

    理由の語: timeout / cli_error / empty / too_short / incomplete / error。
    補足は数と理由の語だけ（本文・辞書の語・CLI の stderr・例外の文は入れない）。
    """
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
