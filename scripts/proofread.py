"""Claude Code CLI によるボイスメモ校閲（文脈考慮の誤り修正＋軽い整え）。

content_pipeline/scripts/02b_proofread.py の CLI 呼び出しパターンを流用。
Claude Max サブスクリプションの容量内で動作（ANTHROPIC_API_KEY 不要・追加課金なし）。

メモは短く単独話者なので、segment 差分マージのような複雑さは不要。
本文テキストを丸ごと渡し、校閲後テキストを受け取るシンプルな方式。
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from glob import glob
from pathlib import Path

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


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
    raise FileNotFoundError(
        "Claude Code CLI バイナリが見つかりません。\n"
        "  対処1: VS Code 拡張 'Anthropic.claude-code' をインストール\n"
        "  対処2: 環境変数 CLAUDE_CODE_BIN にバイナリパスを指定"
    )


def load_system_prompt() -> str:
    path = PROMPTS_DIR / "memo_proofread.md"
    if not path.exists():
        raise FileNotFoundError(f"校閲プロンプトが見つかりません: {path}")
    return path.read_text(encoding="utf-8")


def _call_claude(system_prompt: str, user_message: str, model: str) -> str:
    """Claude Code CLI を呼んで result テキストを返す。

    Windows の PIPE 詰まり回避のため stdin/stdout を tempfile 経由にする
    （02b_proofread.py と同じ手法）。ANTHROPIC_API_KEY は除去して Max を強制。
    """
    import json

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
            raise RuntimeError(
                f"Claude Code CLI failed (exit {cp.returncode}):\n{cp.stderr or ''}"
            )
        raw = stdout_path.read_text(encoding="utf-8")
    finally:
        for p in (stdin_path, stdout_path):
            try:
                p.unlink()
            except OSError:
                pass

    data = json.loads(raw)
    if data.get("is_error"):
        raise RuntimeError(f"Claude error: {data}")
    return data.get("result", "") or ""


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
    """校閲結果が安全基準を満たさない（要約・大幅な欠落の疑い）ときに送出。呼び出し側は生本文にフォールバックする。"""


# 校閲後の文字数が入力のこの割合を下回ったら「要約・欠落の疑い」として棄却する。
# レベルB（フィラー除去＋軽い整え）なら 2〜3 割減が上限の目安。
MIN_LENGTH_RATIO = 0.6


def proofread_body(
    body: str,
    canonicals: list[str],
    *,
    model: str = "claude-sonnet-4-6",
    min_length_ratio: float = MIN_LENGTH_RATIO,
) -> str:
    """本文を Claude で校閲して校閲後テキストを返す。

    呼び出し側は例外を捕捉し、失敗時は元の body を使う（文字起こしは失わない）。
    校閲後が短すぎる場合は ProofreadRejected を送出する（意味を変えない規律の機械的なガード）。
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
    result = _call_claude(system_prompt, user_message, model)
    cleaned = _strip_wrapping(result)
    if not cleaned:
        return body
    ratio = len(cleaned) / max(1, len(body))
    if ratio < min_length_ratio:
        raise ProofreadRejected(
            f"校閲後が短すぎます（{len(cleaned)}/{len(body)} 文字 = {ratio:.0%}、下限 {min_length_ratio:.0%}）。"
            "要約・欠落の疑いがあるため生の文字起こしを使います"
        )
    return cleaned
