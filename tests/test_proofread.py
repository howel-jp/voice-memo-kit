"""proofread.py の安全ガードのテスト。Claude CLI 呼び出しは差し替えるので外部依存なし。

語と文は合成 (public repo のため、利用者の辞書の語・メモの中身を使わない)。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import proofread as p  # noqa: E402


def _no_process(*a, **k):
    raise AssertionError("テストが本物のプロセスを起動しようとした")


def _paras(n: int, start: int = 0) -> list[str]:
    """同じ長さの合成の段落 (i < 100 なら長さがそろう)。"""
    return [f"合成の段落{i:02d}の文です。今日は試しに話します。" for i in range(start, start + n)]


class _Patched(unittest.TestCase):
    """_call_claude_json を差し替える。subprocess.run は塞ぐ (誤って本物の CLI を呼ばないため)。"""

    def setUp(self):
        for patch in (
            mock.patch.object(p, "load_system_prompt", lambda: "SYSTEM"),
            mock.patch.object(p.subprocess, "run", _no_process),
            mock.patch.object(p, "find_claude_bin", lambda: Path("claude-stub.exe")),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _stub(self, reply: str, num_turns=1):
        captured = {}

        def fake(system_prompt, user_message, model):
            captured.update(system=system_prompt, user=user_message, model=model)
            data = {"type": "result", "is_error": False, "result": reply}
            if num_turns is not None:
                data["num_turns"] = num_turns
            return data

        patch = mock.patch.object(p, "_call_claude_json", fake)
        patch.start()
        self.addCleanup(patch.stop)
        return captured


class ProofreadBodyTest(_Patched):
    BODY = "えーと、今日はですね、合成の話をしたいと思います。" * 5

    def test_returns_cleaned_text_and_passes_glossary(self):
        cap = self._stub("今日は合成の話をしたいと思います。" * 5)
        out = p.proofread_body(self.BODY, ["ゾルマ", "Qantrel"], model="m")
        self.assertTrue(out.startswith("今日は"))
        self.assertIn("ゾルマ、Qantrel", cap["user"])
        self.assertIn(self.BODY, cap["user"])
        self.assertEqual(cap["model"], "m")

    def test_strips_code_fence(self):
        self._stub("```\n" + self.BODY + "\n```")
        self.assertEqual(p.proofread_body(self.BODY, []), self.BODY)

    def test_strips_trailing_revision_notes(self):
        # 2026-08-25 に観測: 本文の後ろに「---\n修正3点：…」が付いた
        self._stub(self.BODY + "\n\n---\n\n修正3点：\n- 「力化」→「強化」\n- 「撃つ」→「打つ」\n\n「おめん」は原文のまま残しました。")
        self.assertEqual(p.proofread_body(self.BODY, []), self.BODY)

    def test_keeps_body_separator_without_notes(self):
        # 本文内の区切り線（後ろが本文）は落とさない
        body = self.BODY + "\n\n---\n\n" + self.BODY
        self._stub(body)
        self.assertEqual(p.proofread_body(body, []), body)

    def test_empty_body_short_circuits_without_calling(self):
        cap = self._stub("should not be used")
        self.assertEqual(p.proofread_body("   ", []), "   ")
        self.assertEqual(cap, {})

    def test_empty_reply_is_rejected_not_returned_as_body(self):
        # #3 §2 の 6a: 0.3.3 までは元の本文を返し、校閲済みとして扱っていた
        for reply in ("", "   \n", "```\n```"):
            with self.subTest(reply=reply):
                self._stub(reply)
                with self.assertRaises(p.ProofreadEmpty) as cm:
                    p.proofread_body(self.BODY, [])
                self.assertIsInstance(cm.exception, p.ProofreadRejected)
                self.assertEqual(p.describe_failure(cm.exception), ("empty", "空の結果"))

    def test_too_short_reply_is_rejected(self):
        self._stub("要約: 合成の話。")
        with self.assertRaises(p.ProofreadRejected) as cm:
            p.proofread_body(self.BODY, [])
        self.assertIsInstance(cm.exception, p.ProofreadTooShort)
        code, detail = p.describe_failure(cm.exception)
        self.assertEqual(code, "too_short")
        self.assertIn("下限 60%", detail)

    def test_ratio_threshold_is_configurable(self):
        short = self.BODY[: len(self.BODY) // 2]
        self._stub(short)
        with self.assertRaises(p.ProofreadRejected):
            p.proofread_body(self.BODY, [], min_length_ratio=0.6)
        self.assertEqual(p.proofread_body(self.BODY, [], min_length_ratio=0.4), short)


class IncompleteResultTest(_Patched):
    """一部が欠けた結果 (#4 の U4 run1: 段落 72 → 45、比 0.602、num_turns 3) を校閲済みにしない。"""

    BODY = "\n\n".join(_paras(72))
    FRONT_DROPPED = "\n\n".join(_paras(72)[27:])  # 前の 27 段落 (約 4 割) が無い返り

    def test_u4_shape_passes_the_ratio_guard_alone(self):
        ratio = len(self.FRONT_DROPPED) / len(self.BODY)
        self.assertGreater(ratio, p.MIN_LENGTH_RATIO)  # 今の 60% のガードだけでは通ってしまう形

    def test_multiple_turns_is_incomplete(self):
        self._stub(self.BODY, num_turns=3)  # 段落がそろっていても、続きの turn に入った結果は使わない
        with self.assertRaises(p.ProofreadIncomplete) as cm:
            p.proofread_body(self.BODY, [])
        self.assertEqual(p.describe_failure(cm.exception), ("incomplete", "num_turns 3"))

    def test_dropped_front_paragraphs_is_incomplete_even_with_one_turn(self):
        self._stub(self.FRONT_DROPPED, num_turns=1)
        with self.assertRaises(p.ProofreadIncomplete) as cm:
            p.proofread_body(self.BODY, [])
        self.assertEqual(p.describe_failure(cm.exception), ("incomplete", "段落 72 → 45"))

    def test_u4_run1_shape_reports_num_turns_first(self):
        self._stub(self.FRONT_DROPPED, num_turns=3)
        with self.assertRaises(p.ProofreadIncomplete) as cm:
            p.proofread_body(self.BODY, [])
        self.assertEqual(cm.exception.detail, "num_turns 3")

    def test_more_paragraphs_is_accepted(self):
        # 長文を分けるのは欠けではない (プロンプトが許している)
        body = "\n\n".join(_paras(2))
        reply = "\n\n".join([_paras(1)[0], "合成の段落01の文です。", "今日は試しに話します。"])
        self._stub(reply)
        self.assertEqual(p.proofread_body(body, []), reply)

    def test_missing_num_turns_uses_paragraphs_only(self):
        self._stub(self.BODY, num_turns=None)
        self.assertEqual(p.proofread_body(self.BODY, []), self.BODY)
        self._stub(self.FRONT_DROPPED, num_turns=None)
        with self.assertRaises(p.ProofreadIncomplete):
            p.proofread_body(self.BODY, [])

    def test_unreadable_num_turns_is_incomplete(self):
        self._stub(self.BODY, num_turns="many")
        with self.assertRaises(p.ProofreadIncomplete) as cm:
            p.proofread_body(self.BODY, [])
        self.assertEqual(cm.exception.detail, "num_turns ?")

    def test_count_paragraphs(self):
        self.assertEqual(p.count_paragraphs("a\n\nb\n \nc\n\n\n\nd\n"), 4)
        self.assertEqual(p.count_paragraphs("a\nb"), 1)
        self.assertEqual(p.count_paragraphs("  "), 0)


class DescribeFailureTest(unittest.TestCase):
    def test_reason_words(self):
        cases = [
            (subprocess.TimeoutExpired(["claude"], 900), ("timeout", "900 秒で返らなかった")),
            (p.ClaudeCLIError("合成の失敗の文", detail="exit 1"), ("cli_error", "exit 1")),
            (p.ClaudeCLINotFound("見つかりません"), ("cli_error", "Claude Code CLI が見つからない")),
            (p.ProofreadIncomplete("x", detail="段落 3 → 2"), ("incomplete", "段落 3 → 2")),
            (ValueError("合成の本文を含む例外の文"), ("error", "ValueError")),  # 例外の文は md に入れない
        ]
        for exc, expected in cases:
            with self.subTest(exc=type(exc).__name__):
                self.assertEqual(p.describe_failure(exc), expected)

    def test_not_found_is_still_file_not_found(self):
        self.assertTrue(issubclass(p.ClaudeCLINotFound, FileNotFoundError))
        self.assertTrue(issubclass(p.ClaudeCLIError, RuntimeError))


class CallClaudeJsonTest(unittest.TestCase):
    """_call_claude_json を、subprocess.run のスタブ (CLI の JSON を stdout のファイルに書く) で通す。"""

    def setUp(self):
        for patch in (
            mock.patch.object(p, "find_claude_bin", lambda: Path("claude-stub.exe")),
            mock.patch.object(p, "load_system_prompt", lambda: "SYSTEM"),
        ):
            patch.start()
            self.addCleanup(patch.stop)

    def _run(self, payload: str | None = None, *, returncode: int = 0, stderr: str = "", timeout: bool = False):
        calls = []

        def fake(cmd, *a, **k):
            calls.append({"cmd": cmd, **k, "stdin_text": k["stdin"].read(), "stdout_name": k["stdout"].name})
            if timeout:
                raise subprocess.TimeoutExpired(cmd, k["timeout"])
            if payload is not None:
                k["stdout"].write(payload)
            return subprocess.CompletedProcess(cmd, returncode, None, stderr)

        patch = mock.patch.object(p.subprocess, "run", fake)
        patch.start()
        self.addCleanup(patch.stop)
        return calls

    def test_returns_cli_json_and_call_claude_keeps_str(self):
        calls = self._run(json.dumps({"is_error": False, "num_turns": 1, "result": "合成の校閲後"}, ensure_ascii=False))
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "dummy"}):
            data = p._call_claude_json("S", "合成の依頼", "m")
            self.assertEqual(p._call_claude("S", "合成の依頼", "m"), "合成の校閲後")
        self.assertEqual((data["num_turns"], data["result"]), (1, "合成の校閲後"))
        self.assertEqual(calls[0]["timeout"], 900)
        self.assertEqual(calls[0]["stdin_text"], "合成の依頼")
        self.assertNotIn("ANTHROPIC_API_KEY", calls[0]["env"])
        self.assertIn("--print", calls[0]["cmd"])
        self.assertFalse(Path(calls[0]["stdout_name"]).exists())  # 一時ファイルは消す

    def test_is_error_detail_has_no_result_text(self):
        self._run(json.dumps({"is_error": True, "subtype": "error_during_execution", "num_turns": 4,
                              "result": "合成の途中の本文"}, ensure_ascii=False))
        with self.assertRaises(p.ClaudeCLIError) as cm:
            p._call_claude_json("S", "U", "m")
        self.assertEqual(cm.exception.detail, "is_error (subtype error_during_execution, num_turns 4)")
        self.assertEqual(p.describe_failure(cm.exception)[0], "cli_error")

    def test_nonzero_exit_detail_has_no_stderr(self):
        self._run("", returncode=1, stderr="合成の stderr の文")
        with self.assertRaises(p.ClaudeCLIError) as cm:
            p._call_claude_json("S", "U", "m")
        self.assertEqual(cm.exception.detail, "exit 1")
        self.assertIn("合成の stderr の文", str(cm.exception))  # コンソールには出す

    def test_unreadable_output(self):
        for payload, detail in (("not json", "出力が JSON として読めない"), ("[1]", "出力が JSON のオブジェクトでない")):
            with self.subTest(payload=payload):
                self._run(payload)
                with self.assertRaises(p.ClaudeCLIError) as cm:
                    p._call_claude_json("S", "U", "m")
                self.assertEqual(cm.exception.detail, detail)

    def test_timeout_propagates(self):
        self._run(timeout=True)
        with self.assertRaises(subprocess.TimeoutExpired) as cm:
            p.proofread_body("合成の本文です。", [])
        self.assertEqual(p.describe_failure(cm.exception), ("timeout", "900 秒で返らなかった"))

    def test_proofread_body_reads_num_turns_from_cli_json(self):
        body = "\n\n".join(_paras(3))
        self._run(json.dumps({"is_error": False, "num_turns": 2, "result": body}, ensure_ascii=False))
        with self.assertRaises(p.ProofreadIncomplete):
            p.proofread_body(body, [])


class FindClaudeBinTest(unittest.TestCase):
    def test_not_found_raises_file_not_found_subclass(self):
        with tempfile.TemporaryDirectory() as d:
            env = {k: v for k, v in os.environ.items() if k != "CLAUDE_CODE_BIN"}
            env["USERPROFILE"] = d
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(p.shutil, "which", lambda name: None):
                with self.assertRaises(FileNotFoundError) as cm:
                    p.find_claude_bin()
        self.assertIsInstance(cm.exception, p.ClaudeCLINotFound)


if __name__ == "__main__":
    unittest.main()
