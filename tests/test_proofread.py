"""proofread.py の安全ガードのテスト。Claude CLI 呼び出しは差し替えるので外部依存なし。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import proofread as p  # noqa: E402


class _Patched(unittest.TestCase):
    def setUp(self):
        self._orig_call = p._call_claude
        self._orig_prompt = p.load_system_prompt
        p.load_system_prompt = lambda: "SYSTEM"

    def tearDown(self):
        p._call_claude = self._orig_call
        p.load_system_prompt = self._orig_prompt

    def _stub(self, reply: str):
        captured = {}

        def fake(system_prompt, user_message, model):
            captured["system"] = system_prompt
            captured["user"] = user_message
            captured["model"] = model
            return reply

        p._call_claude = fake
        return captured


class ProofreadBodyTest(_Patched):
    BODY = "えーと、今日はですね、プロジェクトの話をしたいと思います。" * 5

    def test_returns_cleaned_text_and_passes_glossary(self):
        cap = self._stub("今日はプロジェクトの話をしたいと思います。" * 5)
        out = p.proofread_body(self.BODY, ["サンイチ", "Plurality"], model="m")
        self.assertTrue(out.startswith("今日は"))
        self.assertIn("サンイチ、Plurality", cap["user"])
        self.assertIn(self.BODY, cap["user"])
        self.assertEqual(cap["model"], "m")

    def test_strips_code_fence(self):
        self._stub("```\n" + self.BODY + "\n```")
        self.assertEqual(p.proofread_body(self.BODY, []), self.BODY)

    def test_empty_body_short_circuits_without_calling(self):
        cap = self._stub("should not be used")
        self.assertEqual(p.proofread_body("   ", []), "   ")
        self.assertEqual(cap, {})

    def test_empty_reply_falls_back_to_body(self):
        self._stub("")
        self.assertEqual(p.proofread_body(self.BODY, []), self.BODY)

    def test_too_short_reply_is_rejected(self):
        self._stub("要約: プロジェクトの話。")
        with self.assertRaises(p.ProofreadRejected):
            p.proofread_body(self.BODY, [])

    def test_ratio_threshold_is_configurable(self):
        short = self.BODY[: len(self.BODY) // 2]
        self._stub(short)
        with self.assertRaises(p.ProofreadRejected):
            p.proofread_body(self.BODY, [], min_length_ratio=0.6)
        self.assertEqual(p.proofread_body(self.BODY, [], min_length_ratio=0.4), short)


if __name__ == "__main__":
    unittest.main()
