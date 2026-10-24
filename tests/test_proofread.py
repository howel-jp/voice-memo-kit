"""proofread.py の安全ガードのテスト。Claude CLI 呼び出しは差し替えるので外部依存なし。

語と文は合成 (public repo のため、利用者の辞書の語・メモの中身を使わない)。
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
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
        # PR #9 のレビュー 1 回目の任意 1: CRLF・CR と、空白だけの行 (全角空白・nbsp・タブ) も段落の区切り
        self.assertEqual(p.count_paragraphs("a\r\n\r\nb\r\n\r\nc"), 3)
        self.assertEqual(p.count_paragraphs("a\r\rb"), 2)
        self.assertEqual(p.count_paragraphs("a\n\u3000\nb\n\u00a0\nc\n\t\nd"), 4)
        self.assertEqual(p.count_paragraphs("a\r\nb"), 1)  # 1 つの改行は段落の区切りではない

    def test_crlf_reply_is_accepted_and_normalized(self):
        # CRLF の返りを、段落 72 → 1 と数えて誤って捨てない。返す文の改行は LF にそろえる
        self._stub(self.BODY.replace("\n", "\r\n"))
        out = p.proofread_body(self.BODY, [])
        self.assertEqual(out, self.BODY)
        self.assertNotIn("\r", out)

    def test_check_order_empty_then_num_turns_then_ratio_then_paragraphs(self):
        # PR #9 のレビュー 1 回目の任意 3: 判定の順 (空 → num_turns → 比 → 段落) を固定する
        short_and_fewer = "\n\n".join(_paras(10))  # 比 0.14、段落 72 → 10
        cases = [
            ("", 3, p.ProofreadEmpty, "空の結果"),
            (short_and_fewer, 3, p.ProofreadIncomplete, "num_turns 3"),
            (short_and_fewer, 1, p.ProofreadTooShort, None),
            (self.FRONT_DROPPED, 1, p.ProofreadIncomplete, "段落 72 → 45"),
        ]
        for reply, turns, exc_type, detail in cases:
            with self.subTest(exc=exc_type.__name__, turns=turns):
                self._stub(reply, num_turns=turns)
                with self.assertRaises(p.ProofreadRejected) as cm:
                    p.proofread_body(self.BODY, [])
                self.assertIs(type(cm.exception), exc_type)
                if detail is not None:
                    self.assertEqual(cm.exception.detail, detail)


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
            calls.append({"cmd": cmd, **k, "stdin_text": k["stdin"].read(), "stdout_name": k["stdout"].name,
                          "stderr_name": getattr(k["stderr"], "name", None)})
            if timeout:
                raise subprocess.TimeoutExpired(cmd, k["timeout"])
            if payload is not None:
                k["stdout"].write(payload)
            k["stderr"].write(stderr)  # stderr は一時ファイル (#4 の U5)
            return subprocess.CompletedProcess(cmd, returncode, None, None)

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
        # stderr は pipe にしない (#4 の U5)。一時ファイルに書き、終わったら消す
        self.assertIsNot(calls[0]["stderr"], subprocess.PIPE)
        self.assertIsNotNone(calls[0]["stderr_name"])
        self.assertFalse(Path(calls[0]["stderr_name"]).exists())

    def test_is_error_detail_has_no_result_text(self):
        self._run(json.dumps({"is_error": True, "subtype": "error_during_execution", "num_turns": 4,
                              "result": "合成の途中の本文"}, ensure_ascii=False))
        with self.assertRaises(p.ClaudeCLIError) as cm:
            p._call_claude_json("S", "U", "m")
        self.assertEqual(cm.exception.detail, "is_error (subtype error_during_execution, num_turns 4)")
        self.assertEqual(p.describe_failure(cm.exception)[0], "cli_error")

    def test_is_error_detail_omits_missing_fields(self):
        # PR #9 のレビュー 1 回目の任意 4: JSON に無い値を「None」と書かない
        for payload, detail in (({"is_error": True, "num_turns": 4}, "is_error (num_turns 4)"),
                                ({"is_error": True}, "is_error")):
            with self.subTest(detail=detail):
                self._run(json.dumps(payload))
                with self.assertRaises(p.ClaudeCLIError) as cm:
                    p._call_claude_json("S", "U", "m")
                self.assertEqual(cm.exception.detail, detail)

    def test_nonzero_exit_detail_has_no_stderr(self):
        calls = self._run("", returncode=1, stderr="合成の stderr の文")
        with self.assertRaises(p.ClaudeCLIError) as cm:
            p._call_claude_json("S", "U", "m")
        self.assertEqual(cm.exception.detail, "exit 1")
        self.assertNotIsInstance(cm.exception, p.ClaudeCLIResultError)  # 終了コードのエラーはやり直さない側
        self.assertIn("合成の stderr の文", str(cm.exception))  # コンソールには出す (一時ファイルから読む)
        self.assertFalse(Path(calls[0]["stderr_name"]).exists())

    def test_is_error_is_a_result_error(self):
        self._run(json.dumps({"is_error": True, "num_turns": 4}))
        with self.assertRaises(p.ClaudeCLIResultError) as cm:
            p._call_claude_json("S", "U", "m")
        self.assertIsInstance(cm.exception, p.ClaudeCLIError)  # 今の except ClaudeCLIError は今どおり効く
        self.assertEqual(p.describe_failure(cm.exception)[0], "cli_error")

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


_SENTS = [
    "今日は合成の話をします。", "えーと、次の段取りを決めます。", "あの、資料は後で直します。",
    "それで、確認の時間を取りたいです。", "「試しにやってみる」と言っていました。", "うーん、少し迷っています。",
    "「それでいいです。」", "本当に間に合うかな？", "まずは全体の流れを決めたいです！",
]


def _memo(n_paras: int, seed: int = 0, sents: tuple[int, int] = (3, 9)) -> str:
    """合成のメモの本文 (段落は空行区切り。どの段落も文の終わりで終わる)。"""
    rng = random.Random(seed)
    paras = []
    for i in range(n_paras):
        paras.append(f"合成の段落{i:03d}です。" + "".join(rng.choice(_SENTS) for _ in range(rng.randint(*sents))))
    return "\n\n".join(paras)


class EndsWithSentenceEndTest(unittest.TestCase):
    def test_sentence_end(self):
        for text, expected in (("あ。", True), ("あ。」", True), ("あ！）", True), ("あ?", True), ("あ。  \n", True),
                               ("あ」", False), ("あ", False), ("", False), ("あ、", False), ("あ…", False)):
            with self.subTest(text=text):
                self.assertIs(p.ends_with_sentence_end(text), expected)

    def test_check_end_is_off_by_default_and_runs_last(self):
        body = "合成の文です。" * 20
        cut = body[:-1]  # 末尾の「。」が無い (切れた形)
        p.check_result(body, cut)  # 既定は確かめない (3,000 字以下は 0.4.0 と同じ)
        with self.assertRaises(p.ProofreadIncomplete) as cm:
            p.check_result(body, cut, check_end=True)
        self.assertEqual(cm.exception.detail, "末尾が文の終わりでない")
        noted = body + "\n\n（以上、合成の注記を足しました）"  # 区切り線の無い注記 (strip をすり抜ける形)
        with self.assertRaises(p.ProofreadIncomplete):
            p.check_result(body, p._strip_wrapping(noted), check_end=True)
        p.check_result(body + "えーと", cut, check_end=True)  # 入力が文の終わりで終わらないときは確かめない
        with self.assertRaises(p.ProofreadTooShort):  # 順は最後 (短くて末尾も切れた返りは too_short)
            p.check_result(body, body[:30], check_end=True)


class SplitIntoChunksTest(unittest.TestCase):
    MAX = p.CHUNK_CHARS

    def _check_lossless(self, text: str, chunks: list) -> None:
        self.assertEqual("".join(c.sep + c.text for c in chunks), text)
        self.assertTrue(all(0 < len(c.text) <= self.MAX for c in chunks), [len(c.text) for c in chunks])

    def test_short_text_is_not_split(self):
        for text in ("", "合成の文です。", "あ" * self.MAX):
            self.assertEqual(p.split_into_chunks(text), [p.Chunk(text, "", 0)])

    def test_paragraph_boundaries_after_sentence_end_and_even_sizes(self):
        body = _memo(120, seed=1)
        self.assertGreater(len(body), self.MAX * 3)
        chunks = p.split_into_chunks(body)
        self._check_lossless(body, chunks)
        self.assertEqual(len(chunks), math.ceil(len(body) / self.MAX))
        self.assertEqual([c.kind for c in chunks[1:]], [p.KIND_PARAGRAPH_AFTER_SENTENCE] * (len(chunks) - 1))
        self.assertTrue(all(c.sep == "\n\n" for c in chunks[1:]))
        self.assertEqual(sum(p.count_paragraphs(c.text) for c in chunks), 120)
        # 大きさをそろえる: 最後に小さいまとまりが残らない
        self.assertGreaterEqual(min(len(c.text) for c in chunks), self.MAX // 4)

    def test_long_paragraph_is_split_at_sentence_end_not_inside_closing_bracket(self):
        body = _memo(60, seed=2).replace("\n\n", "")  # 1 段落
        chunks = p.split_into_chunks(body)
        self._check_lossless(body, chunks)
        self.assertEqual([c.kind for c in chunks[1:]], [p.KIND_SENTENCE] * (len(chunks) - 1))
        self.assertTrue(all(c.sep == "" for c in chunks[1:]))
        for c in chunks[:-1]:
            self.assertTrue(p.ends_with_sentence_end(c.text), c.text[-5:])  # 分けた境目で文が切れない
        for c in chunks[1:]:
            self.assertNotIn(c.text[0], p.CLOSING_CHARS)  # 「。」と「」」の間で切らない

    def test_cut_after_closing_bracket_when_every_sentence_is_quoted(self):
        body = "「それでいいです。」" * 700  # どの文も「。」」で終わる 1 段落
        chunks = p.split_into_chunks(body)
        self._check_lossless(body, chunks)
        self.assertGreater(len(chunks), 1)
        for c in chunks[1:]:
            self.assertTrue(c.text.startswith("「"), c.text[:3])  # 「。」と「」」の間で切らない
        for c in chunks[:-1]:
            self.assertTrue(c.text.endswith("。」"), c.text[-3:])

    def test_sentence_end_is_preferred_over_paragraph_boundary_without_it(self):
        # 段落の末尾に句点が無い (文の途中の無音で分かれた形) ときは、段落の中の文の終わりで切る
        body = "\n\n".join(pa + "えーと" for pa in _memo(60, seed=3).split("\n\n"))
        chunks = p.split_into_chunks(body)
        self._check_lossless(body, chunks)
        self.assertEqual({c.kind for c in chunks[1:]}, {p.KIND_SENTENCE})

    def test_paragraph_boundary_before_comma_and_hard_cut(self):
        text = _memo(60, seed=4)
        for ch in p.SENTENCE_END_CHARS:
            text = text.replace(ch, "、")
        no_maru = "\n\n".join(pa + "えーと" for pa in text.split("\n\n"))
        chunks = p.split_into_chunks(no_maru)
        self._check_lossless(no_maru, chunks)
        self.assertEqual({c.kind for c in chunks[1:]}, {p.KIND_PARAGRAPH})
        one = no_maru.replace("\n\n", "")
        chunks = p.split_into_chunks(one)
        self._check_lossless(one, chunks)
        self.assertEqual({c.kind for c in chunks[1:]}, {p.KIND_SOFT})
        for c in chunks[:-1]:
            self.assertTrue(c.text.endswith("、"))
        hard = "あ" * (self.MAX * 2 + 10)
        chunks = p.split_into_chunks(hard)
        self._check_lossless(hard, chunks)
        self.assertEqual(([c.kind for c in chunks[1:]], len(chunks)), ([p.KIND_HARD, p.KIND_HARD], 3))

    def test_split_in_two_for_retry(self):
        rng = random.Random(1)
        for trial in range(300):
            text = _memo(rng.randint(1, 30), seed=trial).replace("\n\n", rng.choice(["\n\n", ""]))
            text = text[: rng.randint(2, len(text))]
            with self.subTest(trial=trial):
                parts = p.split_in_two(text)
                self.assertEqual(len(parts), 2 if len(text) > 1 else 1)  # やり直しの呼び出しは 2 回まで
                self.assertEqual("".join(c.sep + c.text for c in parts), text)

    def test_random_bodies_are_lossless_and_bounded(self):
        rng = random.Random(0)
        for trial in range(100):
            with self.subTest(trial=trial):
                paras = []
                target = rng.randint(self.MAX + 1, self.MAX * 7)
                while sum(map(len, paras)) < target:
                    paras.append(_memo(rng.randint(1, 40), seed=rng.randint(0, 999)).replace("\n\n", ""))
                body = rng.choice(["\n\n", "\n \n", "\n\n\n"]).join(paras)
                chunks = p.split_into_chunks(body)
                self._check_lossless(body, chunks)
                # 目安は ⌈字数/3,000⌉。境目の種類を優先するので、合計が 3,000 の倍数に近いと 1 つ増えることがある
                n = math.ceil(len(body) / self.MAX)
                self.assertIn(len(chunks), (n, n + 1))
                self.assertGreaterEqual(min(len(c.text) for c in chunks), self.MAX // 4)
                self.assertTrue({c.kind for c in chunks[1:]} <= {1, 2})  # 文の終わりがあれば、ほかで切らない


class _ChunkStub:
    """_call_claude_json の代わり。reply(body, i) が CLI の JSON を返すか、例外を送出する。"""

    def __init__(self, reply):
        self.reply = reply
        self.bodies: list[str] = []
        self.messages: list[str] = []

    def __call__(self, system_prompt, user_message, model):
        body = user_message.split("【文字起こし本文】\n", 1)[1]
        self.bodies.append(body)
        self.messages.append(user_message)
        return self.reply(body, len(self.bodies) - 1)


def _ok(body: str, turns: int = 1) -> dict:
    return {"type": "result", "is_error": False, "num_turns": turns, "result": body.replace("えーと、", "")}


class ChunkedProofreadTest(_Patched):
    """長い本文を分けて校閲する (#2)。まとまりごとのガード、やり直し 1 回、止め方。"""

    CANON = ["ゾルマ", "Qantrel"]

    def setUp(self):
        super().setUp()
        self.body = _memo(75, seed=5)
        self.chunks = p.split_into_chunks(self.body)
        self.n = len(self.chunks)
        self.assertGreaterEqual(self.n, 3)

    def _run(self, reply):
        stub = _ChunkStub(reply)
        patch = mock.patch.object(p, "_call_claude_json", stub)
        patch.start()
        self.addCleanup(patch.stop)
        out, err = None, None
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                out = p.proofread_body(self.body, self.CANON, model="m")
            except Exception as e:  # noqa: BLE001
                err = e
        return stub, out, err

    def test_split_calls_and_join(self):
        stub, out, err = self._run(lambda b, i: _ok(b))
        self.assertIsNone(err)
        self.assertEqual(stub.bodies, [c.text for c in self.chunks])
        self.assertTrue(all(len(b) <= p.CHUNK_CHARS for b in stub.bodies))
        for c, msg in zip(self.chunks, stub.messages):
            self.assertEqual(msg, p._user_message(c.text, self.CANON))  # 依頼の文は毎回同じ形 (canonical も毎回)
            self.assertIn("ゾルマ、Qantrel", msg)
        self.assertEqual(out, self.body.replace("えーと、", ""))
        self.assertEqual(p.count_paragraphs(out), 75)

    def test_long_single_paragraph_is_joined_without_new_breaks(self):
        self.body = self.body.replace("\n\n", "")
        stub, out, err = self._run(lambda b, i: _ok(b))
        self.assertIsNone(err)
        self.assertGreater(len(stub.bodies), 1)
        self.assertEqual(out, self.body.replace("えーと、", ""))
        self.assertEqual(p.count_paragraphs(out), 1)

    def test_short_body_is_one_call_with_the_040_message_and_no_retry(self):
        self.body = _memo(5, seed=6)
        self.assertLessEqual(len(self.body), p.CHUNK_CHARS)
        stub, out, err = self._run(lambda b, i: _ok(b))
        expected = (
            "以下はボイスメモの文字起こしです。システムプロンプトの方針（レベルB: 誤り修正＋軽い整え、"
            "意味は変えない）に従って校閲し、**校閲後の本文テキストのみ**を返してください。\n\n"
            "【固有名詞 canonical 一覧（最優先で守る）】\nゾルマ、Qantrel\n\n"
            "【文字起こし本文】\n" + self.body
        )
        self.assertEqual(stub.messages, [expected])
        stub, out, err = self._run(lambda b, i: _ok(b, turns=3))
        self.assertEqual(len(stub.bodies), 1)  # やり直さない (0.4.0 と同じ)
        self.assertEqual(p.describe_failure(err), ("incomplete", "num_turns 3"))  # 前置きなし
        stub, out, err = self._run(lambda b, i: _ok(b.rstrip("。」？！")))
        self.assertIsNone(err)  # 末尾の確かめは、分けたときだけ

    def _fail_chunk2(self, failure):
        c2 = self.chunks[1].text

        def reply(b, i):
            if b in c2:  # まとまり 2 と、やり直しの片方
                return failure(b)
            return _ok(b)

        return self._run(reply)

    def test_each_guard_works_per_chunk_and_stops_the_rest(self):
        def raise_(e):
            def f(b):
                raise e
            return f

        cases = [
            ("empty", lambda b: {"num_turns": 1, "result": ""}, p.ProofreadEmpty, "空の結果"),
            ("too_short", lambda b: {"num_turns": 1, "result": b[: len(b) // 3]}, p.ProofreadTooShort, None),
            ("num_turns", lambda b: _ok(b, turns=3), p.ProofreadIncomplete, "num_turns 3"),
            ("paragraphs", lambda b: _ok("\n\n".join(b.split("\n\n")[1:])), p.ProofreadIncomplete, None),
            ("end", lambda b: _ok(b.rstrip()[:-1]), p.ProofreadIncomplete, "末尾が文の終わりでない"),
            ("note", lambda b: _ok(b + "\n\n（以上、合成の注記を足しました）"), p.ProofreadIncomplete,
             "末尾が文の終わりでない"),
            ("is_error", raise_(p.ClaudeCLIResultError("合成の失敗", detail="is_error (num_turns 4)")),
             p.ClaudeCLIResultError, "is_error (num_turns 4)"),
        ]
        for name, failure, exc_type, detail in cases:
            with self.subTest(case=name):
                stub, out, err = self._fail_chunk2(failure)
                self.assertIsNone(out)
                self.assertIsInstance(err, exc_type)
                self.assertEqual(err.proofread_chunk, (2, self.n))
                code, got = p.describe_failure(err)
                self.assertTrue(got.startswith(f"まとまり 2/{self.n}: "), got)
                if detail is not None:
                    self.assertEqual(got, f"まとまり 2/{self.n}: {detail}")
                # まとまり 1、まとまり 2、やり直しの前半 (ここで止まる)。まとまり 3 以降は呼ばない
                self.assertEqual(stub.bodies[:2], [self.chunks[0].text, self.chunks[1].text])
                self.assertEqual(len(stub.bodies), 3)
                self.assertTrue(self.chunks[1].text.startswith(stub.bodies[2]))
                self.assertNotIn(self.chunks[2].text, stub.bodies)

    def test_per_chunk_guard_finds_what_the_whole_body_guard_misses(self):
        # まとまり 1 で段落が 3 つ増え、まとまり 2 で 3 つ欠ける。全体の比と段落の数は 0.4.0 のガードを通る
        c1, c2 = self.chunks[0].text, self.chunks[1].text
        c1_split = c1.replace("です。", "です。\n\n", 3)
        c2_dropped = "\n\n".join(c2.split("\n\n")[3:])
        whole = "\n\n".join([c1_split, c2_dropped] + [c.text for c in self.chunks[2:]])
        p.check_result(self.body, whole, num_turns=1)  # 全体で比べると見逃す

        def reply(b, i):
            if b == c1:
                return _ok(c1_split)
            if b in c2:
                return _ok("\n\n".join(b.split("\n\n")[3 if b == c2 else 1:]))
            return _ok(b)

        stub, out, err = self._run(reply)
        self.assertIsInstance(err, p.ProofreadIncomplete)
        self.assertEqual(p.describe_failure(err)[1].split(": ")[0], f"まとまり 2/{self.n}")

    def test_retry_in_two_halves_can_pass(self):
        c2 = self.chunks[1].text
        stub, out, err = self._run(lambda b, i: _ok(b, turns=3 if b == c2 else 1))
        self.assertIsNone(err)
        self.assertEqual(out, self.body.replace("えーと、", ""))
        halves = p.split_in_two(c2)
        self.assertEqual(len(halves), 2)
        self.assertTrue(all(len(h.text) <= len(c2) * 0.6 for h in halves))
        self.assertEqual(stub.bodies, [self.chunks[0].text, c2] + [h.text for h in halves]
                         + [c.text for c in self.chunks[2:]])

    def test_timeout_and_exit_errors_are_not_retried(self):
        cases = [
            (subprocess.TimeoutExpired(["claude"], 900), subprocess.TimeoutExpired, "timeout", "900 秒で返らなかった"),
            (p.ClaudeCLIError("合成の失敗", detail="exit 1"), p.ClaudeCLIError, "cli_error", "exit 1"),
        ]
        for exc, exc_type, code, detail in cases:
            with self.subTest(code=code):
                def failure(b, exc=exc):
                    raise exc
                stub, out, err = self._fail_chunk2(failure)
                self.assertIsInstance(err, exc_type)
                self.assertEqual(stub.bodies, [self.chunks[0].text, self.chunks[1].text])  # やり直さず、残りも呼ばない
                self.assertEqual(p.describe_failure(err), (code, f"まとまり 2/{self.n}: {detail}"))


class DescribeChunkFailureTest(unittest.TestCase):
    def test_chunk_prefix(self):
        e = p.ProofreadIncomplete("x", detail="num_turns 3")
        self.assertEqual(p.describe_failure(e), ("incomplete", "num_turns 3"))
        e.proofread_chunk = (3, 5)
        self.assertEqual(p.describe_failure(e), ("incomplete", "まとまり 3/5: num_turns 3"))
        t = subprocess.TimeoutExpired(["claude"], 900)
        t.proofread_chunk = (2, 5)
        self.assertEqual(p.describe_failure(t), ("timeout", "まとまり 2/5: 900 秒で返らなかった"))


def _alive(pid: int) -> bool:
    """Windows のプロセスがまだ動いているか (テストの後片付けの確かめ用。止めはしない)。"""
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        return bool(k32.GetExitCodeProcess(h, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        k32.CloseHandle(h)


@unittest.skipUnless(os.name == "nt", "Windows の subprocess.run の振る舞いを確かめるテスト")
class TimeoutDoesNotWaitForGrandchildTest(unittest.TestCase):
    """#4 の U5: 時間切れの後、stderr を持った孫を待たずに返る。

    本物の python の子と孫を、キットの _call_claude_json の引数のまま (コマンドだけ差し替えて) 走らせる。
    Claude CLI は呼ばない。孫は止めの印のファイルで終わる (印が無くても 10 秒で自分で終わる)。
    テストの後に、子と孫がどちらも終わったことを確かめる (PID で見るだけで、止めはしない)。
    """

    TIMEOUT = 2.0

    def test_returns_soon_after_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            stop, pids = Path(d) / "stop", Path(d) / "pids"
            wait_loop = (
                "t = time.time()\n"
                f"while not os.path.exists({str(stop)!r}) and time.time() - t < 10: time.sleep(0.05)\n"
            )
            note_pid = f"import os, subprocess, sys, time\nopen({str(pids)!r}, 'a').write(str(os.getpid()) + '\\n')\n"
            grandchild = note_pid + wait_loop
            # 孫に stderr をそのまま渡す (pipe なら、孫が終わるまで subprocess.run が返らない形)
            child = (note_pid + f"subprocess.Popen([sys.executable, '-c', {grandchild!r}], stderr=sys.stderr)\n"
                     + wait_loop)
            real_run = subprocess.run

            def run(cmd, *a, **k):
                k["timeout"] = self.TIMEOUT
                return real_run([sys.executable, "-c", child], *a, **k)

            try:
                with mock.patch.object(p, "find_claude_bin", lambda: Path("claude-stub.exe")), \
                        mock.patch.object(p.subprocess, "run", run), \
                        mock.patch.object(p.tempfile, "tempdir", d):
                    t0 = time.perf_counter()
                    with self.assertRaises(subprocess.TimeoutExpired):
                        p._call_claude_json("S", "合成の依頼", "m")
                    elapsed = time.perf_counter() - t0
                seen = [int(x) for x in pids.read_text().split()] if pids.exists() else []
                self.assertEqual(len(seen), 2, "子と孫が時間切れの前に起きていない")
                self.assertLess(elapsed, self.TIMEOUT + 3.0)  # pipe なら孫の終わり (最大 10 秒) まで待つ
            finally:
                stop.touch()
                seen = [int(x) for x in pids.read_text().split()] if pids.exists() else []
                deadline = time.time() + 15
                while any(_alive(pid) for pid in seen) and time.time() < deadline:
                    time.sleep(0.05)
                time.sleep(0.2)  # 終わったプロセスがファイルを手放すのを待つ
            self.assertFalse([pid for pid in seen if _alive(pid)], "子か孫が終わらなかった")


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
