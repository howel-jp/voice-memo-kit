"""transcribe.py の設定・パス解決・整形ロジックのテスト（WhisperX / GPU 不要）。

process_one は whisperx のモジュールと subprocess.run (Claude CLI) をスタブにして走らせる。
語と文は合成 (public repo のため、利用者の辞書の語・メモの中身を使わない)。

実行: python -m unittest discover -s tests   （キットのルートで）
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import engine  # noqa: E402
import glossary  # noqa: E402
import proofread  # noqa: E402
import transcribe as t  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "measure"))
import synth  # noqa: E402  (合成の辞書。標準ライブラリだけ)


def char_encoder(text: str):
    """1 字 = 1 トークンのトークナイザの代わり (initial_prompt とチャンクの出力を数える)。"""
    return list(range(len(text))), [(i, i + 1) for i in range(len(text))]


def _cfg(project: Path, global_root: Path | None, extra: list[str] | None = None) -> dict:
    cfg = t.load_config(project)
    cfg["global_root"] = str(global_root) if global_root else None
    if extra is not None:
        cfg["paths"]["extra_glossary_dirs"] = extra
    return cfg


class DeepMergeTest(unittest.TestCase):
    def test_nested_override_keeps_siblings(self):
        base = {"paths": {"inbox": "a", "glossary": "g"}, "whisper": {"model": "large-v3"}}
        out = t._deep_merge(base, {"paths": {"inbox": "b"}})
        self.assertEqual(out["paths"], {"inbox": "b", "glossary": "g"})
        self.assertEqual(out["whisper"], {"model": "large-v3"})
        self.assertEqual(base["paths"]["inbox"], "a")  # 入力は変更しない


class LoadConfigTest(unittest.TestCase):
    def test_default_when_no_project_config(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = t.load_config(Path(d))
        self.assertEqual(cfg["paths"]["glossary"], "voice_memos/glossary")
        self.assertEqual(cfg["paths"]["extra_glossary_dirs"], [])
        self.assertEqual(cfg["global_root"], "~/voice_memos")
        self.assertEqual(cfg["whisper"]["device"], "cuda")

    def test_device_override_in_project_config(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "voice_memos").mkdir()
            (root / "voice_memos" / "config.json").write_text(
                json.dumps({"whisper": {"device": "cpu"}}), encoding="utf-8")
            cfg = t.load_config(root)
        self.assertEqual(cfg["whisper"]["device"], "cpu")
        self.assertEqual(cfg["whisper"]["model"], "large-v3")  # 他のキーは既定のまま

    def test_project_config_overrides_and_tolerates_bom(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "voice_memos").mkdir()
            (root / "voice_memos" / "config.json").write_text(
                json.dumps({"paths": {"extra_glossary_dirs": ["tools/glossary"]}, "proofread": {"enabled": False}}),
                encoding="utf-8-sig",
            )
            cfg = t.load_config(root)
        self.assertEqual(cfg["paths"]["extra_glossary_dirs"], ["tools/glossary"])
        self.assertFalse(cfg["proofread"]["enabled"])
        self.assertEqual(cfg["paths"]["inbox"], "voice_memos/inbox")  # 既定は残る


class ResolvePathsTest(unittest.TestCase):
    def test_layers_and_extra_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "proj"
            glob = Path(d) / "global"
            root.mkdir()
            cfg = _cfg(root, glob, extra=["shared", str(Path(d) / "abs_extra")])
            p = t.resolve_paths(cfg, root)
        self.assertEqual(p["glossary"], root / "voice_memos" / "glossary")
        self.assertEqual(p["extra_glossary_dirs"], [root / "shared", Path(d) / "abs_extra"])
        self.assertEqual(p["global_inbox"], glob / "inbox")
        self.assertEqual(p["global_glossary"], glob / "glossary")
        self.assertEqual(p["global_processed"], glob / "processed")
        self.assertEqual(p["transcripts"], root / "voice_memos" / "transcripts")

    def test_no_global_root(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            cfg = _cfg(root, None)
            p = t.resolve_paths(cfg, root)
        self.assertIsNone(p["global_root"])
        self.assertIsNone(p["global_inbox"])
        self.assertIsNone(p["global_glossary"])


class CollectInputsTest(unittest.TestCase):
    def _make(self, d: str):
        root = Path(d) / "proj"
        glob = Path(d) / "global"
        (root / "voice_memos" / "inbox").mkdir(parents=True)
        (glob / "inbox").mkdir(parents=True)
        (root / "voice_memos" / "inbox" / "p1.m4a").write_bytes(b"")
        (root / "voice_memos" / "inbox" / "note.txt").write_bytes(b"")
        (glob / "inbox" / "g1.mp3").write_bytes(b"")
        cfg = _cfg(root, glob)
        return root, glob, t.resolve_paths(cfg, root)

    def test_project_first_then_global_with_processed_routing(self):
        with tempfile.TemporaryDirectory() as d:
            root, glob, p = self._make(d)
            items = t.collect_inputs(None, p, use_global=True)
            self.assertEqual([f.name for f, _ in items], ["p1.m4a", "g1.mp3"])
            self.assertEqual(items[0][1], root / "voice_memos" / "processed")
            self.assertEqual(items[1][1], glob / "processed")

    def test_no_global_flag(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, p = self._make(d)
            items = t.collect_inputs(None, p, use_global=False)
            self.assertEqual([f.name for f, _ in items], ["p1.m4a"])

    def test_explicit_file_under_global_inbox_routes_to_global_processed(self):
        with tempfile.TemporaryDirectory() as d:
            _, glob, p = self._make(d)
            items = t.collect_inputs(str(glob / "inbox" / "g1.mp3"), p, use_global=True)
            self.assertEqual(items[0][1], glob / "processed")


class MergedGlossaryTest(unittest.TestCase):
    def test_global_extra_project_precedence(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d) / "proj"
            glob = Path(d) / "global"
            for sub in ["voice_memos/glossary", "shared"]:
                (root / sub).mkdir(parents=True)
            (glob / "glossary").mkdir(parents=True)
            (glob / "glossary" / "g.json").write_text(
                json.dumps({"entries": [{"canonical": "X", "notes": "global"}, {"canonical": "G"}]}), encoding="utf-8")
            (root / "shared" / "s.json").write_text(
                json.dumps({"entries": [{"canonical": "X", "notes": "extra", "wrong_variants": ["x1"]}, {"canonical": "S"}]}),
                encoding="utf-8")
            (root / "voice_memos" / "glossary" / "p.json").write_text(
                json.dumps({"entries": [{"canonical": "X", "notes": "project", "wrong_variants": ["x2"]}]}), encoding="utf-8")
            cfg = _cfg(root, glob, extra=["shared"])
            p = t.resolve_paths(cfg, root)
            entries = t.load_merged_glossary(p, use_global=True)
        by = {e["canonical"]: e for e in entries}
        self.assertEqual(set(by), {"X", "G", "S"})
        self.assertEqual(by["X"]["notes"], "project")
        self.assertEqual(by["X"]["wrong_variants"], ["x1", "x2"])


class FormatBodyTest(unittest.TestCase):
    SEGS = [
        {"start": 0.0, "end": 1.0, "text": " こんにちは "},
        {"start": 1.2, "end": 2.0, "text": "今日は"},
        {"start": 5.0, "end": 6.0, "text": "次の段落"},
        {"start": 6.1, "end": 6.5, "text": "   "},
    ]

    def test_paragraph_split_on_gap_and_japanese_concat(self):
        self.assertEqual(t.format_body(self.SEGS, gap_sec=1.5, with_timestamps=False), "こんにちは今日は\n\n次の段落")

    def test_timestamps(self):
        out = t.format_body(self.SEGS, gap_sec=1.5, with_timestamps=True)
        self.assertEqual(out.split("\n\n"), ["[00:00] こんにちは今日は", "[00:05] 次の段落"])


class UniquePathTest(unittest.TestCase):
    def test_suffix_increments(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.md"
            self.assertEqual(t.unique_path(p), p)
            p.write_text("")
            (Path(d) / "x_1.md").write_text("")
            self.assertEqual(t.unique_path(p), Path(d) / "x_2.md")


class OneLineTest(unittest.TestCase):
    def test_one_line_and_bounded(self):
        out = t.one_line("合成の\n理由  です\r\n" + "x" * 300)
        self.assertNotIn("\n", out)
        self.assertTrue(out.startswith("合成の 理由 です x"))
        self.assertEqual(len(out), t.ERROR_DETAIL_MAX_CHARS)
        self.assertEqual(json.loads(json.dumps('引用 "x" と \\', ensure_ascii=False)), '引用 "x" と \\')


class _FakeWhisperx(types.ModuleType):
    """whisperx のスタブ。決めた segments を返す。"""

    def __init__(self, segments: list[dict]):
        super().__init__("whisperx")
        self._segments = segments
        self.asr_options: list = []  # load_model に渡った asr_options

    def load_model(self, name, device, compute_type, language, asr_options=None):
        self.asr_options.append(asr_options)
        segments = self._segments

        class _Model:
            def transcribe(self, audio, batch_size, language, chunk_size):
                return {"segments": [dict(s) for s in segments]}

        return _Model()

    def load_audio(self, path):
        return [0.0] * 16000 * 10


def _segs(texts: list[str]) -> list[dict]:
    """1 つの text を 1 段落にする segments (間を 3 秒あける。段落の区切りは 1.5 秒)。"""
    return [{"start": i * 5.0, "end": i * 5.0 + 2.0, "text": tx} for i, tx in enumerate(texts)]


class _ProcessOneBase(unittest.TestCase):
    """process_one を whisperx と subprocess.run (Claude CLI) のスタブで走らせる土台。"""

    # 合成の語と文。wrong_variants の「ぞるま」が whisperx のスタブの出力に 2 回出る
    ENTRIES = [{"canonical": "ゾルマ", "wrong_variants": ["ぞるま"], "source": "synthetic"}]
    SEGS = [
        {"start": 0.0, "end": 2.0, "text": "えーと、今日はぞるまの件です。"},
        {"start": 2.2, "end": 4.0, "text": "次に進めます。"},
        {"start": 7.0, "end": 9.0, "text": "最後にぞるまを確かめます。"},
    ]
    RAW = "えーと、今日はぞるまの件です。次に進めます。\n\n最後にぞるまを確かめます。"
    REPLACED = "えーと、今日はゾルマの件です。次に進めます。\n\n最後にゾルマを確かめます。"
    KEYS = ["date", "source", "duration", "engine", "segments", "glossary_replacements", "proofread"]
    ERROR_KEYS = KEYS + ["proofread_error", "proofread_error_detail"]

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.audio = self.root / "memo.m4a"
        self.audio.write_bytes(b"")
        self.cfg = t.load_config(self.root)
        self.model = self.cfg["proofread"]["model"]
        self.cli_bodies: list[str] = []
        self.cli_messages: list[str] = []
        engine.clear_model_cache()
        self.addCleanup(engine.clear_model_cache)
        patch = mock.patch.object(proofread, "find_claude_bin", lambda: Path("claude-stub.exe"))
        patch.start()
        self.addCleanup(patch.stop)
        # initial_prompt を数えるトークナイザは差し替える (手元の HF キャッシュを読まない)
        self.encoder_calls: list[str] = []
        self.set_encoder(char_encoder, "")

    def set_encoder(self, encode, note: str):
        def fake_load(model_name="large-v3"):
            self.encoder_calls.append(model_name)
            return encode, note

        patch = mock.patch.object(t, "load_prompt_encoder", fake_load)
        patch.start()
        self.addCleanup(patch.stop)

    def _cli(self, reply="", *, num_turns=1, is_error=False, returncode=0, timeout=False):
        """subprocess.run のスタブ (Claude CLI)。reply は、校閲に渡った本文から result を作る関数か文字列。

        num_turns は数か、校閲に渡った本文から数を作る関数 (分けた校閲で、まとまりごとに変えるため)。
        """

        def fake(cmd, *a, **k):
            message = k["stdin"].read()
            self.cli_messages.append(message)
            body = message.split("【文字起こし本文】\n", 1)[1]
            self.cli_bodies.append(body)
            if timeout:
                raise subprocess.TimeoutExpired(cmd, k["timeout"])
            data = {"type": "result", "is_error": is_error, "result": reply(body) if callable(reply) else reply}
            if is_error:
                data["subtype"] = "error_during_execution"
            turns = num_turns(body) if callable(num_turns) else num_turns
            if turns is not None:
                data["num_turns"] = turns
            k["stdout"].write(json.dumps(data, ensure_ascii=False))
            if returncode:
                k["stderr"].write("合成の stderr の文")  # stderr は一時ファイル (#4 の U5)
            return subprocess.CompletedProcess(cmd, returncode, None, None)

        patch = mock.patch.object(proofread.subprocess, "run", fake)
        patch.start()
        self.addCleanup(patch.stop)

    def _run(self, *, segs=None, entries=None, use_proofread=True, with_timestamps=False) -> dict:
        engine.clear_model_cache()
        fake = _FakeWhisperx(self.SEGS if segs is None else segs)
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"whisperx": fake}), contextlib.redirect_stdout(buf):
            out = t.process_one(
                self.audio, cfg=self.cfg, processed_dir=self.root / "processed", out_dir=self.root / "out",
                entries=self.ENTRIES if entries is None else entries, with_timestamps=with_timestamps,
                keep=True, device="cpu", use_proofread=use_proofread,
            )
        self.whisperx = fake
        self.stdout = buf.getvalue()
        return {**self._parse(out.read_text(encoding="utf-8")), "raw": out.read_bytes()}

    @staticmethod
    def _parse(md: str) -> dict:
        assert md.startswith("---\n")
        end = md.index("\n---\n", 4)
        lines = md[4:end].splitlines()
        fm = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip() for ln in lines}
        rest = md[end + 5:]
        head = rest.split("<details>", 1)[0].strip()
        title, _, body = head.partition("\n\n")
        assert title.startswith("# ボイスメモ "), title
        return {
            "keys": [ln.split(":", 1)[0] for ln in lines],
            "fm": fm,
            "body": body,
            "summaries": re.findall(r"<summary>([^<]*)</summary>", rest),
            "details": dict(re.findall(r"<details>\n<summary>([^<]*)</summary>\n\n(.*?)\n\n</details>", rest, re.DOTALL)),
            "md": md,
        }



class ProcessOneTest(_ProcessOneBase):
    """C5 (#8): 置き換え前の文を残す、校閲されていない本文を校閲済みにしない、失敗の理由を残す。"""

    # --- C5-a: 置き換え前の文が 3 通りとも残る -------------------------------------------

    def test_success_keeps_pre_proofread_and_pre_replacement(self):
        self._cli(lambda b: b.replace("えーと、", ""))
        r = self._run()
        self.assertEqual(r["keys"], self.KEYS)  # 今の 7 キーが同じ順。足すキーは付かない
        self.assertEqual((r["fm"]["proofread"], r["fm"]["glossary_replacements"]), (self.model, "2"))
        self.assertEqual(r["body"], self.REPLACED.replace("えーと、", ""))
        self.assertEqual(r["summaries"], [t.PRE_PROOFREAD_SUMMARY, t.PRE_REPLACEMENT_SUMMARY])
        self.assertEqual(r["details"][t.PRE_PROOFREAD_SUMMARY], self.REPLACED)
        self.assertEqual(r["details"][t.PRE_REPLACEMENT_SUMMARY], self.RAW)
        self.assertEqual(self.cli_bodies, [self.REPLACED])  # 校閲に渡るのは置き換え後

    def test_failure_keeps_pre_replacement(self):
        self._cli(timeout=True)
        r = self._run()
        self.assertEqual(r["keys"], self.ERROR_KEYS)
        self.assertEqual(r["fm"]["proofread"], "false")
        self.assertEqual(r["fm"]["proofread_error"], "timeout")
        self.assertEqual(json.loads(r["fm"]["proofread_error_detail"]), "900 秒で返らなかった")
        self.assertEqual(r["body"], self.REPLACED)
        self.assertEqual(r["summaries"], [t.PRE_REPLACEMENT_SUMMARY])  # 校閲前の節は本文と同じなので付かない
        self.assertEqual(r["details"][t.PRE_REPLACEMENT_SUMMARY], self.RAW)

    def test_no_proofread_keeps_pre_replacement(self):
        self._cli("使われない返り")
        r = self._run(use_proofread=False)
        self.assertEqual(self.cli_bodies, [])
        self.assertEqual(r["keys"], self.KEYS)
        self.assertEqual(r["fm"]["proofread"], "false")
        self.assertEqual(r["body"], self.REPLACED)
        self.assertEqual(r["summaries"], [t.PRE_REPLACEMENT_SUMMARY])
        self.assertEqual(r["details"][t.PRE_REPLACEMENT_SUMMARY], self.RAW)

    def test_pre_replacement_follows_timestamps(self):
        self._cli(lambda b: b)
        r = self._run(with_timestamps=True)
        self.assertEqual(r["details"][t.PRE_REPLACEMENT_SUMMARY],
                         "[00:00] えーと、今日はぞるまの件です。次に進めます。\n\n[00:07] 最後にぞるまを確かめます。")
        self.assertEqual(self.cli_bodies, [self.REPLACED])  # 校閲に渡す文には時刻を付けない

    def test_no_replacement_omits_the_section_because_it_is_the_same_text(self):
        # 置き換えが 0 箇所なら、置き換え前の文は本文か校閲前の節と同じ (SKILL.md §3)
        no_hit = [{"canonical": "ゾルマ", "wrong_variants": ["ぬるぽ"]}]
        for entries in ([], no_hit):
            with self.subTest(entries=entries):
                self._cli(lambda b: b.replace("えーと、", ""))
                ok = self._run(entries=entries)
                self.assertEqual(ok["fm"]["glossary_replacements"], "0")
                self.assertEqual(ok["summaries"], [t.PRE_PROOFREAD_SUMMARY])
                self.assertEqual(ok["details"][t.PRE_PROOFREAD_SUMMARY], self.RAW)
                self._cli(timeout=True)
                failed = self._run(entries=entries)
                self.assertEqual((failed["summaries"], failed["body"]), ([], self.RAW))
                off = self._run(entries=entries, use_proofread=False)
                self.assertEqual((off["summaries"], off["body"]), ([], self.RAW))

    # --- C5-b・C5-c: 校閲されていない本文を校閲済みにせず、理由を残す --------------------

    def test_rejected_results_are_not_marked_proofread(self):
        cases = [
            ({"reply": ""}, "empty", "空の結果"),
            ({"reply": "```\n```"}, "empty", "空の結果"),
            ({"reply": "合成の途中の本文", "is_error": True, "num_turns": 4}, "cli_error",
             "is_error (subtype error_during_execution, num_turns 4)"),
            ({"reply": "合成の途中の本文", "returncode": 1}, "cli_error", "exit 1"),
            ({"reply": "短い。"}, "too_short", None),
            ({"reply": lambda b: b, "num_turns": 2}, "incomplete", "num_turns 2"),
        ]
        for kwargs, code, detail in cases:
            with self.subTest(code=code, detail=detail):
                self._cli(**kwargs)
                r = self._run()
                self.assertEqual(r["keys"], self.ERROR_KEYS)
                self.assertEqual(r["fm"]["proofread"], "false")
                self.assertEqual(r["fm"]["proofread_error"], code)
                got = json.loads(r["fm"]["proofread_error_detail"])
                if detail is not None:
                    self.assertEqual(got, detail)
                self.assertEqual(r["body"], self.REPLACED)
                self.assertEqual(r["summaries"], [t.PRE_REPLACEMENT_SUMMARY])
                self.assertNotIn("合成の途中の本文", r["md"])  # 使わなかった返りは md に入れない
                self.assertNotIn("合成の stderr の文", r["md"])

    def test_u4_run1_shape_is_not_marked_proofread(self):
        # #4 の U4 run1 と同じ形: 前の段落 (72 のうち 27) を落とした返り。比は 60% を超える
        texts = [f"合成の段落{i:02d}でぞるまの話をします。続けて話します。" for i in range(72)]

        def drop_front(body: str) -> str:
            return "\n\n".join(body.split("\n\n")[27:])

        for num_turns, detail in ((3, "num_turns 3"), (1, "段落 72 → 45")):
            with self.subTest(num_turns=num_turns):
                self._cli(drop_front, num_turns=num_turns)
                r = self._run(segs=_segs(texts))
                sent = self.cli_bodies[-1]
                self.assertGreater(len(drop_front(sent)) / len(sent), proofread.MIN_LENGTH_RATIO)
                self.assertEqual(r["fm"]["proofread"], "false")
                self.assertEqual(r["fm"]["proofread_error"], "incomplete")
                self.assertEqual(json.loads(r["fm"]["proofread_error_detail"]), detail)
                self.assertEqual(r["body"], sent)  # 本文は校閲前の全文 (72 段落)
                self.assertEqual(proofread.count_paragraphs(r["body"]), 72)
                self.assertNotIn(t.PRE_PROOFREAD_SUMMARY, r["summaries"])
                self.assertIn("ぞるま", r["details"][t.PRE_REPLACEMENT_SUMMARY])

    def test_empty_transcript_is_not_marked_proofread_and_does_not_call_cli(self):
        self._cli("使われない返り")
        r = self._run(segs=[{"start": 0.0, "end": 1.0, "text": "   "}])
        self.assertEqual(self.cli_bodies, [])
        self.assertEqual(r["keys"], self.KEYS)  # 校閲を試みていないので理由のキーは付かない
        self.assertEqual(r["fm"]["proofread"], "false")
        self.assertEqual((r["body"], r["summaries"]), ("", []))

    def test_crlf_reply_is_proofread_and_written_without_cr_doubling(self):
        # PR #9 のレビュー 1 回目の任意 1: CRLF の返りを incomplete で捨てず、md に \r を重ねない
        self._cli(lambda b: b.replace("えーと、", "").replace("\n", "\r\n"))
        r = self._run()
        self.assertEqual(r["fm"]["proofread"], self.model)
        self.assertNotIn("proofread_error", r["fm"])
        self.assertEqual(r["body"], self.REPLACED.replace("えーと、", ""))
        self.assertNotIn(b"\r\r", r["raw"])


class ProcessOneChunkedTest(_ProcessOneBase):
    """#2: 3,000 字を超える文字起こしは分けて校閲する。1 つでも通らなければ、本文全体を校閲前にする (C5)。"""

    # 合成の 100 段落 (どれも文の終わりで終わる)。どの段落にも置き換え前の語「ぞるま」が 1 回出る
    TEXTS = [f"えーと、合成の段落{i:02d}でぞるまの話をします。続けて段取りを確かめます。"
             f"それから資料の置き場所も決めておきたいです。最後に次の予定を話します。" for i in range(100)]

    def setUp(self):
        super().setUp()
        self.segs = _segs(self.TEXTS)
        self.raw = "\n\n".join(self.TEXTS)
        self.replaced = self.raw.replace("ぞるま", "ゾルマ")
        self.chunks = proofread.split_into_chunks(self.replaced)
        self.assertEqual(len(self.chunks), 3)

    def _run_long(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return self._run(segs=self.segs)

    def test_all_chunks_pass(self):
        self._cli(lambda b: b.replace("えーと、", ""))
        r = self._run_long()
        self.assertEqual(self.cli_bodies, [c.text for c in self.chunks])
        self.assertTrue(all(len(b) <= proofread.CHUNK_CHARS for b in self.cli_bodies))
        self.assertEqual(r["keys"], self.KEYS)  # 今の 7 キーが同じ順。足すキーは付かない
        self.assertEqual((r["fm"]["proofread"], r["fm"]["glossary_replacements"]), (self.model, "100"))
        self.assertEqual(r["body"], self.replaced.replace("えーと、", ""))  # つないだ校閲後
        self.assertEqual(proofread.count_paragraphs(r["body"]), 100)  # 入力の段落が欠けずに残る
        self.assertEqual(r["summaries"], [t.PRE_PROOFREAD_SUMMARY, t.PRE_REPLACEMENT_SUMMARY])
        self.assertEqual(r["details"][t.PRE_PROOFREAD_SUMMARY], self.replaced)  # 校閲前の節は全体
        self.assertEqual(r["details"][t.PRE_REPLACEMENT_SUMMARY], self.raw)

    def test_one_chunk_failing_marks_the_whole_memo_unproofread(self):
        c2 = self.chunks[1].text
        # まとまり 2 とそのやり直しは num_turns 3 (欠けの疑い)。ほかは通る
        self._cli(lambda b: b.replace("えーと、", ""), num_turns=lambda b: 3 if b in c2 else 1)
        r = self._run_long()
        self.assertEqual(r["keys"], self.ERROR_KEYS)
        self.assertEqual(r["fm"]["proofread"], "false")  # 校閲されていない部分があるので、校閲済みと読めない
        self.assertEqual(r["fm"]["proofread_error"], "incomplete")
        self.assertEqual(json.loads(r["fm"]["proofread_error_detail"]), "まとまり 2/3: num_turns 3")
        self.assertEqual(r["body"], self.replaced)  # 本文は全体が校閲前 (通ったまとまり 1 の校閲後も使わない)
        self.assertEqual(r["summaries"], [t.PRE_REPLACEMENT_SUMMARY])
        self.assertEqual(r["details"][t.PRE_REPLACEMENT_SUMMARY], self.raw)
        self.assertNotIn(self.chunks[2].text, self.cli_bodies)  # まとまり 3 は呼ばない
        self.assertEqual(len(self.cli_bodies), 3)  # まとまり 1、2、2 のやり直しの前半

    def test_retry_that_passes_keeps_the_memo_proofread(self):
        c2 = self.chunks[1].text
        self._cli(lambda b: b.replace("えーと、", ""), num_turns=lambda b: 3 if b == c2 else 1)
        r = self._run_long()
        self.assertEqual(r["fm"]["proofread"], self.model)
        self.assertEqual(r["body"], self.replaced.replace("えーと、", ""))
        self.assertEqual(len(self.cli_bodies), 5)  # まとまり 3 つ + まとまり 2 のやり直し 2 つ


class ProcessOnePromptTest(_ProcessOneBase):
    """#1 (C1・C2): 文字起こしのたびに、initial_prompt のトークン数と届く語・届かない語の数がログに出る。

    トークナイザは 1 字 1 トークンの代わり。辞書は合成 (tools/measure/synth.py の mixed 78 項目)。
    """

    MIXED78 = synth.mixed_entries(78, 0)

    def _prompt(self) -> str:
        opts = self.whisperx.asr_options[-1]
        return opts["initial_prompt"] if opts else ""

    def _lines(self) -> list[str]:
        return self.stdout.splitlines()

    def test_fits_logs_tokens_and_passes_the_050_string(self):
        self._cli(lambda b: b)
        r = self._run()
        expected = glossary.build_initial_prompt(self.ENTRIES)
        self.assertEqual(self._prompt(), expected)
        self.assertIn(f"[A] initial_prompt {len(' ' + expected)}/223 トークン、届く 1/1 語", self._lines())
        self.assertNotIn("[WARN] initial_prompt", self.stdout)
        self.assertNotIn("届かない語", self.stdout)
        self.assertEqual(self.encoder_calls, [self.cfg["whisper"]["model"]])  # モデルと同じトークナイザ
        self.assertEqual(r["keys"], self.KEYS)  # md の項目は変わらない

    def test_overflow_without_marks_passes_the_050_string_and_warns(self):
        self._cli(lambda b: b)
        r = self._run(entries=self.MIXED78)
        expected = glossary.build_initial_prompt(self.MIXED78)
        self.assertEqual(self._prompt(), expected)  # 印の無い辞書は 0.5.0 と同じ文字列
        plan = glossary.plan_initial_prompt(self.MIXED78, char_encoder)
        n_reach, n_miss = len(plan.reached), len(plan.missed)
        self.assertGreater(n_miss, 0)
        lines = self._lines()
        self.assertIn(f"[A] initial_prompt {len(' ' + expected)} トークン（Whisper が使うのは後ろ 223）、"
                      f"届く {n_reach}/78 語、届かない {n_miss} 語", lines)
        warn = [ln for ln in lines if "[WARN] initial_prompt" in ln]
        self.assertEqual(len(warn), 1)
        self.assertTrue(warn[0].startswith("  [WARN] initial_prompt が 223 トークンを超えるため"))
        self.assertIn(f"前置きの文と {n_miss} 語は Whisper に届きません（置き換えと校閲には使います）", warn[0])
        self.assertIn('"priority": true', warn[0])
        self.assertIn("  届かない語: " + "、".join(plan.missed), lines)
        self.assertEqual(r["keys"], self.KEYS)
        # 届かない語も、校閲の canonical の一覧には全部渡す
        canon_line = self.cli_messages[-1].split("【固有名詞 canonical 一覧（最優先で守る）】\n", 1)[1].split("\n", 1)[0]
        self.assertEqual(canon_line, "、".join(e["canonical"] for e in self.MIXED78))

    def test_overflow_with_marks_passes_the_selected_prompt(self):
        entries = [dict(e, priority=True) if i % 4 == 0 else dict(e) for i, e in enumerate(self.MIXED78)]
        self._cli(lambda b: b)
        r = self._run(entries=entries)
        plan = glossary.plan_initial_prompt(entries, char_encoder)
        self.assertTrue(plan.selected)
        self.assertEqual(self._prompt(), plan.text)
        self.assertLessEqual(len(" " + plan.text), 223)
        lines = self._lines()
        self.assertIn(f"[A] initial_prompt {plan.tokens}/223 トークン（印の語を優先）、"
                      f"届く {len(plan.reached)}/78 語（うち印 20/20）、届かない {len(plan.missed)} 語", lines)
        warn = [ln for ln in lines if "[WARN] initial_prompt" in ln]
        self.assertEqual(warn, [f"  [WARN] initial_prompt が 223 トークンに収まらないため、{len(plan.missed)} 語は Whisper に"
                                "予告しません（置き換えと校閲には使います）"])
        self.assertIn("  届かない語: " + "、".join(plan.missed), lines)
        self.assertEqual(r["keys"], self.KEYS)

    def test_not_countable_warns_and_passes_the_050_string(self):
        self.set_encoder(None, "合成の理由")
        self._cli(lambda b: b)
        self._run(entries=self.MIXED78)
        self.assertEqual(self._prompt(), glossary.build_initial_prompt(self.MIXED78))
        self.assertIn("  [WARN] initial_prompt のトークン数を数えられません（合成の理由）。"
                      "Whisper が使うのは後ろ 223 トークンまでで、届く語の数は分かりません", self._lines())
        self.assertNotIn("[A] initial_prompt", self.stdout)
        self.assertNotIn("[A] チャンクの出力", self.stdout)

    def test_no_glossary_has_no_prompt_line_but_logs_chunks(self):
        self._cli(lambda b: b)
        self._run(entries=[])
        self.assertIsNone(self.whisperx.asr_options[-1])
        self.assertNotIn("initial_prompt", self.stdout)
        top = max(len(s["text"]) for s in self.SEGS)
        self.assertIn(f"[A] チャンクの出力（数え直し）: 最大 {top} トークン / 3 チャンク（1 チャンクの上限 224）", self._lines())
        self.assertNotIn("[WARN] 1 チャンク", self.stdout)

    def test_chunk_over_200_tokens_warns_with_the_cap_for_a_full_prompt(self):
        segs = [{"start": 0.0, "end": 2.0, "text": "あ" * 210 + "。"}, {"start": 3.0, "end": 4.0, "text": "短い。"}]
        self._cli(lambda b: b)
        self._run(segs=segs, entries=self.MIXED78)  # prompt は 223 を超える → 1 チャンクの上限は 221
        lines = self._lines()
        self.assertIn("[A] チャンクの出力（数え直し）: 最大 211 トークン / 2 チャンク（1 チャンクの上限 221）", lines)
        self.assertIn("  [WARN] 1 チャンクの出力が 200 トークンを超えました（1 チャンク、最大 211）。"
                      "上限 221 に当たると、そのチャンクの末尾は黙って切れます（数は文を数え直した目安）", lines)


class PromptLogLinesTest(unittest.TestCase):
    """prompt_log_lines の、process_one のテストで通らない形と、チャンクの出力の上限。"""

    def test_only_the_prefix_is_cut(self):
        plan = glossary.PromptPlan("合成", 230, 2, ["ゾルマ", "Qantrel"], [], False, 0, 0, False)
        lines = t.prompt_log_lines(plan)
        self.assertEqual(lines[0], "[A] initial_prompt 230 トークン（Whisper が使うのは後ろ 223）、届く 2/2 語、届かない 0 語")
        self.assertEqual(lines[1], "  [WARN] initial_prompt が 223 トークンを超えるため、前置きの文の前の方は Whisper に届きません"
                                   "（語は全部届きます）")
        self.assertEqual(len(lines), 2)

    def test_marked_words_that_do_not_fit_are_counted(self):
        plan = glossary.PromptPlan("合成", 220, 40, ["A"] * 15, ["B"] * 25, True, 40, 15, True)
        lines = t.prompt_log_lines(plan)
        self.assertEqual(lines[0], "[A] initial_prompt 220/223 トークン（印の語を優先）、届く 15/40 語（うち印 15/40）、届かない 25 語")
        self.assertTrue(lines[1].endswith("。印の語のうち 25 語も入りません"))

    def test_no_words_no_lines(self):
        self.assertEqual(t.prompt_log_lines(glossary.plan_initial_prompt([], char_encoder)), [])

    def test_chunk_output_cap(self):
        # 1 チャンクの出力 ≤ min(448 − len(prompt) + 1, 224)。len(prompt) = 使う prompt + 5 (無ければ 4)
        for tokens, cap in ((0, 224), (1, 224), (220, 224), (221, 223), (222, 222), (223, 221), (620, 221)):
            with self.subTest(tokens=tokens):
                self.assertEqual(t.chunk_output_cap(tokens), cap)


class PromptDocsTest(unittest.TestCase):
    """#1 の C3: README に届く長さの目安と優先の付け方があり、ログの例がコードの出す行と同じ。SKILL.md は [WARN] を読む。"""

    def setUp(self):
        self.readme = (Path(t.SKILL_DIR) / "README.md").read_text(encoding="utf-8")
        self.skill = (Path(t.SKILL_DIR) / "SKILL.md").read_text(encoding="utf-8")

    def test_readme_has_the_rule_of_thumb_and_the_mark(self):
        self.assertIn("223 トークン ≒ 前置きの文 24 トークン + 語およそ 25 語。語の長さで 18〜41 語", self.readme)
        self.assertIn('"priority": true', self.readme)
        self.assertIn("### initial_prompt の長さと、届ける語の優先", self.readme)

    def test_readme_log_examples_are_what_the_code_prints(self):
        over = glossary.PromptPlan("x", 620, 78, ["w"] * 28, ["v"] * 50, False, 0, 0, False)
        examples = [  # (plan, 何行目, README で「。…」と略した所)
            (glossary.PromptPlan("x", 113, 15, ["w"] * 15, [], True, 0, 0, False), 0, None),
            (over, 0, None),
            (over, 1, "。届けたい"),
            (glossary.PromptPlan("x", 223, 78, ["w"] * 31, ["v"] * 47, True, 20, 20, True), 0, None),
        ]
        for plan, i, cut in examples:
            line = t.prompt_log_lines(plan)[i]
            shown = line if cut is None else line[: line.index(cut)] + "。…"
            with self.subTest(line=shown):
                self.assertIn(shown, self.readme)

    def test_skill_md_reads_the_warn_lines(self):
        self.assertIn("[WARN] initial_prompt", self.skill)
        self.assertIn('"priority": true', self.skill)
        self.assertIn("[WARN] 1 チャンクの出力が", self.skill)


class HeadingDocsTest(unittest.TestCase):
    """PR #9 のレビュー 1 回目の任意 2: <details> の見出しの文を、SKILL.md・README.md と結ぶ (C5-d)。"""

    def test_headings_are_documented(self):
        for doc in ("SKILL.md", "README.md"):
            text = (Path(t.SKILL_DIR) / doc).read_text(encoding="utf-8")
            for summary in (t.PRE_PROOFREAD_SUMMARY, t.PRE_REPLACEMENT_SUMMARY):
                with self.subTest(doc=doc, summary=summary):
                    self.assertIn(summary, text)

    def test_pre_proofread_heading_keeps_its_prefix(self):
        # tools/measure/transcript_stats.py は <summary>校閲前…</summary> で当てる (0.3.x の md も同じ先頭)
        self.assertTrue(t.PRE_PROOFREAD_SUMMARY.startswith("校閲前"))
        self.assertFalse(t.PRE_REPLACEMENT_SUMMARY.startswith("校閲前"))


if __name__ == "__main__":
    unittest.main()
