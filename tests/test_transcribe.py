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
import proofread  # noqa: E402
import transcribe as t  # noqa: E402


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

    def load_model(self, name, device, compute_type, language, asr_options=None):
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


class ProcessOneTest(unittest.TestCase):
    """C5 (#8): 置き換え前の文を残す、校閲されていない本文を校閲済みにしない、失敗の理由を残す。"""

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
        engine.clear_model_cache()
        self.addCleanup(engine.clear_model_cache)
        patch = mock.patch.object(proofread, "find_claude_bin", lambda: Path("claude-stub.exe"))
        patch.start()
        self.addCleanup(patch.stop)

    def _cli(self, reply="", *, num_turns=1, is_error=False, returncode=0, timeout=False):
        """subprocess.run のスタブ (Claude CLI)。reply は、校閲に渡った本文から result を作る関数か文字列。"""

        def fake(cmd, *a, **k):
            body = k["stdin"].read().split("【文字起こし本文】\n", 1)[1]
            self.cli_bodies.append(body)
            if timeout:
                raise subprocess.TimeoutExpired(cmd, k["timeout"])
            data = {"type": "result", "is_error": is_error, "result": reply(body) if callable(reply) else reply}
            if is_error:
                data["subtype"] = "error_during_execution"
            if num_turns is not None:
                data["num_turns"] = num_turns
            k["stdout"].write(json.dumps(data, ensure_ascii=False))
            return subprocess.CompletedProcess(cmd, returncode, None, "合成の stderr の文" if returncode else "")

        patch = mock.patch.object(proofread.subprocess, "run", fake)
        patch.start()
        self.addCleanup(patch.stop)

    def _run(self, *, segs=None, entries=None, use_proofread=True, with_timestamps=False) -> dict:
        engine.clear_model_cache()
        fake = _FakeWhisperx(self.SEGS if segs is None else segs)
        with mock.patch.dict(sys.modules, {"whisperx": fake}), contextlib.redirect_stdout(io.StringIO()):
            out = t.process_one(
                self.audio, cfg=self.cfg, processed_dir=self.root / "processed", out_dir=self.root / "out",
                entries=self.ENTRIES if entries is None else entries, with_timestamps=with_timestamps,
                keep=True, device="cpu", use_proofread=use_proofread,
            )
        return self._parse(out.read_text(encoding="utf-8"))

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


if __name__ == "__main__":
    unittest.main()
