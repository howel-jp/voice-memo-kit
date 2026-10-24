"""tools/measure (#4 の測りの道具) のテスト。GPU・WhisperX・Claude CLI・トークナイザのファイル無しで動く。

確かめること:
- 合成のデータが、同じ種なら同じになり、決めた字数・形になる。
- 届く語の数え方 (後ろ 223 トークン、1 字が 2 トークンに分かれる境目) を、スタブのトークナイザで。
- 実辞書・transcript のモードが、語・本文・パス・プロジェクト名を出力しない (public repo のため必須)。
- 校閲の時間の道具が、timeout だけを替えてキットの経路を通り、--call-claude 無しでは CLI を呼ばない。
- import しただけでは、重い import・プロセスの起動・外部への接続をしない。
"""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "measure"))
sys.path.insert(0, str(ROOT / "scripts"))

import glossary  # noqa: E402
import proofread  # noqa: E402
import proofread_timing as pt  # noqa: E402
import prompt_tokens as ptok  # noqa: E402
import synth  # noqa: E402
import timeout_wait as tw  # noqa: E402
import transcript_stats as ts  # noqa: E402


def char_encoder(text: str):
    """1 字 = 1 トークンのスタブ。"""
    return list(range(len(text))), [(i, i + 1) for i in range(len(text))]


def split_char_encoder(text: str):
    """1 字 = 2 トークン (どちらもその字の位置) のスタブ。byte-level BPE で字が割れるときの代わり。"""
    ids, offs = [], []
    for i in range(len(text)):
        ids += [2 * i, 2 * i + 1]
        offs += [(i, i + 1), (i, i + 1)]
    return ids, offs


def _entries(words):
    return [{"canonical": w} for w in words]


class SynthTest(unittest.TestCase):
    def test_deterministic_and_seed_dependent(self):
        self.assertEqual(synth.synth_canonicals("M", 30, 0), synth.synth_canonicals("M", 30, 0))
        self.assertNotEqual(synth.synth_canonicals("M", 30, 0), synth.synth_canonicals("M", 30, 1))
        self.assertEqual(synth.synth_memo_text(3000, 0), synth.synth_memo_text(3000, 0))

    def test_profiles_have_declared_lengths_and_are_unique(self):
        for prof, (lo, hi) in synth.PROFILE_LEN.items():
            words = synth.synth_canonicals(prof, 120, 0)
            self.assertEqual(len(words), 120)
            self.assertEqual(len(set(words)), 120, prof)
            for w in words:
                self.assertTrue(lo <= len(w) <= hi, (prof, len(w)))
        self.assertTrue(all(w.isascii() for w in synth.synth_canonicals("E", 50, 0)))

    def test_entries_and_mixed(self):
        es = synth.synth_entries("S", 10, 0)
        self.assertEqual(len(es), 10)
        for e in es:
            self.assertNotEqual(e["wrong_variants"][0], e["canonical"])
        mixed = synth.mixed_entries(78, 0)
        self.assertEqual(len(mixed), 78)
        self.assertEqual(len({e["canonical"] for e in mixed}), 78)

    def test_memo_text_shape(self):
        text = synth.synth_memo_text(5000, 0)
        self.assertGreaterEqual(len(text), 5000)
        self.assertLess(len(text), 5000 + 1000)  # 段落 1 つ分を超えて伸びない
        paras = text.split("\n\n")
        self.assertGreater(len(paras), 5)
        self.assertTrue(all(p and "\n" not in p for p in paras))
        self.assertTrue(any(n in text for n in synth.memo_names(0)))

    def test_glossary_json_is_loadable_by_kit(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "synthetic.json").write_text(synth.glossary_json(synth.mixed_entries(78, 0)), encoding="utf-8")
            loaded = glossary.load_glossary(Path(d))
        self.assertEqual(len(loaded), 78)

    def test_tts_text(self):
        t = synth.synth_tts_text(400, 0)
        self.assertGreaterEqual(len(t), 400)
        self.assertNotIn("\n", t)


class PromptTokensAnalyzeTest(unittest.TestCase):
    WORDS = ["アルファ", "ベータ線", "ガンマ"]

    def test_prefix_matches_kit(self):
        self.assertEqual(glossary.build_initial_prompt(_entries(["X"])), ptok.PREFIX + "X。")
        self.assertEqual(ptok.KEEP, 223)

    def test_no_drop_all_reach(self):
        r = ptok.analyze(_entries(self.WORDS), char_encoder, keep=1000)
        self.assertEqual((r["dropped"], r["reach"], r["miss"]), (0, 3, 0))
        self.assertTrue(r["prefix_reaches"])
        self.assertEqual(r["tokens"], len(ptok.whisper_prompt_text(_entries(self.WORDS))))

    def test_front_is_dropped_back_reaches(self):
        text = ptok.whisper_prompt_text(_entries(self.WORDS))
        # 最後の 2 語 + 区切り + 句点 だけが残る長さ
        keep = len("ベータ線、ガンマ。")
        r = ptok.analyze(_entries(self.WORDS), char_encoder, keep=keep)
        self.assertEqual(r["dropped"], len(text) - keep)
        self.assertEqual(r["reach_flags"], [False, True, True])
        self.assertFalse(r["prefix_reaches"])

    def test_partially_cut_item_does_not_reach(self):
        # 「ベータ線」の先頭の字が 2 トークンに割れ、その 1 つ目だけが落ちる
        keep_chars = len("ベータ線、ガンマ。")
        r = ptok.analyze(_entries(self.WORDS), split_char_encoder, keep=2 * keep_chars - 1)
        self.assertEqual(r["reach_flags"], [False, False, True])
        r2 = ptok.analyze(_entries(self.WORDS), split_char_encoder, keep=2 * keep_chars)
        self.assertEqual(r2["reach_flags"], [False, True, True])

    def test_empty(self):
        r = ptok.analyze([], char_encoder)
        self.assertEqual((r["items"], r["tokens"], r["reach"]), (0, 0, 0))

    def test_max_items_all_reach_with_char_encoder(self):
        n = ptok.max_items_all_reach(char_encoder, "S", 0)
        fits = ptok.analyze(synth.synth_entries("S", n, 0), char_encoder)
        over = ptok.analyze(synth.synth_entries("S", n + 1, 0), char_encoder)
        self.assertEqual(fits["dropped"], 0)
        self.assertGreater(over["dropped"], 0)


SECRET_WORDS = ["ZQX秘密語アルファ", "ZQX社外秘ベータ", "ZQX共有の固有名"]
SECRET_PATHS = ["secretproj-zq", "sharedproj-zq", "shared-zq-dir", "team-zq.json"]


class RealRowsPrivacyTest(unittest.TestCase):
    def _layout(self, d: Path):
        g = d / "home" / "voice_memos" / "glossary"
        g.mkdir(parents=True)
        (g / "people-zq.json").write_text(json.dumps(
            {"entries": [{"canonical": SECRET_WORDS[0]}, {"canonical": "ZQX重複語"}]}, ensure_ascii=False),
            encoding="utf-8")
        p1 = d / "projects" / "secretproj-zq" / "voice_memos" / "glossary"
        p1.mkdir(parents=True)
        (p1 / "team-zq.json").write_text(json.dumps(
            {"entries": [{"canonical": SECRET_WORDS[1]}, {"canonical": "ZQX重複語", "wrong_variants": ["x"]}]},
            ensure_ascii=False), encoding="utf-8")
        p2 = d / "projects" / "sharedproj-zq"
        (p2 / "voice_memos").mkdir(parents=True)
        (p2 / "voice_memos" / "config.json").write_text(
            json.dumps({"paths": {"extra_glossary_dirs": ["shared-zq-dir"]}}), encoding="utf-8")
        (p2 / "shared-zq-dir").mkdir()
        (p2 / "shared-zq-dir" / "shared.json").write_text(json.dumps(
            {"entries": [{"canonical": SECRET_WORDS[2]}]}, ensure_ascii=False), encoding="utf-8")
        (d / "projects" / "no-memos-zq").mkdir()
        return g, d / "projects"

    def test_numbers_only_and_no_words_or_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, projects = self._layout(Path(tmp))
            roots = ptok.find_project_roots(str(projects / "*"))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rows = ptok.real_rows(char_encoder, g, roots)
            table = ptok._table(rows, list(rows[0].keys()))
            dumped = json.dumps(rows, ensure_ascii=False)
        self.assertEqual(len(roots), 2)
        self.assertEqual([r["label"] for r in rows], ["G", "P1", "P2"])
        by = {r["label"]: r for r in rows}
        self.assertEqual(by["G"]["merged_items"], 2)
        self.assertEqual(by["P1"]["layer_items"], {"global": 2, "project": 2})
        self.assertEqual(by["P1"]["merged_items"], 3)  # 重複語は 1 つにまとまる
        self.assertEqual(by["P2"]["layer_items"], {"global": 2, "extra": 1, "project": 0})
        self.assertEqual(by["P2"]["merged_items"], 3)
        for out in (buf.getvalue(), table, dumped):
            for s in SECRET_WORDS + SECRET_PATHS + ["ZQX重複語", "people-zq"]:
                self.assertNotIn(s, out)
            self.assertNotIn("ZQX", out)


class TranscriptStatsPrivacyTest(unittest.TestCase):
    MD_DETAILS = (
        "---\ndate: 2026-01-01\nsource: zq-secret-memo.m4a\nduration: 02:00\nengine: x\nsegments: 4\n"
        "glossary_replacements: 0\nproofread: claude-sonnet-4-6\n---\n\n# ボイスメモ 2026-01-01\n\n"
        "ZQX校閲後の本文です。\n\n<details>\n<summary>校閲前（文字起こし生データ）</summary>\n\n"
        "[00:00] ZQX校閲前の一\n\n[00:40] 文目です\n\n</details>\n"
    )
    MD_BODY = (
        "---\ndate: 2026-01-02\nsource: zq-other.m4a\nduration: 01:00\nengine: x\nsegments: 2\n"
        "glossary_replacements: 0\nproofread: false\n---\n\n# ボイスメモ 2026-01-02\n\nZQX生の本文\n"
    )

    def test_parse_and_counts(self):
        p = ts.parse_transcript(self.MD_DETAILS)
        self.assertEqual((p["seconds"], p["segments"], p["proofread"], p["source"]), (120, 4, True, "details"))
        self.assertEqual(ts.count_chars(p["text"]), len("ZQX校閲前の一文目です"))
        self.assertEqual(ts.count_chars(p["text"]), ts.count_chars_bytes(p["text"]))
        q = ts.parse_transcript(self.MD_BODY)
        self.assertEqual((q["seconds"], q["source"], ts.count_chars(q["text"])), (60, "body", len("ZQX生の本文")))
        self.assertIsNone(ts.parse_transcript("no frontmatter"))

    def test_output_has_numbers_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "zq-proj" / "voice_memos" / "transcripts"
            d.mkdir(parents=True)
            (d / "2026-01-01_zq-secret-memo.md").write_text(self.MD_DETAILS, encoding="utf-8")
            (d / "2026-01-02_zq-other.md").write_text(self.MD_BODY, encoding="utf-8")
            res = ts.stats(ts.collect_files([str(Path(tmp) / "*" / "voice_memos" / "transcripts")]))
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                ts.main(["--root", str(Path(tmp) / "*" / "voice_memos" / "transcripts")])
        self.assertEqual(res["summary"]["files"], 2)
        self.assertEqual([r["label"] for r in res["rows"]], ["T1", "T2"])
        self.assertEqual(res["rows"][0]["chars_per_min"], round(len("ZQX校閲前の一文目です") / 2.0, 1))
        for out in (json.dumps(res, ensure_ascii=False), buf.getvalue()):
            self.assertNotIn("ZQX", out)
            self.assertNotIn("zq-", out)
            self.assertNotIn("本文", out)


class ProofreadTimingTest(unittest.TestCase):
    BODY = "えーと、合成の文です。" * 30 + "\n\n" + "あの、二つ目の段落です。" * 30

    def _fake_run(self, result_text, calls, raise_timeout=False):
        def fake(cmd, *a, **k):
            calls.append(k)
            if raise_timeout:
                raise subprocess.TimeoutExpired(cmd, k["timeout"])
            k["stdout"].write(json.dumps({
                "type": "result", "is_error": False, "duration_ms": 1234, "duration_api_ms": 1100,
                "num_turns": 1, "result": result_text, "usage": {"input_tokens": 10, "output_tokens": 20},
            }, ensure_ascii=False))
            return subprocess.CompletedProcess(cmd, 0, None, "")
        return fake

    def setUp(self):
        self._p = mock.patch.object(proofread, "find_claude_bin", lambda: Path("claude-stub.exe"))
        self._p.start()

    def tearDown(self):
        self._p.stop()

    def test_overrides_timeout_and_reads_cli_json(self):
        calls = []
        cleaned = self.BODY.replace("えーと、", "").replace("あの、", "")
        row, out = pt.measure_one(self.BODY, ["語"], "m", 1800.0, real_run=self._fake_run(cleaned, calls))
        self.assertEqual(calls[0]["timeout"], 1800.0)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(out, cleaned)
        self.assertEqual((row["duration_ms"], row["output_tokens"], row["input_chars"]), (1234, 20, len(self.BODY)))
        self.assertEqual(row["input_paragraphs"], 2)
        self.assertIn("--print", row["flags"])
        self.assertNotIn("result", row)  # 本文は記録に入れない

    def test_empty_result_is_flagged(self):
        row, out = pt.measure_one(self.BODY, [], "m", 10.0, real_run=self._fake_run("", []))
        self.assertTrue(row["returned_input_unchanged"])  # #3 §2 の 6a: 空の結果は元の本文が返る

    def test_timeout_and_rejected(self):
        row, _ = pt.measure_one(self.BODY, [], "m", 1.0, real_run=self._fake_run("", [], raise_timeout=True))
        self.assertEqual(row["status"], "timeout")
        row, _ = pt.measure_one(self.BODY, [], "m", 10.0, real_run=self._fake_run("短い。", []))
        self.assertEqual(row["status"], "rejected")

    def test_dry_run_does_not_call_cli(self):
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(subprocess, "run", side_effect=AssertionError("CLI を呼んだ")), \
                mock.patch.object(subprocess, "Popen", side_effect=AssertionError("プロセスを起こした")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(pt.main(["--sizes", "500", "--out-dir", d]), 0)
            self.assertEqual(list(Path(d).iterdir()), [])
        self.assertIn("dry-run", buf.getvalue())


class TimeoutWaitTest(unittest.TestCase):
    def test_descendants(self):
        procs = [(1, 0, "root.exe"), (2, 1, "child.exe"), (3, 2, "grand.exe"), (4, 9, "other.exe"), (5, 1, "c2.exe")]
        self.assertEqual(tw.descendants(procs, 1), {2: "child.exe", 3: "grand.exe", 5: "c2.exe"})
        self.assertEqual(tw.descendants(procs, 4), {})
        self.assertEqual(tw._names({2: "a.exe", 3: "a.exe", 5: "b.exe"}), {"a.exe": 2, "b.exe": 1})

    def test_claude_mode_needs_flag(self):
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("プロセスを起こした")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(tw.main(["claude", "--timeout", "5"]), 0)
        self.assertIn("dry-run", buf.getvalue())


class ImportSideEffectTest(unittest.TestCase):
    def test_import_is_light(self):
        code = (
            "import sys, subprocess, socket\n"
            "class Boom:\n"
            "    def __init__(self, *a, **k): raise AssertionError('side effect on import')\n"
            "def boom(*a, **k): raise AssertionError('side effect on import')\n"
            "subprocess.Popen = Boom; socket.socket = Boom; socket.create_connection = boom\n"
            f"sys.path.insert(0, {str(ROOT / 'tools' / 'measure')!r})\n"
            "import _kit, synth, prompt_tokens, transcript_stats, proofread_timing, timeout_wait, prompt_budget\n"
            "heavy = ['torch', 'whisperx', 'faster_whisper', 'ctranslate2', 'tokenizers', 'numpy', 'pyannote']\n"
            "print(','.join(m for m in heavy if m in sys.modules))\n"
        )
        cp = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, encoding="utf-8",
                            timeout=120)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(cp.stdout.strip(), "")


if __name__ == "__main__":
    unittest.main()
