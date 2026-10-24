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

    def test_main_does_not_print_argument_paths(self):
        """任意 2 (a): 引数のパス (--global-dir・--projects-glob) を出力しないこと。目印をパスの上の方に置く。"""
        with tempfile.TemporaryDirectory() as tmp:
            g, projects = self._layout(Path(tmp) / "zq-root-marker")
            buf = io.StringIO()
            with mock.patch.object(ptok._kit, "load_hf_tokenizer", lambda path=None: object()), \
                    mock.patch.object(ptok._kit, "hf_encoder", lambda tok: char_encoder), \
                    contextlib.redirect_stdout(buf):
                rc = ptok.main(["--tokenizer", str(Path(tmp) / "zq-tok.json"), "--no-verify", "real",
                                "--global-dir", str(g), "--projects-glob", str(projects / "*")])
        self.assertEqual(rc, 0)
        out = buf.getvalue()
        self.assertIn("rows: 3", out)
        for s in ["ZQX", "zq-", "zq_", "people-zq"]:
            self.assertNotIn(s, out)

    def test_broken_project_config_is_counted_not_fatal(self):
        """任意 5: 壊れた voice_memos/config.json は止まらずに parse_warnings に数える (パスは出さない)。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "brokencfg-zq"
            (root / "voice_memos" / "glossary").mkdir(parents=True)
            (root / "voice_memos" / "config.json").write_text("{broken", encoding="utf-8")
            (root / "voice_memos" / "glossary" / "x.json").write_text(
                json.dumps({"entries": [{"canonical": "ZQX壊れ設定の語"}]}, ensure_ascii=False), encoding="utf-8")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rows = ptok.real_rows(char_encoder, None, ptok.find_project_roots(str(Path(tmp) / "*")))
        self.assertEqual([(r["label"], r["merged_items"], r["parse_warnings"]) for r in rows], [("P1", 1, 1)])
        for out in (buf.getvalue(), json.dumps(rows, ensure_ascii=False)):
            self.assertNotIn("ZQX", out)
            self.assertNotIn("brokencfg", out)


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

    # 0.4.0 (#8) の形: 校閲前の見出しが「校閲前（辞書の置き換え後）」になり、置き換えがあれば「置き換え前」の節が付く
    MD_NEW_OK = (
        "---\ndate: 2026-01-03\nsource: zq-new.m4a\nduration: 02:00\nengine: x\nsegments: 2\n"
        "glossary_replacements: 1\nproofread: claude-sonnet-4-6\n---\n\n# ボイスメモ 2026-01-03\n\n"
        "ZQX校閲後\n\n\n<details>\n<summary>校閲前（辞書の置き換え後）</summary>\n\nZQX置き換え後の文\n\n</details>\n"
        "\n\n<details>\n<summary>置き換え前（Whisper の出力）</summary>\n\nZQX置換前の文\n\n</details>\n"
    )
    MD_NEW_FAILED = (
        "---\ndate: 2026-01-04\nsource: zq-new2.m4a\nduration: 01:00\nengine: x\nsegments: 1\n"
        "glossary_replacements: 1\nproofread: false\nproofread_error: timeout\n"
        "proofread_error_detail: \"900 秒で返らなかった\"\n---\n\n# ボイスメモ 2026-01-04\n\n"
        "ZQX置き換え後の文\n\n\n<details>\n<summary>置き換え前（Whisper の出力）</summary>\n\nZQX置換前の文\n\n</details>\n"
    )

    def test_new_format_counts_the_text_given_to_proofread_once(self):
        p = ts.parse_transcript(self.MD_NEW_OK)
        self.assertEqual((p["proofread"], p["source"], ts.count_chars(p["text"])),
                         (True, "details", len("ZQX置き換え後の文")))
        q = ts.parse_transcript(self.MD_NEW_FAILED)
        self.assertEqual((q["proofread"], q["source"], ts.count_chars(q["text"])),
                         (False, "body", len("ZQX置き換え後の文")))  # 置き換え前の節は数えない

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
                ts.main(["--root", str(d)])  # 任意 2 (a): 目印を含むパスを引数に渡しても出さない
        self.assertEqual(buf.getvalue().count("summary:"), 2)
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
        self.assertEqual((row["first_output_matches_input_para"], row["input_paras_not_matched"]), (0, 0))

    def test_single_call_columns_are_unchanged_and_the_call_is_listed(self):
        # 呼び出しが 1 回のときの列は今までと同じ値 (#4 の数を読み直せるように)。足すのは cli_calls と calls だけ
        calls = []
        cleaned = self.BODY.replace("えーと、", "")
        row, _ = pt.measure_one(self.BODY, ["語"], "m", 1800.0, real_run=self._fake_run(cleaned, calls))
        self.assertEqual((row["num_turns"], row["duration_ms"], row["output_tokens"]), (1, 1234, 20))
        self.assertEqual(row["cli_calls"], 1)
        self.assertEqual(row["calls"][0]["input_chars"], len(self.BODY))
        self.assertEqual((row["calls"][0]["duration_ms"], row["calls"][0]["wall_sec"]), (1234, row["wall_sec_cli"]))

    def test_chunked_proofread_records_each_call(self):
        # 3,000 字を超える本文はキットが分けて校閲する (#2)。まとまりごとの字数・時間・トークンを並べる
        body = synth.synth_memo_text(7000, 1)
        chunks = proofread.split_into_chunks(body)
        self.assertGreater(len(chunks), 1)

        def fake(cmd, *a, **k):
            msg = Path(k["stdin"].name).read_text(encoding="utf-8")
            k["stdout"].write(json.dumps({
                "type": "result", "is_error": False, "duration_ms": 1000, "num_turns": 1,
                "result": msg.split("【文字起こし本文】\n", 1)[1], "usage": {"output_tokens": 10},
            }, ensure_ascii=False))
            return subprocess.CompletedProcess(cmd, 0, None, None)

        with contextlib.redirect_stdout(io.StringIO()):
            row, out = pt.measure_one(body, ["語"], "m", 900.0, real_run=fake)
        n = len(chunks)
        self.assertEqual((row["status"], out), ("ok", body))
        self.assertEqual(row["cli_calls"], n)
        self.assertEqual([c["input_chars"] for c in row["calls"]], [len(c.text) for c in chunks])
        self.assertEqual([c["num_turns"] for c in row["calls"]], [1] * n)
        self.assertEqual((row["duration_ms"], row["output_tokens"], row["num_turns"]), (1000 * n, 10 * n, 1))
        self.assertEqual(row["input_paras_not_matched"], 0)
        self.assertNotIn(chunks[0].text[:20], json.dumps(row["calls"], ensure_ascii=False))  # 本文は記録に入れない

    def test_coverage_detects_missing_front(self):
        paras = [synth.synth_memo_text(200, s).split("\n\n")[0] for s in range(6)]
        body = "\n\n".join(paras)
        cov = pt.coverage(body, "\n\n".join(paras[3:]))
        self.assertEqual(cov["first_output_matches_input_para"], 3)
        self.assertEqual(cov["last_output_matches_input_para"], 5)
        self.assertEqual(cov["input_paras_not_matched"], 3)

    def test_cli_error_keeps_only_error_text_head(self):
        raw = json.dumps({"is_error": True, "num_turns": 4, "result": "E" * 500, "usage": {"output_tokens": 9}})
        meta = pt.cli_meta(raw)
        self.assertEqual((meta["num_turns"], meta["output_tokens"], len(meta["error_result_head"])), (4, 9, 300))
        ok = pt.cli_meta(json.dumps({"is_error": False, "result": "本文"}))
        self.assertNotIn("error_result_head", ok)  # 成功のときは result を入れない

    def test_empty_result_is_flagged(self):
        # #3 §2 の 6a: 0.3.3 までは空の結果で元の本文が返った。0.4.0 からは ProofreadEmpty で棄却される (#8)
        row, out = pt.measure_one(self.BODY, [], "m", 10.0, real_run=self._fake_run("", []))
        self.assertEqual((row["status"], row["error"], out), ("rejected", "ProofreadEmpty", ""))
        self.assertFalse(row["returned_input_unchanged"])

    def test_incomplete_result_is_rejected(self):
        # #4 の U4 run1 の形 (前の段落が無い) は、比のガードを通っても棄却される (#8)
        row, out = pt.measure_one(self.BODY, [], "m", 10.0,
                                  real_run=self._fake_run("あの、二つ目の段落です。" * 40, []))  # 比 0.69
        self.assertEqual((row["status"], row["error"], out), ("rejected", "ProofreadIncomplete", ""))

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
    def test_verified_descendants(self):
        procs = [(1, 0, "root.exe"), (2, 1, "child.exe"), (3, 2, "grand.exe"), (4, 9, "other.exe"),
                 (5, 1, "c2.exe"), (6, 2, "old.exe"), (7, 1, "before-root.exe"), (8, 1, "no-ctime.exe")]
        ct = {1: 100, 2: 150, 3: 160, 4: 10, 5: 120, 6: 120, 7: 50}.get
        got = tw.verified_descendants(procs, 1, 100, ct)
        self.assertEqual(got, {2: ("child.exe", 150), 3: ("grand.exe", 160), 5: ("c2.exe", 120)})
        # 6 は親 (2) より前に作られた = 2 の PID を前に使っていたプロセスの子。7 は root より前。8 は作成時刻が取れない
        self.assertEqual(tw.verified_descendants(procs, 1, None, ct), {})
        self.assertEqual(tw._names(got), {"child.exe": 1, "grand.exe": 1, "c2.exe": 1})
        self.assertEqual(tw._names({2: "a.exe", 3: "a.exe", 5: "b.exe"}), {"a.exe": 2, "b.exe": 1})

    def test_reused_pid_is_not_killed(self):
        """推奨 1 (PR #7 のレビュー 1 回目): 見た PID が、同じ名前の別のプロセスに再利用されていたら止めない。"""
        state = {"procs": [(1, 0, "claude.exe"), (5, 1, "conhost.exe"), (6, 1, "cmd.exe")],
                 "ct": {1: 100, 5: 110, 6: 120}}
        killed = []
        w = tw.Watch([1], snap=lambda: state["procs"], ctime_of=lambda pid: state["ct"].get(pid),
                     kill=lambda pid: killed.append(pid) or True)
        w.seen.update(w.alive_descendants())
        self.assertEqual(set(w.seen), {5, 6})
        # root が終わり、5 は同じ名前の無関係のプロセス (親も作成時刻も別) が再利用。6 は親が消えた本物の子孫
        state["procs"] = [(5, 999, "conhost.exe"), (6, 1, "cmd.exe"), (999, 0, "explorer.exe")]
        state["ct"] = {5: 300, 6: 120, 999: 5}
        left = w.alive_descendants()
        self.assertEqual(left, {6: ("cmd.exe", 120)})
        self.assertEqual(w.kill_verified(left), {6: ("cmd.exe", 120)})
        self.assertEqual(killed, [6])
        # 見張りが見た後で 6 が再利用されたら、止める直前の確かめで止めない
        state["ct"][6] = 400
        self.assertEqual(w.kill_verified({6: ("cmd.exe", 120)}), {})
        self.assertEqual(killed, [6])

    def test_reused_root_pid_children_are_not_ours(self):
        # root (1, 作成 100) が終わり、PID 1 を別のプロセス (作成 500) が使い、その子 9 (作成 600) がいる
        procs = [(1, 0, "new.exe"), (9, 1, "x.exe"), (6, 1, "ours.exe")]
        ct = {1: 500, 9: 600, 6: 120}.get
        self.assertEqual(tw.verified_descendants(procs, 1, 100, ct), {6: ("ours.exe", 120)})

    def test_claude_mode_needs_flag(self):
        with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("プロセスを起こした")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(tw.main(["claude", "--timeout", "5"]), 0)
        self.assertIn("dry-run", buf.getvalue())


class PromptBudgetPureTest(unittest.TestCase):
    def test_predicted_cap_formula(self):
        import prompt_budget as pb

        self.assertEqual(pb.predicted_cap(448, 228), 221)  # 前文 223 + 印 5
        self.assertEqual(pb.predicted_cap(448, 4), 224)    # 前文なし: M // 2 が先に効く
        self.assertEqual(pb.predicted_cap(20, 4), 10)

    def test_prompt_sources(self):
        import prompt_budget as pb

        full = pb._prompt_sources(char_encoder, "full", [0, 10])
        self.assertEqual([(lbl, p) for lbl, _t, p in full], [("items=0", -1), ("items=10", -1)])
        self.assertEqual(full[0][1], "")
        self.assertTrue(full[1][1].startswith(ptok.PREFIX))
        tail = pb._prompt_sources(char_encoder, "tail", [0, 50])
        self.assertEqual([p for _l, _t, p in tail], [0, 50])
        self.assertGreaterEqual(len(char_encoder(" " + tail[0][1])[0]), ptok.KEEP + 10)


class ImportSideEffectTest(unittest.TestCase):
    def test_import_is_light(self):
        code = (
            "import sys, os, subprocess, socket, ctypes\n"
            "class Boom:\n"
            "    def __init__(self, *a, **k): raise AssertionError('side effect on import')\n"
            "class BoomLoader:\n"
            "    def __getattr__(self, name): raise AssertionError('side effect on import')\n"
            "def boom(*a, **k): raise AssertionError('side effect on import')\n"
            "subprocess.Popen = Boom; socket.socket = Boom; socket.create_connection = boom\n"
            # 任意 2 (b): os のプロセスの起動と ctypes の DLL の読み込みも塞ぐ
            "for n in dir(os):\n"
            "    if n.startswith(('spawn', 'exec')) or n in ('system', 'startfile', 'popen'):\n"
            "        setattr(os, n, boom)\n"
            "ctypes.CDLL = Boom; ctypes.WinDLL = Boom\n"
            "ctypes.cdll = ctypes.windll = BoomLoader()\n"
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
