"""ほかのスキルとの約束: キットの関数の呼び方と戻り値の形 (#3 §2 の行 7、C6。#18 で足した)。

meeting-transcript・clip-shorts (git の外のユーザースキル) は、このキットの関数を import している。約束するのは次の 11 関数:
- glossary.load_glossary / merge_glossaries / build_initial_prompt / apply_replacements_to_segments
- transcribe.load_config / resolve_paths / load_merged_glossary
- engine.get_model
- transcribe.on_complete_settings / run_on_complete / run_on_complete_result (#18。前の 2 つは 0.8.0 から使われている。
  result は 0.9.0 から)
呼び方 (引数の名前・種類) と戻り値の型を変えないことを、同じ呼び方で呼んで確かめる。
足してよいのは省略できる引数だけ (#3 §4)。

関数のほかに、使う側が頼っている形も約束に入れる (#18):
- resolve_paths の戻りの output_root・output_root_missing の鍵と、cfg["output_root"] を差し替えて呼ぶ使い方
  (置き場の根の候補の解決を、ほかの項にも使い回す)。output_root の鍵を外した cfg でも呼べること
- run_on_complete の行の頭: 成功 (exit 0) は「[E] 終わりの知らせ: exit 0」、それ以外は「  [WARN] 」で始まる
  (使う側は、成功の頭で始まらなければ失敗とみなす)。run_on_complete_result の項 (line・started・timed_out・exit_code・ok)
- 設定の誤りの ValueError の文の頭が、設定の項の名前 (output_root.transcripts・on_complete.command など) であること
  (使う側は、頭に自分の項の名前 (meeting.) を足して出す)
- PR #19 のレビュー 1 回目への対応で足したもの: load_config の戻りに、config.json のキットの知らない最上位の項が残り、
  whisper の 5 つの鍵があること。run_on_complete が md の絶対パスを最後の引数にし、作業フォルダを project_root にすること。
  started は起こした後の例外でも True であること。候補の環境変数の展開。差し替えは置き換え (既定とは重ならない) であること

あわせて、`import transcribe` (meeting-transcript がする) が、プロセスの起動・外部への接続・
重い import (torch・whisperx など) をしないことを確かめる。GPU・WhisperX・Claude CLI 無しで動く。
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import inspect
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))

import _import_guard  # noqa: E402
import engine  # noqa: E402
import glossary  # noqa: E402
import transcribe  # noqa: E402

P = inspect.Parameter
POS = P.POSITIONAL_OR_KEYWORD
KW = P.KEYWORD_ONLY
VAR = P.VAR_POSITIONAL

# (名前, 種類, 既定値があるか)。0.3.3 (88ccfed) の形
EXPECTED = {
    glossary.load_glossary: [("glossary_dir", POS, False)],
    glossary.merge_glossaries: [("layers", VAR, False)],
    glossary.build_initial_prompt: [("entries", POS, False)],
    glossary.apply_replacements_to_segments: [("segments", POS, False), ("entries", POS, False)],
    transcribe.load_config: [("project_root", POS, False)],
    transcribe.resolve_paths: [("cfg", POS, False), ("project_root", POS, False)],
    transcribe.load_merged_glossary: [("paths", POS, False), ("use_global", POS, False)],
    engine.get_model: [("model_name", KW, True), ("compute_type", KW, True), ("language", KW, True),
                       ("device", KW, True), ("initial_prompt", KW, True)],
    # 0.8.0 (fdeb3d4) の形 (#18)
    transcribe.on_complete_settings: [("cfg", POS, False)],
    transcribe.run_on_complete: [("md_path", POS, False), ("command", POS, False), ("timeout", POS, False),
                                 ("project_root", POS, False)],
    # 0.9.0 の形 (#18)
    transcribe.run_on_complete_result: [("md_path", POS, False), ("command", POS, False), ("timeout", POS, False),
                                        ("project_root", POS, False)],
}

# run_on_complete の行の頭 (#18)。使う側はこの文字のまま読むので、キットの定数ではなく文字で書く
OK_HEAD = "[E] 終わりの知らせ: exit 0"
WARN_HEAD = "  [WARN] "
PY = sys.executable


class SignatureTest(unittest.TestCase):
    def test_parameters_are_unchanged_or_only_optional_ones_added(self):
        for func, expected in EXPECTED.items():
            with self.subTest(func=f"{func.__module__}.{func.__name__}"):
                params = list(inspect.signature(func).parameters.values())
                got = [(q.name, q.kind, q.default is not P.empty) for q in params[: len(expected)]]
                self.assertEqual(got, expected)
                for extra in params[len(expected):]:  # 足したものは省略できること
                    self.assertTrue(extra.default is not P.empty or extra.kind is P.VAR_KEYWORD, extra.name)

    def test_engine_get_model_defaults(self):
        d = {q.name: q.default for q in inspect.signature(engine.get_model).parameters.values()}
        self.assertEqual((d["model_name"], d["compute_type"], d["language"], d["device"], d["initial_prompt"]),
                         ("large-v3", "int8", "ja", "cuda", ""))


class _FakeWhisperx(types.ModuleType):
    def __init__(self):
        super().__init__("whisperx")
        self.calls = []

    def load_model(self, name, device, compute_type, language, asr_options=None):
        self.calls.append((name, device, compute_type, language, asr_options))
        return object()


class CallStyleTest(unittest.TestCase):
    """meeting-transcript・clip-shorts と同じ呼び方で呼ぶ (語は合成)。"""

    def test_glossary_functions(self):
        with tempfile.TemporaryDirectory() as d:
            gdir = Path(d) / "glossary"
            gdir.mkdir()
            (gdir / "a.json").write_text(json.dumps(
                {"entries": [{"canonical": "ゾルマ", "wrong_variants": ["ぞるま"]}, {"canonical": "Qantrel"}]},
                ensure_ascii=False), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                layer = glossary.load_glossary(gdir)
                missing = glossary.load_glossary(Path(d) / "none")
                none = glossary.load_glossary(None)
        self.assertIsInstance(layer, list)
        self.assertTrue(all(isinstance(e, dict) for e in layer))
        self.assertEqual((missing, none), ([], []))

        layers = [layer, [{"canonical": "ゾルマ", "wrong_variants": ["ゾル間"]}]]
        merged = glossary.merge_glossaries(*layers) if layers else []
        self.assertIsInstance(merged, list)
        self.assertEqual(glossary.merge_glossaries(), [])

        prompt = glossary.build_initial_prompt(merged) if merged else ""
        self.assertIsInstance(prompt, str)
        self.assertIsInstance(glossary.build_initial_prompt([]), str)
        # #1: 戻り値の中身も 0.5.0 と同じ (2 つのスキルの prompt を変えない)。印 ("priority") があっても並べ替えない
        self.assertEqual(prompt, "以下は本音声で言及される固有名詞の一覧です: ゾルマ、Qantrel。")
        # 先頭の語に印を付ける (印の語を後ろへ並べ替える変異を、この test だけで拾う。PR #11 のレビュー 1 回目の推奨 2)
        marked = [{"canonical": "ゾルマ", "priority": True}, {"canonical": "Qantrel"}, {"canonical": ""},
                  {"canonical": "ヌビト", "priority": True}, {"canonical": "Zemutra"}]
        self.assertEqual(glossary.build_initial_prompt(marked),
                         "以下は本音声で言及される固有名詞の一覧です: ゾルマ、Qantrel、ヌビト、Zemutra。")
        self.assertEqual(glossary.build_initial_prompt([]), "")

        utts = [{"start": 0.0, "end": 1.0, "text": "ぞるまの件", "speaker": "合成"}]
        out = glossary.apply_replacements_to_segments(utts, merged)
        self.assertIsInstance(out, tuple)
        new_utts, log = out
        self.assertIsInstance(new_utts, list)
        self.assertIsInstance(log, list)
        self.assertEqual(new_utts[0]["text"], "ゾルマの件")
        self.assertEqual(new_utts[0]["speaker"], "合成")  # ほかのキーは残る
        self.assertEqual(utts[0]["text"], "ぞるまの件")  # 入力は書き換えない

    def test_transcribe_functions(self):
        no_global = True  # 利用者のグローバルの辞書 (~/voice_memos/glossary) は読まない
        with tempfile.TemporaryDirectory() as d:
            project_root = Path(d)
            with contextlib.redirect_stdout(io.StringIO()):
                cfg = transcribe.load_config(project_root)
                paths = transcribe.resolve_paths(cfg, project_root)
                entries = transcribe.load_merged_glossary(transcribe.resolve_paths(cfg, project_root), not no_global)
        self.assertIsInstance(cfg, dict)
        self.assertIsInstance(paths, dict)
        self.assertIsInstance(entries, list)
        for key in ("glossary", "extra_glossary_dirs", "inbox", "transcripts", "processed", "global_root",
                    "global_glossary", "global_inbox", "global_processed", "output_root", "output_root_missing"):
            self.assertIn(key, paths)
        self.assertEqual((paths["output_root"], paths["output_root_missing"]), (None, False))  # 既定は根なし

    def test_load_config_keeps_items_the_kit_does_not_know(self):
        # 使う側は、config.json に自分の項 (meeting) を書いて load_config の戻りから読み、whisper の鍵も読む
        # (PR #19 のレビュー 1 回目の推奨 2)
        own = {"output_root": {"candidates": ["."], "transcripts": "合成", "processed": "合成の録音"},
               "on_complete": {"command": [PY], "timeout_sec": 30}}
        with tempfile.TemporaryDirectory() as d:
            project_root = Path(d)
            (project_root / "voice_memos").mkdir()
            (project_root / "voice_memos" / "config.json").write_text(
                json.dumps({"meeting": own, "合成の項": [1, 2], "whisper": {"device": "cpu"}}, ensure_ascii=False),
                encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                cfg = transcribe.load_config(project_root)
                default = transcribe.load_config(project_root / "voice_memos")  # config.json の無いプロジェクト
        self.assertEqual(cfg["meeting"], own)  # キットの知らない最上位の項は、そのまま残る
        self.assertEqual(cfg["合成の項"], [1, 2])
        for w in (cfg["whisper"], default["whisper"]):
            for key in ("model", "compute_type", "language", "batch_size", "chunk_size"):
                self.assertIn(key, w)
        self.assertEqual(cfg["whisper"]["device"], "cpu")  # 書いた鍵だけ変わり、ほかは既定
        self.assertEqual({k: v for k, v in cfg["whisper"].items() if k != "device"},
                         {k: v for k, v in default["whisper"].items() if k != "device"})

    def test_engine_get_model(self):
        fake = _FakeWhisperx()
        with tempfile.TemporaryDirectory() as d:
            w = transcribe.load_config(Path(d))["whisper"]
        engine.clear_model_cache()
        try:
            with mock.patch.dict(sys.modules, {"whisperx": fake}), contextlib.redirect_stdout(io.StringIO()):
                model = engine.get_model(model_name=w["model"], compute_type=w["compute_type"],
                                         language=w["language"], device="cpu", initial_prompt="合成の予告")
                again = engine.get_model(model_name=w["model"], compute_type=w["compute_type"],
                                         language=w["language"], device="cpu", initial_prompt="合成の予告")
        finally:
            engine.clear_model_cache()
        self.assertIs(model, again)
        self.assertEqual(len(fake.calls), 1)


def _first_word(e: Exception) -> str:
    """誤りの文の頭 (最初の空白まで)。使う側は、ここに自分の項の名前を足して出す。"""
    return str(e).split(" ", 1)[0]


class OnCompleteContractTest(unittest.TestCase):
    """meeting-transcript と同じ呼び方で、on_complete の関数を呼ぶ (#18)。起こす子は、合成のスクリプトだけ。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.project = self.dir / "proj"
        self.project.mkdir()
        self.md = self.dir / "out" / "合成 の 記録.md"
        self.md.parent.mkdir()
        self.md.write_text("# 合成\n", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def script(self, name: str, body: str) -> list[str]:
        path = self.dir / name
        path.write_text(body, encoding="utf-8")
        return [PY, "-I", str(path)]

    def test_settings_from_the_on_complete_item_only(self):
        with contextlib.redirect_stdout(io.StringIO()):
            cfg = transcribe.load_config(self.project)
        self.assertIsNone(transcribe.on_complete_settings(cfg))  # 既定は起こさない
        self.assertIsNone(transcribe.on_complete_settings({}))
        self.assertIsNone(transcribe.on_complete_settings({"on_complete": {"command": []}}))
        # 使う側は、自分の項 (meeting.on_complete) だけを入れた dict を渡す
        got = transcribe.on_complete_settings({"on_complete": {"command": [PY, "合成.py"], "timeout_sec": 30}})
        self.assertEqual(got, ([PY, "合成.py"], 30.0))
        self.assertIsInstance(got, tuple)
        self.assertIsInstance(got[0], list)
        self.assertIsInstance(got[1], float)
        command, timeout = transcribe.on_complete_settings({"on_complete": {"command": [PY]}})
        self.assertEqual((command, timeout), ([PY], 120.0))  # 上限を書かなければ既定
        self.assertIsInstance(timeout, float)

    def test_settings_errors_start_with_the_item_name(self):
        for oc, head in (("node tool.mjs", "on_complete"),
                         ({"command": "node tool.mjs"}, "on_complete.command"),
                         ({"command": [PY, ""]}, "on_complete.command"),
                         ({"command": [PY], "timeout_sec": 0}, "on_complete.timeout_sec"),
                         ({"command": [PY], "timeout_sec": True}, "on_complete.timeout_sec")):
            with self.subTest(oc=oc), self.assertRaises(ValueError) as cm:
                transcribe.on_complete_settings({"on_complete": oc})
            self.assertEqual(_first_word(cm.exception), head)

    def run_both(self, command: list[str], timeout: float):
        line = transcribe.run_on_complete(self.md, command, timeout, self.project)
        result = transcribe.run_on_complete_result(self.md, command, timeout, self.project)
        self.assertIsInstance(line, str)
        self.assertNotIn("\n", line)
        self.assertEqual({f.name for f in dataclasses.fields(result)}, {"line", "started", "timed_out", "exit_code"})
        self.assertIsInstance(result.line, str)
        self.assertIsInstance(result.started, bool)
        self.assertIsInstance(result.timed_out, bool)
        self.assertIsInstance(result.ok, bool)
        # 使う側の読み方 (成功の頭で始まるときだけ通った) と、値の ok が合う
        self.assertEqual(result.ok, result.line.startswith(OK_HEAD))
        self.assertEqual(line.startswith(OK_HEAD), result.ok)
        return line, result

    def test_success(self):
        # 起こし方 (PR #19 のレビュー 1 回目の推奨 3): command の後ろに md の絶対パスを 1 つ足し、作業フォルダは project_root
        record = self.dir / "record.jsonl"
        cmd = self.script("ok.py", "import json, os, sys\n"
                                   "with open(sys.argv[1], 'a', encoding='utf-8') as f:\n"
                                   "    f.write(json.dumps({'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n"
                                   'print("受け取りました")\n') + [str(record), "合成の引数"]
        line, result = self.run_both(cmd, 30)
        self.assertEqual(line, f"{OK_HEAD}（受け取りました）")
        self.assertEqual(result.line, line)
        self.assertEqual((result.ok, result.started, result.timed_out, result.exit_code), (True, True, False, 0))
        recs = [json.loads(s) for s in record.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(recs), 2)  # run_on_complete と run_on_complete_result で 1 回ずつ
        for rec in recs:
            self.assertEqual(rec["args"], [str(record), "合成の引数", os.path.abspath(self.md)])
            self.assertEqual(Path(rec["cwd"]).resolve(), self.project.resolve())

    def test_relative_md_path_is_passed_as_absolute(self):
        record = self.dir / "args.txt"
        cmd = self.script("arg.py", "import sys\nopen(sys.argv[1], 'w', encoding='utf-8').write(sys.argv[-1])\n")
        here = os.getcwd()
        os.chdir(self.dir)
        try:  # 作業フォルダからの相対パスで渡しても
            result = transcribe.run_on_complete_result(Path("out") / self.md.name, cmd + [str(record)], 30, self.project)
        finally:
            os.chdir(here)
        self.assertTrue(result.ok, result.line)
        self.assertEqual(record.read_text(encoding="utf-8"), os.path.abspath(self.md))

    def test_started_stays_true_when_the_output_cannot_be_read(self):
        # 子は走ったのに、出力の要点を読めないとき (PR #19 のレビュー 1 回目の推奨 1)。終わりの値で返し、要点は付けない
        marker = self.dir / "ran.txt"
        cmd = self.script("mark.py", "import sys\nopen(sys.argv[1], 'w').write('ran')\nprint('合成')\n") + [str(marker)]
        with mock.patch.object(transcribe, "_read_last_line", side_effect=OSError("合成の読めない")):
            result = transcribe.run_on_complete_result(self.md, cmd, 30, self.project)
        self.assertTrue(marker.is_file())
        self.assertEqual((result.ok, result.started, result.timed_out, result.exit_code), (True, True, False, 0))
        self.assertEqual(result.line, OK_HEAD)

    def test_exception_while_waiting_is_a_started_failure(self):
        # 起こした後、終わりを待つ途中の例外: 子を止め、started は True、終わりの値は None
        class FakeProc:
            pid = 0

            def __init__(self, *a, **k):
                pass

            def wait(self, timeout=None):
                raise OSError("合成の待ちの誤り")

        with mock.patch.object(transcribe.subprocess, "Popen", FakeProc), \
                mock.patch.object(transcribe, "_stop_process_tree") as stop:
            result = transcribe.run_on_complete_result(self.md, [PY], 30, self.project)
        self.assertEqual(stop.call_count, 1)
        self.assertEqual((result.ok, result.started, result.timed_out, result.exit_code), (False, True, False, None))
        self.assertTrue(result.line.startswith(WARN_HEAD), result.line)
        self.assertIn("合成の待ちの誤り", result.line)

    def test_truth_value_is_not_the_outcome(self):
        # 成否は ok で読む。値そのものの真偽は、失敗でも真 (README に書いた。PR #19 のレビュー 1 回目の任意 1)
        result = transcribe.run_on_complete_result(self.md, [str(self.dir / "無い" / "prog.exe")], 30, self.project)
        self.assertFalse(result.ok)
        self.assertTrue(bool(result))

    def test_success_without_output(self):
        line, result = self.run_both(self.script("quiet.py", "pass\n"), 30)
        self.assertEqual(line, OK_HEAD)
        self.assertEqual((result.ok, result.exit_code), (True, 0))

    def test_failure(self):
        cmd = self.script("fail.py", 'import sys\nsys.stderr.write("合成の失敗\\n")\nsys.exit(3)\n')
        line, result = self.run_both(cmd, 30)
        self.assertTrue(line.startswith(WARN_HEAD), line)
        self.assertFalse(line.startswith(OK_HEAD))
        self.assertIn("exit 3", line)
        self.assertEqual(result.line, line)
        self.assertEqual((result.ok, result.started, result.timed_out, result.exit_code), (False, True, False, 3))

    def test_timeout(self):
        cmd = self.script("sleep.py", 'import time\nprint("眠ります", flush=True)\ntime.sleep(30)\n')
        line, result = self.run_both(cmd, 1)
        for got in (line, result.line):
            self.assertTrue(got.startswith(WARN_HEAD), got)
        self.assertEqual((result.ok, result.started, result.timed_out, result.exit_code), (False, True, True, None))

    def test_cannot_start(self):
        missing = str(self.dir / "無い" / "prog.exe")
        line, result = self.run_both([missing], 30)
        self.assertTrue(line.startswith(WARN_HEAD), line)
        self.assertEqual(result.line, line)
        self.assertEqual((result.ok, result.started, result.timed_out, result.exit_code), (False, False, False, None))

    def test_line_heads_are_the_kits(self):
        # キットの定数が、約束の文字と同じ (定数を変えたら、ここで止まる)
        self.assertEqual((transcribe.ON_COMPLETE_OK_HEAD, transcribe.ON_COMPLETE_WARN_HEAD), (OK_HEAD, WARN_HEAD))


class ResolvePathsSwapTest(unittest.TestCase):
    """cfg["output_root"] を差し替えて resolve_paths を呼ぶ (meeting-transcript が、自分の項に使い回す。#18)。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.project = self.dir / "proj"
        self.project.mkdir()
        self.place = self.dir / "置き場"
        self.place.mkdir()
        self.absent = self.dir / "無い"
        with contextlib.redirect_stdout(io.StringIO()):
            self.cfg = transcribe.load_config(self.project)
        self.base = transcribe.resolve_paths(self.cfg, self.project)
        self.out = {"candidates": [str(self.absent), str(self.place)], "transcripts": "logs", "processed": "audio"}

    def tearDown(self):
        self._tmp.cleanup()

    def test_swapped_root_is_used(self):
        before_cfg, before_out = copy.deepcopy(self.cfg), copy.deepcopy(self.out)
        got = transcribe.resolve_paths({**self.cfg, "output_root": self.out}, self.project)
        self.assertEqual(got["output_root"], self.place)  # ある最初の候補
        self.assertIs(got["output_root_missing"], False)
        self.assertEqual((got["transcripts"], got["processed"]), (self.place / "logs", self.place / "audio"))
        self.assertEqual(got["inbox"], self.base["inbox"])  # ほかの項は変わらない
        self.assertEqual((self.cfg, self.out), (before_cfg, before_out))  # 渡したものを書き換えない

    def test_no_candidate_on_this_pc(self):
        got = transcribe.resolve_paths({**self.cfg, "output_root": {**self.out, "candidates": [str(self.absent)]}},
                                       self.project)
        self.assertIsNone(got["output_root"])
        self.assertIs(got["output_root_missing"], True)
        self.assertEqual((got["transcripts"], got["processed"]), (self.base["transcripts"], self.base["processed"]))

    def test_without_the_output_root_item(self):
        # 使う側は、上の段に根がある cfg から鍵を外して、既定の置き場を得る (PR #19 のレビュー 1 回目の任意 3)
        with_root = {**self.cfg, "output_root": self.out}
        self.assertEqual(transcribe.resolve_paths(with_root, self.project)["output_root"], self.place)
        plain = transcribe.resolve_paths({k: v for k, v in with_root.items() if k != "output_root"}, self.project)
        self.assertEqual((plain["output_root"], plain["output_root_missing"]), (None, False))
        self.assertEqual((plain["transcripts"], plain["processed"]), (self.base["transcripts"], self.base["processed"]))

    def test_candidates_with_environment_variables(self):
        # 候補の解決 (環境変数の展開・展開できない候補を飛ばす) も使い回せる (PR #19 のレビュー 1 回目の推奨 3)
        forms = ["$VMK_KIT_API_PLACE", "${VMK_KIT_API_PLACE}"] + (["%VMK_KIT_API_PLACE%"] if os.name == "nt" else [])
        with mock.patch.dict(os.environ, {"VMK_KIT_API_PLACE": str(self.place)}):
            os.environ.pop("VMK_KIT_API_UNSET", None)
            for form in forms:
                with self.subTest(form=form):
                    got = transcribe.resolve_paths(
                        {**self.cfg, "output_root": {**self.out, "candidates": ["${VMK_KIT_API_UNSET}", form]}},
                        self.project)
                    self.assertEqual(got["output_root"], self.place)
                    self.assertEqual(got["transcripts"], self.place / "logs")
            got = transcribe.resolve_paths({**self.cfg, "output_root": {**self.out, "candidates": ["${VMK_KIT_API_UNSET}"]}},
                                           self.project)
            self.assertEqual((got["output_root"], got["output_root_missing"]), (None, True))

    def test_errors_start_with_the_item_name(self):
        outside = str(self.dir.resolve())
        for key, bad in (("processed", "../audio"), ("transcripts", outside), ("transcripts", "")):
            for candidates in ([str(self.place)], [str(self.absent)]):  # 根があってもなくても誤り
                with self.subTest(key=key, bad=bad, candidates=candidates), self.assertRaises(ValueError) as cm:
                    transcribe.resolve_paths({**self.cfg, "output_root": {**self.out, key: bad,
                                                                          "candidates": candidates}}, self.project)
                self.assertEqual(_first_word(cm.exception), f"output_root.{key}")

    def test_swapped_item_replaces_the_default(self):
        # 差し替えは置き換えで、既定の output_root とは重ならない。鍵が無ければ誤り (PR #19 のレビュー 1 回目の任意 2)
        for key in ("transcripts", "processed"):
            with self.subTest(key=key), self.assertRaises(ValueError) as cm:
                transcribe.resolve_paths({**self.cfg, "output_root": {k: v for k, v in self.out.items() if k != key}},
                                         self.project)
            self.assertEqual(_first_word(cm.exception), f"output_root.{key}")


class DocsTest(unittest.TestCase):
    """README の「ほかのスキルとの約束」が、約束の全部を挙げている (#18)。"""

    def test_readme_lists_the_contract(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertTrue("\n## ほかのスキルとの約束" in readme, "README に節「ほかのスキルとの約束」が無い")
        section = readme.split("\n## ほかのスキルとの約束", 1)[1].split("\n## ", 1)[0]
        for func in EXPECTED:
            name = f"{func.__module__}.{func.__name__}"
            self.assertIn(f"`{name}`", section, name)
        for word in ("output_root_missing", OK_HEAD, "[WARN]", "started", "timed_out", "exit_code", "`ok`",
                     "on_complete.command", "output_root.processed", "tests/test_kit_api.py", "省略できる引数",
                     # PR #19 のレビュー 1 回目への対応で足した約束
                     "知らない最上位の項", "`chunk_size`", "md の絶対パス", "最後の引数", "作業フォルダを `project_root`",
                     "差し替えは置き換え", "成否は `ok` で読む"):
            self.assertIn(word, section, word)


class ImportSideEffectTest(unittest.TestCase):
    # 塞ぐもの (プロセスの起動・ソケット・ctypes の DLL の読み込み) と重いモジュールの一覧は
    # tests/_import_guard.py の 1 か所 (#12。tests/test_measure.py と同じ前置きを使う)
    def test_import_transcribe_is_light(self):
        cp = _import_guard.import_in_child(ROOT / "scripts", ["transcribe", "glossary", "engine", "proofread"])
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(cp.stdout.strip(), "loaded:")


if __name__ == "__main__":
    unittest.main()
