"""ほかのスキルが import している 7 関数の呼び方と戻り値の型 (#3 §2 の行 7、C6)。

meeting-transcript・clip-shorts (git の外のユーザースキル) は、このキットの次の 7 関数を import している:
glossary.load_glossary / merge_glossaries / build_initial_prompt / apply_replacements_to_segments、
transcribe.load_config / resolve_paths / load_merged_glossary、engine.get_model。
呼び方 (引数の名前・種類) と戻り値の型を変えないことを、同じ呼び方で呼んで確かめる。
足してよいのは省略できる引数だけ (#3 §4)。

あわせて、`import transcribe` (meeting-transcript がする) が、プロセスの起動・外部への接続・
重い import (torch・whisperx など) をしないことを確かめる。GPU・WhisperX・Claude CLI 無しで動く。
"""
from __future__ import annotations

import contextlib
import inspect
import io
import json
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
}


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
                    "global_glossary", "global_inbox", "global_processed"):
            self.assertIn(key, paths)

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


class ImportSideEffectTest(unittest.TestCase):
    # 塞ぐもの (プロセスの起動・ソケット・ctypes の DLL の読み込み) と重いモジュールの一覧は
    # tests/_import_guard.py の 1 か所 (#12。tests/test_measure.py と同じ前置きを使う)
    def test_import_transcribe_is_light(self):
        cp = _import_guard.import_in_child(ROOT / "scripts", ["transcribe", "glossary", "engine", "proofread"])
        self.assertEqual(cp.returncode, 0, cp.stderr)
        self.assertEqual(cp.stdout.strip(), "loaded:")


if __name__ == "__main__":
    unittest.main()
