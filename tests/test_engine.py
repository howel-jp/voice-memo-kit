"""engine.py のモデルキャッシュと、initial_prompt を数えるトークナイザのテスト。

whisperx・faster_whisper・tokenizers をスタブに差し替えるので、GPU・実モデル・HF キャッシュ不要。
"""
from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import engine  # noqa: E402


class _FakeModel:
    def __init__(self, tag):
        self.tag = tag

    def transcribe(self, audio, batch_size, language, chunk_size):
        return {"segments": [{"start": 0.0, "end": 1.0, "text": f"seg-{self.tag}"}]}


class _FakeWhisperx(types.ModuleType):
    def __init__(self):
        super().__init__("whisperx")
        self.load_calls: list[dict] = []

    def load_model(self, name, device, compute_type, language, asr_options=None):
        self.load_calls.append({"name": name, "device": device, "compute_type": compute_type,
                                "language": language, "asr_options": asr_options})
        return _FakeModel(len(self.load_calls))

    def load_audio(self, path):
        return [0.0] * 16000 * 3  # 3 秒


class ModelCacheTest(unittest.TestCase):
    def setUp(self):
        self.fake = _FakeWhisperx()
        self._orig = sys.modules.get("whisperx")
        sys.modules["whisperx"] = self.fake
        engine.clear_model_cache()

    def tearDown(self):
        engine.clear_model_cache()
        if self._orig is not None:
            sys.modules["whisperx"] = self._orig
        else:
            sys.modules.pop("whisperx", None)

    def test_same_settings_load_once(self):
        r1 = engine.transcribe_audio(Path("a.m4a"), initial_prompt="固有名詞: X。")
        r2 = engine.transcribe_audio(Path("b.m4a"), initial_prompt="固有名詞: X。")
        self.assertEqual(len(self.fake.load_calls), 1)
        self.assertEqual(self.fake.load_calls[0]["asr_options"], {"initial_prompt": "固有名詞: X。"})
        self.assertEqual(r1["segments"][0]["text"], "seg-1")
        self.assertEqual(r2["segments"][0]["text"], "seg-1")
        self.assertAlmostEqual(r1["duration"], 3.0)
        self.assertEqual(r1["engine"], "whisperx-large-v3-int8")

    def test_different_prompt_or_settings_reload(self):
        engine.transcribe_audio(Path("a.m4a"), initial_prompt="P1")
        engine.transcribe_audio(Path("b.m4a"), initial_prompt="P2")
        engine.transcribe_audio(Path("c.m4a"), initial_prompt="P2", compute_type="float16")
        engine.transcribe_audio(Path("d.m4a"), initial_prompt="P2", compute_type="float16")
        self.assertEqual(len(self.fake.load_calls), 3)

    def test_no_prompt_passes_none_asr_options(self):
        engine.transcribe_audio(Path("a.m4a"))
        self.assertIsNone(self.fake.load_calls[0]["asr_options"])


class LocalEntryNotFoundError(Exception):
    """huggingface_hub の、キャッシュに無いときの例外の代わり (名前で見分ける)。"""


class _Encoding:
    def __init__(self, text):
        self.ids = [ord(c) for c in text]
        self.offsets = [(i, i + 1) for i in range(len(text))]


class LoadPromptEncoderTest(unittest.TestCase):
    """#1: initial_prompt のトークン数を、モデルを読む前に、モデルと同じ tokenizer.json で数える。"""

    def setUp(self):
        engine._PROMPT_ENCODERS.clear()
        self.addCleanup(engine._PROMPT_ENCODERS.clear)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snapshot = Path(tmp.name) / "snapshot"
        self.snapshot.mkdir()
        (self.snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.download_calls: list[tuple] = []
        self.from_file_calls: list[str] = []
        self.encode_calls: list[tuple] = []
        self.download_error: Exception | None = None

        def download_model(name, local_files_only=False):
            self.download_calls.append((name, local_files_only))
            if self.download_error is not None:
                raise self.download_error
            return str(self.snapshot)

        test = self

        class _Tokenizer:
            def __init__(self, path):
                self.path = path

            @classmethod
            def from_file(cls, path):
                test.from_file_calls.append(path)
                return cls(path)

            def encode(self, text, add_special_tokens=True):
                test.encode_calls.append((text, add_special_tokens))
                return _Encoding(text)

        fw = types.ModuleType("faster_whisper")
        utils = types.ModuleType("faster_whisper.utils")
        utils.download_model = download_model
        fw.utils = utils
        tok = types.ModuleType("tokenizers")
        tok.Tokenizer = _Tokenizer
        patch = mock.patch.dict(sys.modules, {"faster_whisper": fw, "faster_whisper.utils": utils, "tokenizers": tok})
        patch.start()
        self.addCleanup(patch.stop)

    def test_name_is_resolved_offline_and_cached(self):
        encode, note = engine.load_prompt_encoder("large-v3")
        self.assertEqual(note, "")
        self.assertEqual(self.download_calls, [("large-v3", True)])  # local_files_only: ネットにつながない
        self.assertEqual(self.from_file_calls, [str(self.snapshot / "tokenizer.json")])
        ids, offsets = encode(" 合成")
        self.assertEqual((ids, offsets), ([ord(" "), ord("合"), ord("成")], [(0, 1), (1, 2), (2, 3)]))
        self.assertEqual(self.encode_calls, [(" 合成", False)])  # 特殊トークンを足さない
        again, _ = engine.load_prompt_encoder("large-v3")
        self.assertIs(again, encode)
        self.assertEqual(len(self.from_file_calls), 1)

    def test_directory_model_uses_its_tokenizer_json(self):
        encode, note = engine.load_prompt_encoder(str(self.snapshot))
        self.assertIsNotNone(encode)
        self.assertEqual(self.download_calls, [])
        self.assertEqual(self.from_file_calls, [str(self.snapshot / "tokenizer.json")])

    def test_not_countable_returns_reason_and_is_retried(self):
        (self.snapshot / "tokenizer.json").unlink()
        self.assertEqual(engine.load_prompt_encoder("large-v3"), (None, "モデルの tokenizer.json が無い"))
        self.download_error = LocalEntryNotFoundError("not cached")
        encode, note = engine.load_prompt_encoder("large-v3")
        self.assertIsNone(encode)
        self.assertIn("キャッシュに無い", note)
        self.download_error = RuntimeError("合成の失敗")
        self.assertEqual(engine.load_prompt_encoder("large-v3"), (None, "RuntimeError"))
        # 読めなかったものはキャッシュしない (初回のダウンロードの後に数えられるように)
        self.download_error = None
        (self.snapshot / "tokenizer.json").write_text("{}", encoding="utf-8")
        encode, note = engine.load_prompt_encoder("large-v3")
        self.assertIsNotNone(encode)
        self.assertEqual(note, "")


if __name__ == "__main__":
    unittest.main()
