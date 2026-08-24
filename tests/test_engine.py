"""engine.py のモデルキャッシュのテスト。whisperx をスタブに差し替えるので GPU・実モデル不要。"""
from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
