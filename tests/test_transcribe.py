"""transcribe.py の設定・パス解決・整形ロジックのテスト（WhisperX / GPU 不要）。

実行: python -m unittest discover -s tests   （キットのルートで）
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

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


if __name__ == "__main__":
    unittest.main()
