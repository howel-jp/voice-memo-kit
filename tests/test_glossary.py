"""glossary.py の純粋関数テスト（GPU・外部依存なし）。

実行: python -m unittest discover -s tests   （キットのルートで）
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import glossary as g  # noqa: E402


def _write_json(path: Path, data: dict, bom: bool = False) -> None:
    text = json.dumps(data, ensure_ascii=False)
    path.write_text(text, encoding="utf-8-sig" if bom else "utf-8")


class LoadGlossaryTest(unittest.TestCase):
    def test_missing_dir_returns_empty(self):
        self.assertEqual(g.load_glossary(None), [])
        self.assertEqual(g.load_glossary(Path("Z:/no/such/dir")), [])

    def test_loads_all_json_with_source_and_tolerates_bom(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _write_json(root / "a.json", {"entries": [{"canonical": "Alpha", "wrong_variants": ["あるふぁ"]}]})
            _write_json(root / "b.json", {"entries": [{"canonical": "Beta"}]}, bom=True)
            (root / "broken.json").write_text("{not json", encoding="utf-8")
            (root / "ignored.txt").write_text("x", encoding="utf-8")
            entries = g.load_glossary(root)
        self.assertEqual([e["canonical"] for e in entries], ["Alpha", "Beta"])
        self.assertEqual(entries[0]["source"], "a")
        self.assertEqual(entries[1]["source"], "b")


class MergeGlossariesTest(unittest.TestCase):
    def test_later_layer_wins_and_variants_union(self):
        low = [{"canonical": "X", "wrong_variants": ["x1", "x2"], "notes": "low"}]
        high = [{"canonical": "X", "wrong_variants": ["x2", "x3"], "notes": "high"}]
        merged = g.merge_glossaries(low, high)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["notes"], "high")
        self.assertEqual(merged[0]["wrong_variants"], ["x1", "x2", "x3"])

    def test_order_is_first_appearance(self):
        merged = g.merge_glossaries([{"canonical": "A"}], [{"canonical": "B"}, {"canonical": "A"}])
        self.assertEqual([e["canonical"] for e in merged], ["A", "B"])

    def test_entries_without_canonical_are_dropped(self):
        merged = g.merge_glossaries([{"wrong_variants": ["?"]}, {"canonical": "", "wrong_variants": []}])
        self.assertEqual(merged, [])


class InitialPromptTest(unittest.TestCase):
    def test_empty_when_no_canonicals(self):
        self.assertEqual(g.build_initial_prompt([]), "")

    def test_lists_canonicals_neutral_ending(self):
        p = g.build_initial_prompt([{"canonical": "A"}, {"canonical": "B"}])
        self.assertIn("A、B", p)
        self.assertTrue(p.endswith("。"))
        self.assertNotIn("してください", p)  # 命令文は hallucination 混入の原因になる


class GuardedReplaceTest(unittest.TestCase):
    def test_simple_replace_counts_all(self):
        out, n = g.guarded_str_replace("ab ab", "ab", "X", {"X"})
        self.assertEqual((out, n), ("X X", 2))

    def test_prefix_collision_guard(self):
        # 「三位一」は「三位一体の単一点」の接頭辞なので、直後が「体」なら置換しない
        canonicals = {"サンイチ", "三位一体の単一点"}
        out, n = g.guarded_str_replace("三位一体の単一点と三位一", "三位一", "サンイチ", canonicals)
        self.assertEqual(out, "三位一体の単一点とサンイチ")
        self.assertEqual(n, 1)


class ApplyToSegmentsTest(unittest.TestCase):
    def test_replaces_text_and_logs(self):
        segs = [{"start": 0.0, "end": 1.0, "text": "ざびんぐの話"}, {"start": 1.0, "end": 2.0, "text": "無関係"}]
        entries = [{"canonical": "ザッピング", "wrong_variants": ["ざびんぐ"], "source": "t"}]
        new, log = g.apply_replacements_to_segments(segs, entries)
        self.assertEqual(new[0]["text"], "ザッピングの話")
        self.assertEqual(new[1]["text"], "無関係")
        self.assertEqual(log, [{"wrong": "ざびんぐ", "canonical": "ザッピング", "count": 1, "source": "t"}])
        self.assertEqual(segs[0]["text"], "ざびんぐの話")  # 入力は変更しない


if __name__ == "__main__":
    unittest.main()
