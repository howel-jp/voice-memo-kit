"""glossary.py の純粋関数テスト（GPU・外部依存なし。トークナイザは差し替え）。

語は合成（public repo のため、利用者の辞書の語を使わない）。合成の辞書は tools/measure/synth.py で作る。

実行: python -m unittest discover -s tests   （キットのルートで）
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "measure"))
sys.path.insert(0, str(ROOT / "scripts"))

import glossary as g  # noqa: E402
import synth  # noqa: E402  (合成の辞書。標準ライブラリだけ)

LIMIT = 223


def char_encoder(text: str):
    """1 字 = 1 トークンのトークナイザの代わり。"""
    return list(range(len(text))), [(i, i + 1) for i in range(len(text))]


def split_char_encoder(text: str):
    """1 字 = 2 トークン (どちらもその字の範囲) の代わり。byte-level BPE で字が割れるとき。"""
    ids, offs = [], []
    for i in range(len(text)):
        ids += [2 * i, 2 * i + 1]
        offs += [(i, i + 1), (i, i + 1)]
    return ids, offs


def reach_050(words: list[str], tokens_per_char: int) -> list[bool]:
    """0.5.0 の文字列を Whisper が後ろ 223 トークンで切ったとき、欠けずに残る語 (plan と別の数え方)。

    1 字が tokens_per_char トークンのとき、残るのは末尾の 223 // tokens_per_char 字 (割れた字は欠ける)。
    """
    s = " " + g.PROMPT_PREFIX + "、".join(words) + "。"
    keep_from = len(s) - LIMIT // tokens_per_char if len(s) * tokens_per_char > LIMIT else 0
    out, pos = [], len(" " + g.PROMPT_PREFIX)
    for w in words:
        assert s[pos:pos + len(w)] == w
        out.append(pos >= keep_from)
        pos += len(w) + 1
    return out


def greedy_expected(words: list[str], marks: list[bool]) -> set[int]:
    """1 字 1 トークンのときの、規則 3 の選び方を足し算だけでなぞったもの (plan と別の数え方)。"""
    order = [i for i, m in enumerate(marks) if not m] + [i for i, m in enumerate(marks) if m]
    base = 1 + len(g.PROMPT_PREFIX) + 1  # " " + 前置き + 「。」
    chosen: set[int] = set()
    for i in reversed(order):
        trial = chosen | {i}
        cost = base + sum(len(words[j]) for j in trial) + (len(trial) - 1)
        if cost <= LIMIT:
            chosen = trial
    return chosen


def _marked(entries: list[dict], every: int) -> list[dict]:
    return [dict(e, priority=True) if i % every == 0 else dict(e) for i, e in enumerate(entries)]


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


class PlanInitialPromptTest(unittest.TestCase):
    """#1: initial_prompt の長さと、届ける語の優先 (C1・C2)。"""

    MIXED78 = synth.mixed_entries(78, 0)

    def test_limit_is_faster_whisper_tail(self):
        self.assertEqual(g.PROMPT_MAX_TOKENS, 223)
        self.assertEqual(g.PROMPT_MAX_TOKENS, g.WHISPER_MAX_LENGTH // 2 - 1)

    def test_fits_is_the_050_string_even_with_marks(self):
        entries = [{"canonical": "ゾルマ", "priority": True}, {"canonical": "Qantrel"}, {"canonical": ""}]
        for encode in (char_encoder, split_char_encoder):
            with self.subTest(encode=encode.__name__):
                p = g.plan_initial_prompt(entries, encode)
                self.assertEqual(p.text, g.build_initial_prompt(entries))  # 印があっても並べ替えない
                self.assertEqual(p.tokens, len(encode(" " + p.text)[0]))
                self.assertEqual((p.words, p.reached, p.missed), (2, ["ゾルマ", "Qantrel"], []))
                self.assertEqual((p.prefix_reached, p.marked, p.marked_reached, p.selected), (True, 1, 1, False))

    def test_overflow_without_marks_is_the_050_string(self):
        # C2: 設定を足さない辞書 (印なし) は 0.5.0 と同じ文字列なので、届く語は 0.5.0 と同じ
        for prof in ("S", "M", "L", "E", "mixed"):
            entries = synth.mixed_entries(78, 0) if prof == "mixed" else synth.synth_entries(prof, 78, 0)
            words = [e["canonical"] for e in entries]
            for encode, tpc in ((char_encoder, 1), (split_char_encoder, 2)):
                with self.subTest(profile=prof, encode=encode.__name__):
                    p = g.plan_initial_prompt(entries, encode)
                    self.assertEqual(p.text, g.build_initial_prompt(entries))
                    self.assertGreater(p.tokens, LIMIT)
                    flags = reach_050(words, tpc)
                    self.assertEqual(p.reached, [w for w, ok in zip(words, flags) if ok])
                    self.assertEqual(p.missed, [w for w, ok in zip(words, flags) if not ok])
                    self.assertEqual(len(p.reached) + len(p.missed), 78)
                    self.assertFalse(p.prefix_reached)
                    self.assertFalse(p.selected)
                    self.assertEqual((p.marked, p.marked_reached), (0, 0))

    def test_c2_marked_words_and_prefix_all_reach(self):
        # C2: 合成の辞書 78 項目・優先の印 20 個 (4 項目おき。前の方の印の語は 0.5.0 では落ちる)
        entries = _marked(self.MIXED78, 4)
        words = [e["canonical"] for e in entries]
        marks = [g.is_priority(e) for e in entries]
        self.assertEqual(sum(marks), 20)
        self.assertFalse(reach_050(words, 1)[0])  # 先頭の印の語は 0.5.0 では届かない
        p = g.plan_initial_prompt(entries, char_encoder)
        self.assertTrue(p.selected)
        whisper_text = " " + p.text.strip()
        self.assertEqual(p.tokens, len(whisper_text))
        self.assertLessEqual(p.tokens, LIMIT)  # Whisper は全部を使う (後ろ 223 で切っても欠けない)
        self.assertTrue(p.text.startswith(g.PROMPT_PREFIX))  # 前置きの文が入る
        self.assertTrue(p.prefix_reached)
        marked_words = [w for w, m in zip(words, marks) if m]
        for w in marked_words:
            self.assertIn(w, p.reached)
        self.assertEqual((p.marked, p.marked_reached), (20, 20))
        # 届く語は、足し算でなぞった選び方と同じ。届かない語のどれを足しても 223 を超える (詰め残しが無い)
        self.assertEqual(set(p.reached), {words[i] for i in greedy_expected(words, marks)})
        for w in p.missed:
            self.assertGreater(p.tokens + len(w) + 1, LIMIT)
        # 並び: 印の無い語 (辞書の順) → 印の語 (辞書の順)。優先の高い語が後ろ
        body = p.text[len(g.PROMPT_PREFIX):-1].split("、")
        self.assertEqual(body, [w for w in words if w in p.reached and w not in marked_words] + marked_words)
        self.assertEqual(sorted(p.reached + p.missed), sorted(words))

    def test_too_many_marks_prefers_later_marked_words(self):
        # 長い語に全部印を付けると全部は入らない。印の語の中では辞書の後ろの方が先 (入らない語は飛ばして次へ)
        entries = [dict(e, priority=True) for e in synth.synth_entries("L", 40, 0)]
        words = [e["canonical"] for e in entries]
        p = g.plan_initial_prompt(entries, char_encoder)
        self.assertTrue(p.selected)
        self.assertLess(p.marked_reached, 40)
        self.assertIn(words[-1], p.reached)
        self.assertEqual(set(p.reached), {words[i] for i in greedy_expected(words, [True] * 40)})
        self.assertLessEqual(p.tokens, LIMIT)

    def test_without_encoder_marks_go_last_and_counts_are_unknown(self):
        p = g.plan_initial_prompt(self.MIXED78, None)
        self.assertEqual(p.text, g.build_initial_prompt(self.MIXED78))
        self.assertEqual((p.tokens, p.reached, p.missed, p.prefix_reached, p.marked_reached), (None,) * 5)
        entries = _marked(self.MIXED78, 4)
        p2 = g.plan_initial_prompt(entries, None)
        body = p2.text[len(g.PROMPT_PREFIX):-1].split("、")
        marked_words = [e["canonical"] for e in entries if g.is_priority(e)]
        self.assertEqual(body[-20:], marked_words)
        self.assertEqual(sorted(body), sorted(e["canonical"] for e in entries))

    def test_priority_must_be_json_true(self):
        for value in ("true", 1, "yes", None, False):
            with self.subTest(value=value):
                self.assertFalse(g.is_priority({"canonical": "ゾルマ", "priority": value}))
        self.assertTrue(g.is_priority({"canonical": "ゾルマ", "priority": True}))
        entries = [dict(e, priority="true") for e in self.MIXED78]
        p = g.plan_initial_prompt(entries, char_encoder)
        self.assertEqual((p.marked, p.selected), (0, False))
        self.assertEqual(p.text, g.build_initial_prompt(entries))

    def test_empty(self):
        for entries in ([], [{"canonical": ""}, {"wrong_variants": ["x"]}]):
            p = g.plan_initial_prompt(entries, char_encoder)
            self.assertEqual((p.text, p.tokens, p.words, p.reached, p.missed), ("", 0, 0, [], []))
            self.assertEqual(g.plan_initial_prompt(entries, None).text, "")

    def test_count_prompt_tokens_matches_whisperx_form(self):
        self.assertEqual(g.count_prompt_tokens("  ab  ", char_encoder), len(" ab"))
        self.assertEqual(g.count_prompt_tokens("", char_encoder), 0)


class GuardedReplaceTest(unittest.TestCase):
    def test_simple_replace_counts_all(self):
        out, n = g.guarded_str_replace("ab ab", "ab", "X", {"X"})
        self.assertEqual((out, n), ("X X", 2))

    def test_prefix_collision_guard(self):
        # 合成の語。「ヌビト」は「ヌビトラ谷の測候所」の接頭辞なので、直後が「ラ」なら置換しない
        canonicals = {"Nubito", "ヌビトラ谷の測候所"}
        out, n = g.guarded_str_replace("ヌビトラ谷の測候所とヌビト", "ヌビト", "Nubito", canonicals)
        self.assertEqual(out, "ヌビトラ谷の測候所とNubito")
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
