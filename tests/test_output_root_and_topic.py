"""#14: 置き場の根 (output_root)・録音の時刻 (recorded_at)・話題 (topic)・一時ファイルからの書き込み・--list のテスト。

WhisperX・GPU・Claude CLI は使わない (whisperx のモジュールと subprocess.run をスタブにする)。
パスは一時ディレクトリだけ。語と文は合成 (public repo のため、利用者のパス・メモの中身を使わない)。

実行: python -m unittest discover -s tests   （キットのルートで）
"""
from __future__ import annotations

import contextlib
import datetime
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import proofread  # noqa: E402
import transcribe as t  # noqa: E402
from test_transcribe import _ProcessOneBase  # noqa: E402  (process_one をスタブで走らせる土台)

try:  # 前付けを YAML として読む道具の代わり (venv に入っていれば。無ければ JSON の確かめだけ)
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

ENV = "VMK_TEST_OUTPUT_ROOT"  # 試しだけで使う環境変数の名前
UNSET = "VMK_TEST_NEVER_SET_VARIABLE"


def _yaml_front(md: str) -> dict:
    """md の前付け (最初の --- の対の間) を YAML として読む。"""
    assert md.startswith("---\n")
    return yaml.safe_load(md[4:md.index("\n---\n", 4)])


def _write_config(project: Path, data: dict) -> None:
    (project / "voice_memos").mkdir(parents=True, exist_ok=True)
    (project / "voice_memos" / "config.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _cfg_with_root(project: Path, candidates, transcripts="voice_memos/inbox", processed="attachments/voice_memos"):
    cfg = t.load_config(project)
    cfg["global_root"] = None
    cfg["output_root"] = {"candidates": candidates, "transcripts": transcripts, "processed": processed}
    return cfg


class ResolveOutputRootTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.project = self.base / "proj"
        self.project.mkdir()
        self.root = self.base / "other_repo"
        self.root.mkdir()

    def test_defaults_keep_the_old_places(self):
        cfg = t.load_config(self.project)
        self.assertEqual(cfg["output_root"]["candidates"], [])
        self.assertFalse(cfg["topic"]["enabled"])
        p = t.resolve_paths(cfg, self.project)
        self.assertEqual(p["transcripts"], self.project / "voice_memos" / "transcripts")
        self.assertEqual(p["processed"], self.project / "voice_memos" / "processed")
        self.assertIsNone(p["output_root"])
        self.assertFalse(p["output_root_missing"])

    def test_first_existing_candidate_is_used(self):
        missing = self.base / "not_here"
        p = t.resolve_paths(_cfg_with_root(self.project, [str(missing), str(self.root)]), self.project)
        self.assertEqual(p["output_root"], self.root)
        self.assertFalse(p["output_root_missing"])
        self.assertEqual(p["transcripts"], self.root / "voice_memos" / "inbox")
        self.assertEqual(p["processed"], self.root / "attachments" / "voice_memos")
        # 音声の受け箱・辞書・グローバル層は変えない
        self.assertEqual(p["inbox"], self.project / "voice_memos" / "inbox")
        self.assertEqual(p["glossary"], self.project / "voice_memos" / "glossary")
        self.assertFalse(missing.exists())

    def test_env_var_candidate_is_expanded(self):
        other = self.base / "second"
        other.mkdir()
        cfg = _cfg_with_root(self.project, ["${" + ENV + "}", str(other)])
        with mock.patch.dict(os.environ, {ENV: str(self.root)}):
            self.assertEqual(t.resolve_paths(cfg, self.project)["output_root"], self.root)
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            # 展開できない環境変数の候補は飛ばして、次の候補へ
            self.assertEqual(t.resolve_paths(cfg, self.project)["output_root"], other)

    @unittest.skipUnless(os.name == "nt", "%VAR% は Windows の書き方")
    def test_percent_env_var_on_windows(self):
        cfg = _cfg_with_root(self.project, ["%" + ENV + "%"])
        with mock.patch.dict(os.environ, {ENV: str(self.root)}):
            self.assertEqual(t.resolve_paths(cfg, self.project)["output_root"], self.root)

    def test_none_exists_falls_back_and_never_creates_the_root(self):
        missing = self.base / "notes_repo_not_on_this_pc"
        cfg = _cfg_with_root(self.project, ["${" + UNSET + "}", str(missing), "${" + UNSET + "}/sub"])
        os.environ.pop(UNSET, None)
        p = t.resolve_paths(cfg, self.project)
        self.assertIsNone(p["output_root"])
        self.assertTrue(p["output_root_missing"])
        self.assertEqual(p["transcripts"], self.project / "voice_memos" / "transcripts")
        self.assertEqual(p["processed"], self.project / "voice_memos" / "processed")
        self.assertFalse(missing.exists())

    def test_unexpanded_var_is_skipped_even_if_a_folder_has_that_name(self):
        os.environ.pop(UNSET, None)
        literal = self.project / ("${" + UNSET + "}")  # 展開されずに残った名前のフォルダ (相対パスとして当たりうる)
        literal.mkdir()
        p = t.resolve_paths(_cfg_with_root(self.project, ["${" + UNSET + "}", "$" + UNSET]), self.project)
        self.assertIsNone(p["output_root"])
        self.assertTrue(p["output_root_missing"])

    def test_a_file_is_not_a_root(self):
        f = self.base / "a_file"
        f.write_text("x", encoding="utf-8")
        p = t.resolve_paths(_cfg_with_root(self.project, [str(f)]), self.project)
        self.assertIsNone(p["output_root"])
        self.assertTrue(p["output_root_missing"])

    def test_relative_candidate_is_from_the_project_root(self):
        (self.project / "inner").mkdir()
        p = t.resolve_paths(_cfg_with_root(self.project, ["inner"]), self.project)
        self.assertEqual(p["output_root"], self.project / "inner")

    def test_one_string_is_one_candidate(self):
        p = t.resolve_paths(_cfg_with_root(self.project, str(self.root)), self.project)
        self.assertEqual(p["output_root"], self.root)

    def test_empty_env_var_candidate_is_skipped(self):
        other = self.base / "second"
        other.mkdir()
        trap = self.base / "trap"  # 空の変数を展開すると、この在るフォルダを指してしまう候補
        trap.mkdir()
        for value in ("", "   "):
            for raw in ("${" + ENV + "}", "${" + ENV + "}/sub", "$" + ENV + "/sub", "${" + ENV + "}" + str(trap),
                        "%" + ENV + "%" + str(trap)):
                with self.subTest(value=value, raw=raw), mock.patch.dict(os.environ, {ENV: value}):
                    # 空の変数を展開すると /sub (ドライブの根の下) になるので、候補ごと飛ばす
                    p = t.resolve_paths(_cfg_with_root(self.project, [raw, str(other)]), self.project)
                    self.assertEqual(p["output_root"], other)

    def test_subpaths_must_stay_under_the_root(self):
        bad = ["../outside", "voice_memos/../../outside", "", "   ", None, str(self.base / "abs"), "/abs"]
        if os.name == "nt":  # Windows でない所では、\abs は相対の 1 つの名前
            bad += ["\\abs", "C:rel", "a\\..\\..\\x"]
        for key in ("transcripts", "processed"):
            for value in bad:
                for candidates in ([str(self.root)], [str(self.base / "missing")]):  # 根が無い PC でも同じに断る
                    with self.subTest(key=key, value=value, root_exists=candidates == [str(self.root)]):
                        kw = {"transcripts": "ok", "processed": "ok", key: value}
                        with self.assertRaises(ValueError):
                            t.resolve_paths(_cfg_with_root(self.project, candidates, **kw), self.project)


class ProcessOneOutputRootTest(_ProcessOneBase):
    def test_md_and_audio_go_under_the_root(self):
        root = self.root / "other_repo"
        root.mkdir()
        cfg = _cfg_with_root(self.root, [str(root)])
        p = t.resolve_paths(cfg, self.root)
        self._cli(lambda b: b)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(sys.modules, {"whisperx": _fake(self.SEGS)}):
            out = t.process_one(self.audio, cfg=cfg, processed_dir=p["processed"], out_dir=p["transcripts"],
                                entries=self.ENTRIES, with_timestamps=False, keep=False, device="cpu",
                                use_proofread=True)
        self.assertEqual(out.parent, root / "voice_memos" / "inbox")
        self.assertTrue((root / "attachments" / "voice_memos" / "memo.m4a").exists())
        self.assertFalse(self.audio.exists())


def _fake(segs):
    from test_transcribe import _FakeWhisperx
    return _FakeWhisperx(segs)


class RecordedAtTest(_ProcessOneBase):
    def test_recorded_at_is_the_audio_mtime_with_offset(self):
        stamp = 1_700_000_000  # 合成の時刻
        os.utime(self.audio, (stamp, stamp))
        self._cli(lambda b: b)
        r = self._run()
        self.assertEqual(r["keys"], self.KEYS)
        at = datetime.datetime.fromisoformat(r["fm"]["recorded_at"])
        self.assertIsNotNone(at.tzinfo)
        self.assertEqual(int(at.timestamp()), stamp)
        self.assertEqual(r["fm"]["date"], at.strftime("%Y-%m-%d"))  # 録音日と同じ出どころ・同じ日
        self.assertRegex(r["fm"]["recorded_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")

    def test_source_is_a_json_string_for_names_yaml_would_misread(self):
        # YAML の引用符なしでは null になる・切れる・前付けの全体が読めなくなる名前 (どれも合成。Windows で使える字だけ)
        names = ["#1 memo.m4a", "a #b.m4a", "[x] memo.m4a", "&x.m4a", "!x.m4a", "%x.m4a", "@x.m4a",
                 "'x.m4a", "{x}.m4a", "- x.m4a", "合成 メモ.m4a"]
        self._cli(lambda b: b)
        for name in names:
            with self.subTest(name=name):
                self.audio = self.root / name
                self.audio.write_bytes(b"")
                r = self._run()
                self.assertEqual(json.loads(r["fm"]["source"]), name)
                if yaml is not None:
                    self.assertEqual(_yaml_front(r["md"])["source"], name)


class AtomicWriteTest(_ProcessOneBase):
    def test_no_temporary_file_is_left_even_with_a_stale_one(self):
        stamp = 1_700_000_000
        os.utime(self.audio, (stamp, stamp))
        date = datetime.datetime.fromtimestamp(stamp).strftime("%Y-%m-%d")
        out_dir = self.root / "out"
        out_dir.mkdir()
        stale = out_dir / f".{date}_memo.md.tmp"  # 前の回に残った一時ファイル (同じ名前)
        stale.write_text("old", encoding="utf-8")
        self._cli(lambda b: b)
        r = self._run()
        self.assertEqual(sorted(p.name for p in out_dir.iterdir()), [f"{date}_memo.md"])
        self.assertEqual(r["keys"], self.KEYS)

    def test_failed_replace_removes_the_temporary_file(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / "x.md"
            with mock.patch.object(t.os, "replace", side_effect=PermissionError("合成の失敗")):
                with self.assertRaises(PermissionError):
                    t.write_atomic(target, "合成\n")
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_failed_write_removes_the_partial_temporary_file(self):
        real_write_text = Path.write_text

        def partial(self_path, text, *a, **k):
            real_write_text(self_path, text[:3], *a, **k)  # 途中まで書いてから、容量切れの代わり
            raise OSError(28, "合成の容量切れ")

        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(Path, "write_text", partial):
                with self.assertRaises(OSError):
                    t.write_atomic(Path(d) / "x.md", "合成の本文\n")
            self.assertEqual(list(Path(d).iterdir()), [])

    def test_bytes_are_the_same_as_write_text(self):
        with tempfile.TemporaryDirectory() as d:
            text = "---\ndate: 2000-01-01\n---\n\n# 合成\n\n一行目\n二行目\n"
            a, b = Path(d) / "a.md", Path(d) / "b.md"
            a.write_text(text, encoding="utf-8")
            t.write_atomic(b, text)
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["a.md", "b.md"])


class TopicTest(_ProcessOneBase):
    """話題 (D)。校閲 (C) と話題の CLI 呼び出しを、システムプロンプトで分けるスタブで走らせる。"""

    def setUp(self):
        super().setUp()
        self.cfg["topic"]["enabled"] = True
        self.calls: list[dict] = []

    def _stub(self, *, proofread_reply=lambda b: b.replace("えーと、", ""), topic_reply="合成の段取り",
              topic_timeout=False, topic_returncode=0):
        topic_prompt = proofread.load_topic_prompt()

        def fake(cmd, *a, **k):
            message = k["stdin"].read()
            system = cmd[cmd.index("--append-system-prompt") + 1]
            body = message.split("【文字起こし本文】\n", 1)[1]
            kind = "topic" if system == topic_prompt else "proofread"
            self.calls.append({"kind": kind, "body": body, "timeout": k["timeout"], "message": message,
                               "model": cmd[cmd.index("--model") + 1]})
            if kind == "topic" and topic_timeout:
                raise subprocess.TimeoutExpired(cmd, k["timeout"])
            reply = proofread_reply(body) if kind == "proofread" else topic_reply
            k["stdout"].write(json.dumps({"type": "result", "is_error": False, "result": reply, "num_turns": 1},
                                         ensure_ascii=False))
            return subprocess.CompletedProcess(cmd, topic_returncode if kind == "topic" else 0, None, None)

        patch = mock.patch.object(proofread.subprocess, "run", fake)
        patch.start()
        self.addCleanup(patch.stop)

    def _run_topic(self, **kw) -> dict:
        engine_cache = sys.modules.get("engine")
        if engine_cache is not None:
            engine_cache.clear_model_cache()
        buf = io.StringIO()
        with mock.patch.dict(sys.modules, {"whisperx": _fake(kw.pop("segs", self.SEGS))}), contextlib.redirect_stdout(buf):
            out = t.process_one(self.audio, cfg=self.cfg, processed_dir=self.root / "processed",
                                out_dir=self.root / "out", entries=self.ENTRIES, with_timestamps=False, keep=True,
                                device="cpu", use_proofread=kw.pop("use_proofread", True),
                                use_topic=kw.pop("use_topic", True))
        self.stdout = buf.getvalue()
        return self._parse(out.read_text(encoding="utf-8"))

    def test_topic_is_written_from_the_proofread_body(self):
        self._stub(topic_reply="話題：「合成の段取りの相談」\n\n補足の行")
        r = self._run_topic()
        self.assertEqual(r["keys"], self.KEYS + ["topic"])
        self.assertEqual(json.loads(r["fm"]["topic"]), "合成の段取りの相談")
        self.assertEqual([c["kind"] for c in self.calls], ["proofread", "topic"])
        topic_call = self.calls[1]
        self.assertEqual(topic_call["body"], self.REPLACED.replace("えーと、", ""))  # 校閲後の本文を渡す
        self.assertEqual(topic_call["timeout"], proofread.TOPIC_TIMEOUT_SEC)
        self.assertIn("30 字まで", topic_call["message"])
        self.assertIn("[D] 話題: 合成の段取りの相談", self.stdout)

    def test_quotes_backslash_colon_hash_and_dashes_survive_the_frontmatter(self):
        # PR #15 のレビュー 1 回目の推奨 1: 前付けを読む道具が、元の文字列に戻せること (json.dumps を外すと落ちる)
        want = '合成の "件" \\ と: #x ---'
        self._stub(topic_reply='話題：「合成の "件" \\ と: #x ---」\n二行目')
        r = self._run_topic()
        self.assertEqual(r["keys"], self.KEYS + ["topic"])
        self.assertEqual(json.loads(r["fm"]["topic"]), want)
        self.assertEqual(r["md"].count("\n---\n"), 1)  # 前付けの終わりは 1 つだけ (話題の --- で切れない)
        if yaml is not None:
            front = _yaml_front(r["md"])
            self.assertEqual(front["topic"], want)
            self.assertEqual(front["source"], "memo.m4a")

    def test_unprintable_characters_are_dropped_and_yaml_can_read_it(self):
        # PR #15 のレビュー 1 回目の任意 2: DEL・C1 の制御文字は YAML が読めない。幅の無い字・向きの字も残さない
        self._stub(topic_reply="合成\x7fの\u200b段\u202e取り\x9f\ufeff\u2028と\x85相談")
        r = self._run_topic()
        self.assertEqual(json.loads(r["fm"]["topic"]), "合成の段取り と 相談")
        if yaml is not None:
            self.assertEqual(_yaml_front(r["md"])["topic"], "合成の段取り と 相談")

    def test_without_proofread_the_transcript_is_used(self):
        self._stub()
        r = self._run_topic(use_proofread=False)
        self.assertEqual([c["kind"] for c in self.calls], ["topic"])
        self.assertEqual(self.calls[0]["body"], self.REPLACED)
        self.assertEqual(r["fm"]["proofread"], "false")
        self.assertEqual(json.loads(r["fm"]["topic"]), "合成の段取り")

    def test_max_chars_from_config(self):
        self.cfg["topic"]["max_chars"] = 5
        self._stub(topic_reply="あいうえおかきくけこ")
        r = self._run_topic()
        self.assertEqual(json.loads(r["fm"]["topic"]), "あいうえお")
        self.assertIn("5 字まで", self.calls[-1]["message"])

    def test_topic_model_falls_back_to_the_proofread_model(self):
        self._stub()
        self._run_topic()
        self.assertIsNone(self.cfg["topic"]["model"])  # 既定は null
        self.assertEqual(self.calls[-1]["model"], self.model)
        self.assertIn(f"[D] 話題を付けています（{self.model}）", self.stdout)

    def test_topic_model_from_config(self):
        self.cfg["topic"]["model"] = "synthetic-topic-model"
        self._stub()
        self._run_topic()
        self.assertEqual([(c["kind"], c["model"]) for c in self.calls],
                         [("proofread", self.model), ("topic", "synthetic-topic-model")])

    def test_timeout_keeps_the_memo_and_records_the_reason(self):
        self._stub(topic_timeout=True)
        r = self._run_topic()
        self.assertEqual(r["keys"], self.KEYS + ["topic_error", "topic_error_detail"])
        self.assertEqual(r["fm"]["topic_error"], "timeout")
        self.assertEqual(r["fm"]["proofread"], self.model)  # 校閲は残る
        self.assertEqual(r["body"], self.REPLACED.replace("えーと、", ""))
        self.assertIn("[WARN] 話題を付けられません（timeout", self.stdout)

    def test_cli_error_and_empty_reply(self):
        for kw, code in (({"topic_returncode": 1}, "cli_error"), ({"topic_reply": "「」"}, "empty"),
                         ({"topic_reply": "  \n  "}, "empty")):
            with self.subTest(code=code, kw=kw):
                self.calls.clear()
                self._stub(**kw)
                r = self._run_topic()
                self.assertNotIn("topic", r["fm"])
                self.assertEqual(r["fm"]["topic_error"], code)

    def test_disabled_does_not_call_the_cli_for_a_topic(self):
        self._stub()
        r = self._run_topic(use_topic=False)
        self.assertEqual([c["kind"] for c in self.calls], ["proofread"])
        self.assertEqual(r["keys"], self.KEYS)

    def test_empty_transcript_has_no_topic_and_no_call(self):
        self._stub()
        r = self._run_topic(segs=[{"start": 0.0, "end": 1.0, "text": "   "}])
        self.assertEqual(self.calls, [])
        self.assertEqual(r["keys"], self.KEYS)

    def test_long_text_is_cut_before_sending(self):
        self._stub()
        long_text = "合成の文。" * (proofread.TOPIC_INPUT_MAX_CHARS // 5 + 100)
        proofread.make_topic(long_text, model="m")
        self.assertEqual(len(self.calls[-1]["body"]), proofread.TOPIC_INPUT_MAX_CHARS)

    def test_make_topic_on_empty_text_does_not_call(self):
        self._stub()
        with self.assertRaises(proofread.TopicEmpty):
            proofread.make_topic("  \n ", model="m")
        self.assertEqual(self.calls, [])


class CleanTopicTest(unittest.TestCase):
    def test_shapes(self):
        cases = [
            ("合成の段取り", "合成の段取り"),
            ("話題：合成の段取り", "合成の段取り"),
            ("Topic: synthetic plan", "synthetic plan"),
            ("「合成の段取り」", "合成の段取り"),
            ("**合成の段取り**", "合成の段取り"),
            ("# 合成の段取り", "合成の段取り"),
            ("- 「合成の段取り」", "合成の段取り"),
            ("「合成」と「段取り」", "「合成」と「段取り」"),  # 中にも括弧があるときは外さない
            ("\n\n合成の段取り\n二行目", "合成の段取り"),
            ("```\n合成の段取り\n```", "合成の段取り"),
            ("合成の　　段取り  と\t相談", "合成の 段取り と 相談"),
            ("", ""),
            ("「」", ""),  # 中が空の対も外す (空として make_topic が TopicEmpty にする)
            ('""', ""),
            # PR #15 のレビュー 1 回目の任意 1: ラベルの語だけの行は飛ばす。太字のラベルも外す
            ("## 話題\n合成の段取り", "合成の段取り"),
            ("話題：\n\n「合成の段取り」", "合成の段取り"),
            ("**話題**\n合成の段取り", "合成の段取り"),
            ("**話題:** 合成の段取り", "合成の段取り"),
            ("**話題**：合成の段取り", "合成の段取り"),
            ("話題の整理", "話題の整理"),  # ラベルの語で始まるだけの話題は残す
            ("話題", ""),  # ラベルの語だけ (次の行が無い) は空
            # PR #15 のレビュー 2 回目の推奨 1: 値が太字のとき、後ろに ** を残さない (1 回目の head と同じ形)
            ("話題: **合成の段取り**", "合成の段取り"),
            ("話題：**合成の段取り**", "合成の段取り"),
            ("**話題：合成の段取り**", "合成の段取り"),  # 行の全体が太字
            ("Topic: **synthetic plan**", "synthetic plan"),
            ("話題: __合成の段取り__", "合成の段取り"),  # __ の対も外す (1 回目の head では残っていた)
            ("**話題**：**合成の段取り**", "合成の段取り"),
            ("__話題__: 合成の段取り", "合成の段取り"),
            ("__init__ の整理", "__init__ の整理"),  # 全体を囲まない __ は残す
            ("titles の整理", "titles の整理"),
            # 任意 2: 印字できない字は落とす。改行の類・全角の空白は空白にそろえる
            ("合成\x7fの\u200b段取り\x9f", "合成の段取り"),
            ("合成\u202eの\ufeff段取り", "合成の段取り"),
            ("合成\u2028の\x85段取り　相談", "合成 の 段取り 相談"),
        ]
        for raw, want in cases:
            with self.subTest(raw=raw):
                self.assertEqual(proofread.clean_topic(raw), want)

    def test_length_is_cut_by_code_points(self):
        self.assertEqual(proofread.clean_topic("あ" * 40), "あ" * proofread.TOPIC_MAX_CHARS)
        self.assertEqual(proofread.clean_topic("あいう えお", 4), "あいう")  # 切った末の空白は落とす

    def test_describe_failure_reads_topic_empty(self):
        self.assertEqual(proofread.describe_failure(proofread.TopicEmpty("x", detail="空の結果")), ("empty", "空の結果"))


class MainListTest(unittest.TestCase):
    """--list: 対象と置き場を出すだけ。文字起こし・移動・フォルダの作成をしない。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.project = self.base / "proj"
        (self.project / "voice_memos" / "inbox").mkdir(parents=True)
        self.audio = self.project / "voice_memos" / "inbox" / "a.m4a"
        self.audio.write_bytes(b"")

    def _main(self, *extra) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        argv = ["transcribe.py", "--project", str(self.project), "--no-global", *extra]
        with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = t.main()
        return code, out.getvalue(), err.getvalue()

    def test_missing_root_warns_and_creates_nothing(self):
        missing = self.base / "notes_repo_not_on_this_pc"
        _write_config(self.project, {"output_root": {"candidates": [str(missing)], "transcripts": "voice_memos/inbox"}})
        code, out, _ = self._main("--list")
        self.assertEqual(code, 0)
        self.assertIn(t.OUTPUT_ROOT_MISSING_WARN, out)
        self.assertIn(f"transcripts={self.project.resolve() / 'voice_memos' / 'transcripts'}\n", out)
        self.assertNotIn("output_root=", out)
        self.assertIn("対象 1 件", out)
        self.assertFalse(missing.exists())
        self.assertFalse((self.project / "voice_memos" / "transcripts").exists())
        self.assertFalse((self.project / "voice_memos" / "processed").exists())
        self.assertTrue(self.audio.exists())

    def test_missing_root_with_out_dir_warns_only_about_the_audio(self):
        # PR #15 のレビュー 2 回目の任意 1: md は --out-dir に出るので、既定に戻るのは音声の退避先だけ
        missing = self.base / "notes_repo_not_on_this_pc"
        out_dir = self.base / "elsewhere"
        _write_config(self.project, {"output_root": {"candidates": [str(missing)]}})
        code, out, _ = self._main("--list", "--out-dir", str(out_dir))
        self.assertEqual(code, 0)
        self.assertIn(t.OUTPUT_ROOT_MISSING_WARN_OUT_DIR, out.splitlines())
        self.assertNotIn(t.OUTPUT_ROOT_MISSING_WARN, out.splitlines())
        self.assertIn(f"transcripts={out_dir}", out.splitlines())
        self.assertFalse(out_dir.exists())  # --list は作らない
        self.assertFalse(missing.exists())

    def test_existing_root_is_shown(self):
        root = self.base / "other_repo"
        root.mkdir()
        _write_config(self.project, {"output_root": {"candidates": [str(root)], "transcripts": "voice_memos/inbox",
                                                     "processed": "attachments/voice_memos"}})
        code, out, _ = self._main("--list")
        self.assertEqual(code, 0)
        # PR #15 のレビュー 1 回目の推奨 2: = の後ろはパスだけ。根は別の行
        lines = out.splitlines()
        self.assertIn(f"transcripts={root / 'voice_memos' / 'inbox'}", lines)
        self.assertIn(f"processed  ={root / 'attachments' / 'voice_memos'}", lines)
        self.assertIn(f"output_root={root}", lines)
        self.assertLess(lines.index(f"processed  ={root / 'attachments' / 'voice_memos'}"), lines.index(f"output_root={root}"))
        self.assertNotIn("[WARN]", out)
        self.assertEqual(sorted(p.name for p in root.iterdir()), [])  # 作らない

    def test_no_audio_lists_zero(self):
        self.audio.unlink()
        code, out, _ = self._main("--list")
        self.assertEqual(code, 0)
        self.assertIn("対象 0 件", out)

    def test_wrong_subpath_is_an_error(self):
        _write_config(self.project, {"output_root": {"candidates": [str(self.base)], "transcripts": "../x"}})
        code, _, err = self._main("--list")
        self.assertEqual(code, 1)
        self.assertIn("ERROR: 設定の誤り", err)


class DocsTest(unittest.TestCase):
    """SKILL.md・README が、置き場を決め打ちで見ず、新しい欄と [WARN] を同じ文で書いている。"""

    def setUp(self):
        self.skill = (Path(t.SKILL_DIR) / "SKILL.md").read_text(encoding="utf-8")
        self.readme = (Path(t.SKILL_DIR) / "README.md").read_text(encoding="utf-8")

    def test_warn_text_is_the_same_in_code_and_docs(self):
        text = t.OUTPUT_ROOT_MISSING_WARN.strip()
        for name, doc in (("SKILL.md", self.skill), ("README.md", self.readme)):
            with self.subTest(doc=name):
                self.assertIn(text, doc)
        # --out-dir のときの後半の文は README に書く (SKILL.md は --out-dir を使わない)
        tail = t.OUTPUT_ROOT_MISSING_WARN_OUT_DIR.split("ありません。", 1)[1]
        self.assertIn(tail, self.readme)

    def test_skill_uses_the_launcher_for_places(self):
        self.assertIn("--list", self.skill)
        self.assertIn('--list "<音声パス>"', self.skill)  # 引数ありの形 (PR #15 のレビュー 1 回目の任意 5)
        self.assertIn("transcripts=", self.skill)
        self.assertIn("output_root=", self.skill)  # 根は別の行 (推奨 2)
        self.assertIn("output_root=", self.readme)
        self.assertNotIn("Get-ChildItem voice_memos\\inbox", self.skill)
        for key in ("recorded_at", "topic", "topic_error"):
            with self.subTest(key=key):
                self.assertIn(f"`{key}", self.skill)

    def test_readme_documents_output_root_and_topic(self):
        for word in ("output_root", "candidates", "recorded_at", '"topic"', "--no-topic", "--list"):
            with self.subTest(word=word):
                self.assertIn(word, self.readme)

    def test_topic_prompt_exists_and_says_one_line(self):
        self.assertIn("1 行", proofread.load_topic_prompt())


if __name__ == "__main__":
    unittest.main()
