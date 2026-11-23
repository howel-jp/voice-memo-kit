"""#16: 終わりの知らせ (on_complete) のテスト。

md を書いて音声を退かした後に、設定したコマンドを、md の絶対パスを最後の引数にして、シェルを通さずに起こす。
起こす子は、このテストの python (sys.executable) で走る合成のスクリプトだけ。パスは一時ディレクトリだけ。
WhisperX・GPU・Claude CLI は使わない (whisperx のモジュールをスタブにし、校閲は --no-proofread で止める)。
語と文は合成 (public repo のため、利用者のパス・メモの中身を使わない)。

実行: python -m unittest discover -s tests   （キットのルートで）
"""
from __future__ import annotations

import contextlib
import io
import json
import locale
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 単体で走らせるとき (python -m unittest tests.test_on_complete)

import engine  # noqa: E402
import transcribe as t  # noqa: E402
from test_transcribe import _FakeWhisperx, char_encoder  # noqa: E402

PY = sys.executable
SEGS = [{"start": 0.0, "end": 2.0, "text": "合成の文です。"}, {"start": 2.2, "end": 4.0, "text": "次の文です。"}]

# 起こされたときの様子を 1 行の JSON で記録する子のスクリプト。引数: <記録のファイル> <受け箱> <md>
HOOK = r'''
import json, os, sys
from pathlib import Path
log, inbox, md = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[-1])
rec = {
    "args": sys.argv[1:],
    "cwd": os.getcwd(),
    "md_exists": md.is_file(),
    "md_is_absolute": md.is_absolute(),
    "hidden_left": sorted(p.name for p in md.parent.iterdir() if p.name.startswith(".")),
    "inbox": sorted(p.name for p in inbox.iterdir()) if inbox.is_dir() else None,
    "first_line": md.read_text(encoding="utf-8").splitlines()[0] if md.is_file() else None,
}
with log.open("a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print("1 行目")
print()
print("受け取った " + md.name)
'''

# 孫を起こしてから眠る子。孫は 2.5 秒後に印のファイルを書く (止められていれば書かない)。引数: <印> <md>
SLEEPER = r'''
import subprocess, sys, time
subprocess.Popen([sys.executable, "-c",
                  "import sys, time; time.sleep(2.5); open(sys.argv[1], 'w').write('alive')", sys.argv[1]])
print("眠ります", flush=True)
time.sleep(30)
'''


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class SettingsTest(unittest.TestCase):
    def test_default_config_runs_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = t.load_config(Path(d))
        self.assertEqual(cfg["on_complete"], {"command": [], "timeout_sec": t.ON_COMPLETE_DEFAULT_TIMEOUT_SEC})
        self.assertIsNone(t.on_complete_settings(cfg))

    def test_missing_or_empty_runs_nothing(self):
        for oc in (None, {}, {"command": []}, {"command": None}, {"command": [], "timeout_sec": "誤りでも見ない"}):
            with self.subTest(oc=oc):
                cfg = {} if oc is None else {"on_complete": oc}
                self.assertIsNone(t.on_complete_settings(cfg))

    def test_command_and_timeout(self):
        self.assertEqual(t.on_complete_settings({"on_complete": {"command": ["node", "x.mjs"]}}),
                         (["node", "x.mjs"], 120.0))
        self.assertEqual(t.on_complete_settings({"on_complete": {"command": ["node"], "timeout_sec": 0.5}}),
                         (["node"], 0.5))
        self.assertEqual(t.on_complete_settings({"on_complete": {"command": ["node"], "timeout_sec": 60}}),
                         (["node"], 60.0))

    def test_wrong_shapes_are_errors(self):
        wrong = [
            ["node"],  # on_complete が dict でない
            {"command": "node x.mjs"},  # 1 つの文字列 (シェルを通さない)
            {"command": ["node", 1]},
            {"command": [""]},
            {"command": ["node", "  "]},
            {"command": ["node"], "timeout_sec": 0},
            {"command": ["node"], "timeout_sec": -1},
            {"command": ["node"], "timeout_sec": "60"},
            {"command": ["node"], "timeout_sec": True},
            {"command": ["node"], "timeout_sec": None},
            {"command": ["node"], "timeout_sec": math.inf},
            {"command": ["node"], "timeout_sec": math.nan},
        ]
        for oc in wrong:
            with self.subTest(oc=oc):
                with self.assertRaises(ValueError) as cm:
                    t.on_complete_settings({"on_complete": oc})
                self.assertIn("on_complete", str(cm.exception))

    def test_json_infinity_is_an_error(self):
        # Python の json は Infinity・NaN を読むので、設定のファイルから来うる
        cfg = json.loads('{"on_complete": {"command": ["node"], "timeout_sec": Infinity}}')
        with self.assertRaises(ValueError):
            t.on_complete_settings(cfg)

    def test_project_config_overrides_like_other_settings(self):
        # 重ね方は今のまま: キットの既定 → プロジェクトの config.json。command は丸ごと置き換え、timeout_sec は既定が残る
        with tempfile.TemporaryDirectory() as d:
            project = Path(d)
            _write(project / "voice_memos" / "config.json", json.dumps({"on_complete": {"command": ["node", "a.mjs"]}}))
            cfg = t.load_config(project)
        self.assertEqual(t.on_complete_settings(cfg), (["node", "a.mjs"], 120.0))


class ArgvTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.md = self.root / "out" / "2026-01-02_合成 & メモ.md"

    def test_md_path_is_the_last_absolute_argument(self):
        argv = t.on_complete_argv(["node", "tools/x.mjs", "--flag"], self.md, self.root)
        self.assertEqual(argv, ["node", "tools/x.mjs", "--flag", os.path.abspath(self.md)])

    def test_relative_md_path_becomes_absolute(self):
        argv = t.on_complete_argv(["node"], Path("rel") / "a.md", self.root)
        self.assertTrue(os.path.isabs(argv[-1]))
        self.assertEqual(argv[-1], os.path.abspath(Path("rel") / "a.md"))

    def test_relative_program_with_a_separator_is_from_the_project_root(self):
        for prog in ("tools/hook.exe", "tools\\hook.exe", "./hook.exe"):
            with self.subTest(prog=prog):
                self.assertEqual(t.on_complete_argv([prog], self.md, self.root)[0], str(self.root / prog))

    def test_bare_name_and_absolute_program_are_kept(self):
        self.assertEqual(t.on_complete_argv(["node"], self.md, self.root)[0], "node")
        self.assertEqual(t.on_complete_argv([PY], self.md, self.root)[0], PY)


class RunOnCompleteTest(unittest.TestCase):
    """本物の子プロセス (このテストの python) で起こす。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.md = _write(self.root / "out" / "2026-01-02_memo.md", "---\n---\n")

    def run_code(self, code: str, timeout: float = 30) -> str:
        script = _write(self.root / "child.py", code)
        return t.run_on_complete(self.md, [PY, "-I", str(script)], timeout, self.root)

    def test_success_shows_the_last_non_empty_line(self):
        line = self.run_code('import sys\nprint("前の行")\nprint("受け取った", sys.argv[-1])\nprint()\nprint("  ")\n')
        self.assertEqual(line, f"[E] 終わりの知らせ: exit 0（受け取った {os.path.abspath(self.md)}）")

    def test_success_without_output(self):
        self.assertEqual(self.run_code("pass\n"), "[E] 終わりの知らせ: exit 0")

    def test_failure_shows_exit_and_stderr(self):
        line = self.run_code('import sys\nprint("ふつうの出力")\nprint("合成の失敗", file=sys.stderr)\nsys.exit(3)\n')
        self.assertEqual(line, f"  [WARN] {t.ON_COMPLETE_FAILED}（exit 3: 合成の失敗）。{t.ON_COMPLETE_DONE_NOTE}")

    def test_failure_falls_back_to_stdout_and_to_nothing(self):
        line = self.run_code('import sys\nprint("出力だけ")\nsys.exit(4)\n')
        self.assertIn("（exit 4: 出力だけ）", line)
        self.assertEqual(self.run_code("import sys\nsys.exit(5)\n"),
                         f"  [WARN] {t.ON_COMPLETE_FAILED}（exit 5）。{t.ON_COMPLETE_DONE_NOTE}")

    def test_long_and_multi_line_output_is_one_short_line(self):
        line = self.run_code('print("a\\nb")\nprint("長い " * 300)\n')
        self.assertNotIn("\n", line)
        tail = line.split("（", 1)[1].rsplit("）", 1)[0]
        self.assertEqual(len(tail), t.ON_COMPLETE_TAIL_MAX_CHARS)
        self.assertTrue(tail.startswith("長い 長い"))

    def test_colors_and_control_characters_are_dropped(self):
        line = self.run_code('import sys\nsys.stdout.write("\\x1b[1;31m赤い\\x1b[0m\\x07 字\\tと\\x00タブ\\n")\n')
        self.assertEqual(line, "[E] 終わりの知らせ: exit 0（赤い 字 とタブ）")

    def test_progress_rewritten_with_cr_shows_the_last_state(self):
        line = self.run_code('import sys\nsys.stdout.write("進み 10%\\r進み 100%\\r\\n")\n')
        self.assertEqual(line, "[E] 終わりの知らせ: exit 0（進み 100%）")

    def test_output_that_is_not_utf8_does_not_break(self):
        # cp932 のバイト。日本語の Windows (ANSI コードページが 932) なら読める
        line = self.run_code('import sys\nsys.stdout.buffer.write("合成の字".encode("cp932") + b"\\n")\n')
        self.assertTrue(line.startswith("[E] 終わりの知らせ: exit 0"))
        self.assertNotIn("\n", line)
        if os.name == "nt" and "合成の字".encode("cp932").decode("mbcs", "replace") == "合成の字":
            self.assertEqual(line, "[E] 終わりの知らせ: exit 0（合成の字）")

    def test_utf8_mode_does_not_change_the_fallback(self):
        # Python の UTF-8 モードでは getpreferredencoding が UTF-8 を返すが、Windows の読み替えは mbcs のまま
        with mock.patch.object(locale, "getpreferredencoding", lambda do_setlocale=True: "UTF-8"):
            text = t._decode_output("合成の字".encode("cp932"))
        if os.name == "nt" and "合成の字".encode("cp932").decode("mbcs", "replace") == "合成の字":
            self.assertEqual(text, "合成の字")
        self.assertIsInstance(text, str)

    def test_timeout_stops_the_command_and_its_children(self):
        marker = self.root / "grandchild_alive.txt"
        script = _write(self.root / "sleeper.py", SLEEPER)
        start = time.monotonic()
        # 印のパスを渡すために、md の前の引数として印を置く (md は最後の引数)
        line = t.run_on_complete(self.md, [PY, "-I", str(script), str(marker)], 1, self.root)
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 15)
        self.assertEqual(line, f"  [WARN] 終わりの知らせのコマンドが 1 {t.ON_COMPLETE_TIMED_OUT}（眠ります）。"
                               f"{t.ON_COMPLETE_DONE_NOTE}")
        if os.name == "nt":  # taskkill /T で孫も止まる (Windows 以外は起こした子だけを止める)
            time.sleep(3.5)
            self.assertFalse(marker.exists(), "孫が止まらずに印を書いた")

    def test_missing_program_cannot_start(self):
        line = t.run_on_complete(self.md, [str(self.root / "no_such_program.exe")], 30, self.root)
        self.assertTrue(line.startswith(f"  [WARN] {t.ON_COMPLETE_CANNOT_START}（FileNotFoundError"), line)
        self.assertTrue(line.endswith(f"）。{t.ON_COMPLETE_DONE_NOTE}"))

    def test_batch_files_are_not_started(self):
        marker = self.root / "batch_ran.txt"
        for name in ("hook.cmd", "HOOK.BAT"):
            with self.subTest(name=name):
                bat = _write(self.root / name, f'@echo off\r\necho ran> "{marker}"\r\n')
                with mock.patch.object(t.subprocess, "Popen") as popen:
                    line = t.run_on_complete(self.md, [str(bat)], 30, self.root)
                popen.assert_not_called()
                self.assertIn(t.ON_COMPLETE_CANNOT_START, line)
                self.assertIn(".bat・.cmd は起こしません", line)
                # 差し替えなしでも起こさない
                t.run_on_complete(self.md, [str(bat)], 30, self.root)
                self.assertFalse(marker.exists())

    def test_cwd_is_the_project_root_and_no_shell(self):
        out = self.root / "seen.json"
        code = ("import json, os, sys\n"
                f"open({str(out)!r}, 'w', encoding='utf-8').write(json.dumps([os.getcwd(), sys.argv[1:]]))\n")
        line = self.run_code(code)
        self.assertTrue(line.startswith("[E]"), line)
        md = self.root / "out" / "2026-01-02_a & b %PATH% ^x.md"  # シェルなら読み替えられる字
        md.write_text("", encoding="utf-8")
        t.run_on_complete(md, [PY, "-I", str(self.root / "child.py")], 30, self.root)
        cwd, args = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(os.path.normcase(cwd), os.path.normcase(str(self.root)))
        self.assertEqual(args, [os.path.abspath(md)])

    def test_tail_reading_skips_a_cut_first_line(self):
        # 末尾 64 KB だけを読む。境目で切れた字があっても、最後の行は正しく読む
        with tempfile.TemporaryFile() as f:
            f.write("あ".encode("utf-8") * 30000 + b"\n" + "最後の行".encode("utf-8") + b"\n")
            self.assertEqual(t._read_last_line(f), "最後の行")
        with tempfile.TemporaryFile() as f:
            self.assertEqual(t._read_last_line(f), "")


class MainOnCompleteTest(unittest.TestCase):
    """main の中で起こす時・回数・出力。process_one は whisperx のスタブで本当に md を書く。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name)
        self.project = (self.base / "proj").resolve()
        self.inbox = self.project / "voice_memos" / "inbox"
        self.inbox.mkdir(parents=True)
        (self.inbox / "a.m4a").write_bytes(b"")
        self.log = self.base / "hook_log.jsonl"
        self.hook = _write(self.base / "hook.py", HOOK)
        engine.clear_model_cache()
        self.addCleanup(engine.clear_model_cache)
        patch = mock.patch.object(t, "load_prompt_encoder", lambda model_name="large-v3": (char_encoder, ""))
        patch.start()
        self.addCleanup(patch.stop)

    def config(self, on_complete: dict | None) -> None:
        data = {"on_complete": on_complete} if on_complete is not None else {}
        _write(self.project / "voice_memos" / "config.json", json.dumps(data, ensure_ascii=False))

    def hook_command(self) -> list[str]:
        return [PY, "-I", str(self.hook), str(self.log), str(self.inbox)]

    def main(self, *extra) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        argv = ["transcribe.py", "--project", str(self.project), "--no-global", "--no-proofread", *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(sys.modules, {"whisperx": _FakeWhisperx(SEGS)}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = t.main()
        return code, out.getvalue(), err.getvalue()

    def records(self) -> list[dict]:
        if not self.log.exists():
            return []
        return [json.loads(ln) for ln in self.log.read_text(encoding="utf-8").splitlines()]

    def transcripts(self) -> list[Path]:
        d = self.project / "voice_memos" / "transcripts"
        return sorted(d.glob("*.md")) if d.is_dir() else []

    def test_default_config_starts_nothing(self):
        with mock.patch.object(t.subprocess, "Popen") as popen:
            code, out, _ = self.main()
        self.assertEqual(code, 0)
        popen.assert_not_called()
        self.assertNotIn("[E]", out)
        self.assertEqual(len(self.transcripts()), 1)

    def test_runs_after_the_md_is_written_and_the_audio_is_moved(self):
        self.config({"command": self.hook_command()})
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        [md] = self.transcripts()
        [rec] = self.records()
        self.assertEqual(rec["args"], [str(self.log), str(self.inbox), os.path.abspath(md)])
        self.assertTrue(rec["md_is_absolute"])
        self.assertEqual(os.path.normcase(rec["cwd"]), os.path.normcase(str(self.project)))
        self.assertTrue(rec["md_exists"])
        self.assertEqual(rec["first_line"], "---")
        self.assertEqual(rec["hidden_left"], [])  # 書きかけの一時ファイル (.<名前>.tmp) は無い
        self.assertEqual(rec["inbox"], [])  # 音声は退いている
        self.assertTrue((self.project / "voice_memos" / "processed" / "a.m4a").exists())
        lines = out.splitlines()
        start = f"[E] 終わりの知らせ: md を書くたびに走らせます（上限 120 秒）: {subprocess.list2cmdline(self.hook_command())}"
        self.assertIn(start, lines)
        done = f"[E] 終わりの知らせ: exit 0（受け取った {md.name}）"
        self.assertIn(done, lines)
        self.assertLess(lines.index(f"  -> {md}"), lines.index(done))
        self.assertLess(lines.index(done), lines.index("完了: 1/1 件"))

    def test_keep_leaves_the_audio_and_still_runs(self):
        self.config({"command": self.hook_command()})
        code, _, _ = self.main("--keep")
        self.assertEqual(code, 0)
        [rec] = self.records()
        self.assertEqual(rec["inbox"], ["a.m4a"])
        self.assertTrue(rec["md_exists"])

    def test_once_per_md(self):
        (self.inbox / "b.m4a").write_bytes(b"")
        self.config({"command": self.hook_command()})
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        mds = self.transcripts()
        self.assertEqual(len(mds), 2)
        self.assertEqual(sorted(r["args"][-1] for r in self.records()), sorted(os.path.abspath(m) for m in mds))
        self.assertEqual(out.count("[E] 終わりの知らせ: exit 0"), 2)

    def test_no_run_for_audio_whose_md_was_not_written(self):
        (self.inbox / "b.m4a").write_bytes(b"")
        self.config({"command": self.hook_command()})
        real = t.process_one

        def flaky(audio, **kw):
            if audio.name == "a.m4a":
                raise RuntimeError("合成の失敗")
            return real(audio, **kw)

        with mock.patch.object(t, "process_one", flaky):
            code, out, err = self.main()
        self.assertEqual(code, 0)
        [rec] = self.records()
        self.assertTrue(rec["args"][-1].endswith("_b.md"))
        self.assertIn("a.m4a の処理に失敗", err)
        self.assertIn("完了: 1/2 件", out)

    def test_failing_command_does_not_fail_the_transcription(self):
        script = _write(self.base / "fail.py", 'import sys\nprint("合成の失敗", file=sys.stderr)\nsys.exit(3)\n')
        self.config({"command": [PY, "-I", str(script)]})
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.transcripts()), 1)
        self.assertIn(f"  [WARN] {t.ON_COMPLETE_FAILED}（exit 3: 合成の失敗）。{t.ON_COMPLETE_DONE_NOTE}", out.splitlines())
        self.assertIn("完了: 1/1 件", out)

    def test_missing_command_does_not_fail_the_transcription(self):
        self.config({"command": [str(self.base / "no_such_program.exe")]})
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertIn(t.ON_COMPLETE_CANNOT_START, out)
        self.assertIn("完了: 1/1 件", out)

    def test_timeout_from_config(self):
        script = _write(self.base / "sleep.py", "import time\ntime.sleep(30)\n")
        self.config({"command": [PY, "-I", str(script)], "timeout_sec": 1})
        start = time.monotonic()
        code, out, _ = self.main()
        self.assertLess(time.monotonic() - start, 15)
        self.assertEqual(code, 0)
        self.assertIn("（上限 1 秒）", out)
        self.assertIn(f"  [WARN] 終わりの知らせのコマンドが 1 {t.ON_COMPLETE_TIMED_OUT}。{t.ON_COMPLETE_DONE_NOTE}",
                      out.splitlines())

    def test_no_on_complete_flag(self):
        self.config({"command": self.hook_command()})
        code, out, _ = self.main("--no-on-complete")
        self.assertEqual(code, 0)
        self.assertEqual(self.records(), [])
        self.assertNotIn("[E]", out)
        self.assertEqual(len(self.transcripts()), 1)

    def test_list_starts_nothing(self):
        self.config({"command": self.hook_command()})
        code, out, _ = self.main("--list")
        self.assertEqual(code, 0)
        self.assertEqual(self.records(), [])
        self.assertNotIn("[E]", out)

    def test_wrong_config_stops_before_transcribing(self):
        for extra in ((), ("--list",), ("--no-on-complete",)):
            with self.subTest(extra=extra):
                self.config({"command": "node x.mjs"})
                code, _, err = self.main(*extra)
                self.assertEqual(code, 1)
                self.assertIn("ERROR: 設定の誤り: on_complete.command", err)
                self.assertEqual(self.transcripts(), [])
                self.assertTrue((self.inbox / "a.m4a").exists())


class DocsTest(unittest.TestCase):
    """README・SKILL.md・config.default.json が、コードと同じ文と値で書いている。"""

    def setUp(self):
        self.skill = (Path(t.SKILL_DIR) / "SKILL.md").read_text(encoding="utf-8")
        self.readme = (Path(t.SKILL_DIR) / "README.md").read_text(encoding="utf-8")

    def test_line_texts_are_the_same_in_code_and_readme(self):
        for text in (t.ON_COMPLETE_FAILED, t.ON_COMPLETE_TIMED_OUT, t.ON_COMPLETE_CANNOT_START,
                     t.ON_COMPLETE_DONE_NOTE, "[E] 終わりの知らせ: exit 0", "[E] 終わりの知らせ: md を書くたびに走らせます"):
            with self.subTest(text=text):
                self.assertIn(text, self.readme)

    def test_readme_documents_the_setting(self):
        for word in ("on_complete", '"command"', '"timeout_sec"', "--no-on-complete", ".bat", ".cmd"):
            with self.subTest(word=word):
                self.assertIn(word, self.readme)

    def test_skill_tells_the_user_about_the_warning(self):
        self.assertIn("[WARN] 終わりの知らせ", self.skill)

    def test_default_config(self):
        cfg = json.loads((Path(t.SKILL_DIR) / "config.default.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["on_complete"], {"command": [], "timeout_sec": t.ON_COMPLETE_DEFAULT_TIMEOUT_SEC})


if __name__ == "__main__":
    unittest.main()
