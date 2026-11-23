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
        self.assertEqual(t.on_complete_settings({"on_complete": {"command": ["node"], "timeout_sec": 86400}}),
                         (["node"], 86400.0))

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
            # PR #17 のレビュー 1 回目の任意 3: Windows の待ち (ミリ秒の 32 ビット) で回り込む値と、float にできない整数
            {"command": ["node"], "timeout_sec": 86400.5},
            {"command": ["node"], "timeout_sec": 4294967.296},
            {"command": ["node"], "timeout_sec": 10 ** 400},
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
        argv = t.on_complete_argv([PY, "tools/x.mjs", "--flag"], self.md, self.root)
        self.assertEqual(argv, [PY, "tools/x.mjs", "--flag", os.path.abspath(self.md)])

    def test_relative_md_path_becomes_absolute(self):
        argv = t.on_complete_argv(["node"], Path("rel") / "a.md", self.root)
        self.assertTrue(os.path.isabs(argv[-1]))
        self.assertEqual(argv[-1], os.path.abspath(Path("rel") / "a.md"))

    def test_relative_program_with_a_separator_is_from_the_project_root(self):
        for prog in ("tools/hook.exe", "tools\\hook.exe", "./hook.exe"):
            with self.subTest(prog=prog):
                self.assertEqual(t.on_complete_argv([prog], self.md, self.root)[0], str(self.root / prog))

    def test_absolute_program_is_kept(self):
        self.assertEqual(t.on_complete_argv([PY], self.md, self.root)[0], PY)


@unittest.skipUnless(os.name == "nt", "PATHEXT と CreateProcess の探し方は Windows のもの")
class FindOnPathTest(unittest.TestCase):
    """区切りの無い名前は PATH のフォルダだけで引く (PR #17 のレビュー 1 回目の推奨 2)。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.a, self.b, self.cwd = self.root / "a", self.root / "b", self.root / "cwd"
        for d in (self.a, self.b, self.cwd):
            d.mkdir()
        old = os.getcwd()
        os.chdir(self.cwd)
        self.addCleanup(os.chdir, old)

    def env(self, *dirs: Path, pathext: str = ".COM;.EXE;.BAT;.CMD"):
        return mock.patch.dict(os.environ, {"PATH": os.pathsep.join(str(d) for d in dirs), "PATHEXT": pathext})

    def test_first_folder_on_path_with_pathext(self):
        (self.b / "tool.EXE").write_bytes(b"")
        with self.env(self.a, self.b):
            self.assertEqual(t._find_on_path("tool"), str(self.b / "tool.exe"))  # 拡張子は小文字で試す
        (self.a / "tool.CMD").write_bytes(b"")
        with self.env(self.a, self.b):  # フォルダの順が先 (a の .cmd が b の .exe より先に見つかる)
            self.assertEqual(os.path.normcase(t._find_on_path("tool")), os.path.normcase(str(self.a / "tool.cmd")))

    def test_current_folder_and_relative_entries_are_not_searched(self):
        (self.cwd / "tool.exe").write_bytes(b"")
        with self.env(self.a, Path(".")):
            self.assertIsNone(t._find_on_path("tool"))
        with self.env(self.a):
            with self.assertRaises(FileNotFoundError) as cm:
                t.on_complete_argv(["tool"], self.root / "x.md", self.root)
        self.assertIn("PATH に見つかりません: tool", str(cm.exception))

    def test_name_with_an_extension_is_tried_as_it_is(self):
        (self.a / "tool.exe").write_bytes(b"")
        with self.env(self.a):
            self.assertEqual(t._find_on_path("tool.exe"), str(self.a / "tool.exe"))
            self.assertEqual(t.on_complete_argv(["tool.exe"], self.root / "x.md", self.root)[0],
                             str(self.a / "tool.exe"))


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

    @unittest.skipUnless(os.name == "nt", "cmd.exe に読ませるのは Windows")
    def test_batch_files_and_cmd_are_not_started(self):
        marker = self.root / "batch_ran.txt"
        _write(self.root / "hook.cmd", f'@echo off\r\necho ran> "{marker}"\r\n')
        _write(self.root / "HOOK.BAT", f'@echo off\r\necho ran> "{marker}"\r\n')
        cmd_exe = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "cmd.exe")
        cases = [
            [str(self.root / "hook.cmd")],
            [str(self.root / "HOOK.BAT")],
            # PR #17 のレビュー 1 回目の任意 1: Windows は名前の末尾の空白と . を落として開く
            [str(self.root / "hook.cmd ")],
            [str(self.root / "hook.cmd.")],
            # 推奨 3: cmd /c を最初に書く形
            ["cmd", "/c", "echo"],
            [cmd_exe, "/c", "echo"],
            [cmd_exe[:-len(".exe")] + " ", "/c", "echo"],  # 拡張子なし + 末尾の空白 (Windows は cmd.exe を開く)
            [str(self.root / "tool.ps1")],  # .exe でないもの
        ]
        for command in cases:
            with self.subTest(command=command):
                with mock.patch.object(t.subprocess, "Popen") as popen:
                    line = t.run_on_complete(self.md, command, 30, self.root)
                popen.assert_not_called()
                self.assertIn(t.ON_COMPLETE_CANNOT_START, line)
                self.assertIn("起こすのは実行ファイル（.exe）だけです", line)
                # 差し替えなしでも起こさない
                t.run_on_complete(self.md, command, 30, self.root)
                self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "nt", "拡張子で決めるのは Windows")
    def test_executables_are_allowed_even_with_a_trailing_space_or_dot(self):
        for exe in (PY, PY + " ", PY + ".", r"C:\tools\hook", r"C:\tools\hook.COM", r"C:\my.tools\hook"):
            with self.subTest(exe=exe):
                self.assertFalse(t._unsafe_program(exe))
        for exe in (r"C:\tools\hook.cmd", r"C:\tools\hook.bat ", r"C:\tools\cmd", r"C:\tools\CMD.EXE. ", r"C:\tools\x.ps1"):
            with self.subTest(exe=exe):
                self.assertTrue(t._unsafe_program(exe))

    @unittest.skipUnless(os.name == "nt", "PATHEXT は Windows のもの")
    def test_bare_name_that_is_a_cmd_on_path_is_refused_with_the_reason(self):
        # npm のように、名前だけで書いたものの中身が .cmd のとき
        bindir = self.root / "bin"
        _write(bindir / "npmlike.cmd", "@echo off\r\n")
        with mock.patch.dict(os.environ, {"PATH": str(bindir)}), mock.patch.object(t.subprocess, "Popen") as popen:
            line = t.run_on_complete(self.md, ["npmlike", "run", "x"], 30, self.root)
        popen.assert_not_called()
        # PR #17 のレビュー 2 回目の任意 1: 書いた名前そのものを勧めず、.exe の絶対パスを勧める
        self.assertEqual(line, f"  [WARN] {t.ON_COMPLETE_CANNOT_START}（npmlike を PATH で引くと npmlike.cmd になるため"
                               f"起こしません。{t.ON_COMPLETE_EXE_ABSOLUTE}）。{t.ON_COMPLETE_DONE_NOTE}")

    def test_bare_name_is_looked_up_on_path_and_run(self):
        out = self.root / "argv0.txt"
        script = _write(self.root / "argv0.py", f"import sys\nopen({str(out)!r}, 'w', encoding='utf-8').write(sys.executable)\n")
        name = Path(PY).stem  # python
        with mock.patch.dict(os.environ, {"PATH": str(Path(PY).parent)}):
            line = t.run_on_complete(self.md, [name, "-I", str(script)], 30, self.root)
        self.assertEqual(line, "[E] 終わりの知らせ: exit 0")
        # PR #17 のレビュー 2 回目の任意 2: 起きたのは PATH で引いた python (元の名前を CreateProcess に渡すと、
        # 起こした側の exe のフォルダの python が先に見つかる)
        self.assertEqual(os.path.normcase(out.read_text(encoding="utf-8")), os.path.normcase(PY))

    def test_missing_bare_name_cannot_start(self):
        with mock.patch.dict(os.environ, {"PATH": str(self.root / "empty")}):
            line = t.run_on_complete(self.md, ["no_such_tool_for_test"], 30, self.root)
        self.assertEqual(line, f"  [WARN] {t.ON_COMPLETE_CANNOT_START}（FileNotFoundError: PATH に見つかりません: "
                               f"no_such_tool_for_test）。{t.ON_COMPLETE_DONE_NOTE}")

    def test_returns_while_a_grandchild_still_holds_the_output(self):
        # stdout・stderr を一時ファイルで受ける理由 (#4 の U5。PR #17 のレビュー 1 回目の任意 5):
        # pipe だと、孫が出力を持っている間は、子が終わっても返らない
        code = ("import subprocess, sys, tempfile\n"
                "subprocess.Popen([sys.executable, '-I', '-c', 'import time; time.sleep(6)'],"
                " stdout=sys.stdout, stderr=sys.stderr, cwd=tempfile.gettempdir())\n"
                "print('孫を残して終わります')\n")
        start = time.monotonic()
        line = self.run_code(code)
        self.assertLess(time.monotonic() - start, 4)
        self.assertEqual(line, "[E] 終わりの知らせ: exit 0（孫を残して終わります）")

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

    def test_lines_in_different_encodings_are_read_one_by_one(self):
        # PR #17 のレビュー 1 回目の任意 2: UTF-8 でない行が混じっても、最後の UTF-8 の行は化けない
        with tempfile.TemporaryFile() as f:
            f.write("前の行".encode("cp932") + b"\r\n" + "最後の行".encode("utf-8") + b"\r\n")
            self.assertEqual(t._read_last_line(f), "最後の行")
        if os.name == "nt" and "合成の字".encode("cp932").decode("mbcs", "replace") == "合成の字":
            with tempfile.TemporaryFile() as f:
                f.write("前の行".encode("utf-8") + b"\n" + "合成の字".encode("cp932") + b"\n")
                self.assertEqual(t._read_last_line(f), "合成の字")


# md を動かして、標準出力の最後の行に「moved: <絶対パス>」を出す子 (#20)。引数: <動かす先のフォルダ> <md>
MOVER = r'''
import os, sys
dest_dir, md = sys.argv[1], sys.argv[-1]
os.makedirs(dest_dir, exist_ok=True)
dest = os.path.join(dest_dir, os.path.basename(md))
os.replace(md, dest)
print("並べました")
print("moved: " + dest)
'''


class MovedTest(unittest.TestCase):
    """md の移り先 (#20): 通ったときの標準出力の最後の空でない行が「moved: <絶対パス>」なら、そこにファイルがあるかを確かめる。"""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.md = _write(self.root / "out" / "2026-01-02_memo.md", "---\n---\n")

    def run_child(self, code: str, *args: str) -> t.OnCompleteResult:
        script = _write(self.root / "child.py", code)
        return t.run_on_complete_result(self.md, [PY, "-I", str(script), *args], 30, self.root)

    def run_writer(self, data: bytes, exit_code: int = 0) -> t.OnCompleteResult:
        """子は、渡したバイト列をそのまま標準出力に書く (字の符号を決めて試す)。"""
        blob = self.root / "out.bin"
        blob.write_bytes(data)
        return self.run_child("import sys\nsys.stdout.buffer.write(open(sys.argv[1], 'rb').read())\n"
                              f"sys.exit({exit_code})\n", str(blob))

    def assert_moved(self, result: t.OnCompleteResult, dest: Path) -> None:
        self.assertTrue(result.ok, result.line)
        self.assertEqual(result.moved_to, os.path.abspath(dest))
        self.assertEqual(result.moved_line, t.ON_COMPLETE_MOVED_HEAD + os.path.abspath(dest))

    def assert_not_moved(self, result: t.OnCompleteResult) -> None:
        self.assertEqual((result.moved_to, result.moved_line), (None, ""))

    def assert_unusable(self, result: t.OnCompleteResult, reason: str) -> None:
        self.assertTrue(result.ok, result.line)
        self.assertIsNone(result.moved_to)
        self.assertTrue(result.moved_line.startswith(f"{t.ON_COMPLETE_WARN_HEAD}{t.ON_COMPLETE_MOVED_UNUSABLE}（{reason}"),
                        result.moved_line)
        self.assertNotIn("\n", result.moved_line)

    def test_moved_md_is_read_from_the_last_line(self):
        dest_dir = self.root / "保管"
        result = self.run_child(MOVER, str(dest_dir))
        dest = dest_dir / self.md.name
        self.assertTrue(dest.is_file())
        self.assertFalse(self.md.exists())
        self.assert_moved(result, dest)
        # ランチャーの 1 行は今までどおり (要点は最後の行)
        self.assertEqual(result.line, f"{t.ON_COMPLETE_OK_HEAD}（moved: {dest}）")

    def test_spaces_in_the_path_are_kept(self):
        # 要点は空白をまとめるが、移り先は空白を変えずに読む (続く空白・全角の空白。タブは Windows の名前に使えない)
        dest = _write(self.root / "保管  の\u3000場所" / "合成 \u3000 の  記録.md", "")
        result = self.run_writer(("moved: " + str(dest) + "\n").encode("utf-8"))
        self.assert_moved(result, dest)
        self.assertNotIn("\u3000", result.line)  # 要点の側は、全角の空白も空白 1 つにまとめる (今までどおり)

    def test_long_path_is_not_cut(self):
        # 要点は 200 字で切るが、移り先は切らない
        dest_dir = self.root
        while len(str(dest_dir)) < 200:
            dest_dir = dest_dir / "合成のフォルダ"
        dest = _write(dest_dir / "合成の長い名前の記録.md", "")
        self.assertGreater(len(str(dest)), t.ON_COMPLETE_TAIL_MAX_CHARS)
        result = self.run_writer(("moved: " + str(dest) + "\n").encode("utf-8"))
        self.assert_moved(result, dest)

    def test_crlf_colors_and_no_space_after_the_colon(self):
        dest = _write(self.root / "保管" / "記録.md", "")
        for data in ((f"moved: {dest}\r\n").encode("utf-8"),
                     (f"\x1b[32mmoved: {dest}\x1b[0m\n").encode("utf-8"),
                     (f"moved:{dest}  \n\n  \n").encode("utf-8")):
            with self.subTest(data=data):
                self.assert_moved(self.run_writer(data), dest)

    def test_moved_to_is_normalized(self):
        # README の約束: moved_to と移り先の行は、正規化した絶対パス (PR #21 のレビュー 1 回目の推奨 1)
        dest = _write(self.root / "保管" / "記録.md", "")
        written = str(self.root / "保管" / "下" / ".." / "記録.md").replace("\\", "/")
        result = self.run_writer(f"moved: {written}\n".encode("utf-8"))
        self.assert_moved(result, dest)
        self.assertNotIn("..", result.moved_to)
        if os.name == "nt":
            self.assertNotIn("/", result.moved_to)

    def test_moved_must_start_the_line(self):
        # 行の途中の moved: は読まない (同、推奨 1)。前の見えない字 (BOM・BEL・ゼロ幅の空白) は外して読む (同、任意 1)
        dest = _write(self.root / "保管" / "記録.md", "")
        self.assert_not_moved(self.run_writer(f"済みました moved: {dest}\n".encode("utf-8")))
        for head in ("﻿", "\x07", "​", "﻿​ "):
            with self.subTest(head=head):
                self.assert_moved(self.run_writer(f"{head}moved: {dest}\n".encode("utf-8")), dest)

    def test_invisible_characters_inside_the_mark_are_a_warning(self):
        # 見えない字を落とすと moved: の行に見えるのに、そのままでは読めない行は、黙らずに WARN (同、任意 1)
        dest = _write(self.root / "保管" / "記録.md", "")
        for data in (f"moved: {dest}\n".encode("utf-16-le"), f"mo​ved: {dest}\n".encode("utf-8")):
            with self.subTest(data=data[:12]):
                result = self.run_writer(data)
                self.assertIn("moved:", result.line)  # 要点には moved: と出る
                self.assert_unusable(result, "印字できない字が混じっています")

    def test_path_in_the_windows_code_page(self):
        # UTF-8 で読めない行は Windows の既定のコードページで読む (要点と同じ)
        dest = _write(self.root / "保管" / "合成の記録.md", "")
        text = "moved: " + str(dest) + "\n"
        try:
            ok = os.name == "nt" and text.encode("cp932").decode("mbcs", "replace") == text
        except UnicodeEncodeError:  # 一時ディレクトリのパスが cp932 で書けない
            ok = False
        if not ok:
            self.skipTest("Windows の既定のコードページが cp932 のときだけ")
        self.assert_moved(self.run_writer(text.encode("cp932")), dest)

    def test_only_the_last_non_empty_line_of_stdout_counts(self):
        dest = _write(self.root / "保管" / "記録.md", "")
        self.assert_not_moved(self.run_writer(f"moved: {dest}\n終わりました\n".encode("utf-8")))
        self.assert_not_moved(self.run_child(f"import sys\nprint('moved: ' + {str(dest)!r}, file=sys.stderr)\n"))
        self.assert_not_moved(self.run_writer(f"Moved: {dest}\n".encode("utf-8")))  # 書き方は小文字の moved: だけ
        self.assert_not_moved(self.run_writer(b""))

    def test_failed_or_timed_out_command_is_not_read(self):
        dest = _write(self.root / "保管" / "記録.md", "")
        failed = self.run_writer(f"moved: {dest}\n".encode("utf-8"), exit_code=3)
        self.assertFalse(failed.ok)
        self.assert_not_moved(failed)
        script = _write(self.root / "slow.py", f"import time\nprint('moved: ' + {str(dest)!r}, flush=True)\ntime.sleep(30)\n")
        slow = t.run_on_complete_result(self.md, [PY, "-I", str(script)], 1, self.root)
        self.assertTrue(slow.timed_out)
        self.assert_not_moved(slow)

    def test_unusable_paths_are_warnings(self):
        missing = self.root / "保管" / "無い.md"
        self.assert_unusable(self.run_writer(f"moved: {missing}\n".encode("utf-8")), f"ファイルがありません: {missing}")
        self.assert_unusable(self.run_writer(b"moved: out/2026-01-02_memo.md\n"),
                             "絶対パスではありません: out/2026-01-02_memo.md")
        self.assert_unusable(self.run_writer(b"moved:\n"), "パスがありません")
        self.assert_unusable(self.run_writer(f"moved: {self.root}\n".encode("utf-8")),
                             f"ファイルがありません: {self.root}")  # フォルダ
        self.assert_unusable(self.run_writer(f'moved: "{self.md}"\n'.encode("utf-8")), "絶対パスではありません")  # 引用符で囲まない
        if os.name == "nt":  # 今のドライブからのパス (\x・/x) は、絶対パスとみなさない
            drive_relative = os.path.splitdrive(os.path.abspath(self.md))[1]
            self.assert_unusable(self.run_writer(f"moved: {drive_relative}\n".encode("utf-8")),
                                 f"絶対パスではありません: {drive_relative}")

    def test_the_moved_line_is_one_line_even_for_odd_output(self):
        result = self.run_writer(b"moved: C:/\x00\x07\xff\n")
        self.assertTrue(result.ok)
        self.assertIsNone(result.moved_to)
        self.assertTrue(result.moved_line.startswith(t.ON_COMPLETE_WARN_HEAD), result.moved_line)
        for ch in ("\n", "\x00", "\x07"):
            self.assertNotIn(ch, result.moved_line)

    def test_run_on_complete_line_is_unchanged(self):
        # ほかのスキルとの約束: run_on_complete は 1 行の str のまま (移り先の行は足さない)
        script = _write(self.root / "mover.py", MOVER)
        line = t.run_on_complete(self.md, [PY, "-I", str(script), str(self.root / "保管")], 30, self.root)
        self.assertEqual(line, f"{t.ON_COMPLETE_OK_HEAD}（moved: {self.root / '保管' / self.md.name}）")


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

    def test_moved_md_line_follows_the_exit_line(self):
        # #20: コマンドが md を動かして moved: の行を出すと、exit 0 の行の次に移り先の行が出る (: の後ろはパスだけ)
        mover = _write(self.base / "mover.py", MOVER)
        archive = self.base / "保管"
        self.config({"command": [PY, "-I", str(mover), str(archive)]})
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertEqual(self.transcripts(), [])  # 置き場からは動いた
        [dest] = sorted(archive.glob("*.md"))
        lines = out.splitlines()
        written = next(ln for ln in lines if ln.startswith("  -> "))
        self.assertFalse(Path(written[len("  -> "):]).exists())
        done = f"{t.ON_COMPLETE_OK_HEAD}（moved: {dest}）"
        moved = f"{t.ON_COMPLETE_MOVED_HEAD}{os.path.abspath(dest)}"
        audio = f"  moved: {self.project / 'voice_memos' / 'processed' / 'a.m4a'}"  # 音声を退かした行 (前からある行)
        self.assertEqual([lines.index(x) for x in (written, audio, done, moved)],
                         sorted(lines.index(x) for x in (written, audio, done, moved)))
        self.assertEqual(lines.index(moved), lines.index(done) + 1)
        self.assertEqual(Path(moved[len(t.ON_COMPLETE_MOVED_HEAD):]), dest)
        self.assertIn("完了: 1/1 件", lines)

    def test_moved_line_per_md(self):
        (self.inbox / "b.m4a").write_bytes(b"")
        mover = _write(self.base / "mover.py", MOVER)
        archive = self.base / "保管"
        self.config({"command": [PY, "-I", str(mover), str(archive)]})
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        lines = out.splitlines()
        moved = [ln for ln in lines if ln.startswith(t.ON_COMPLETE_MOVED_HEAD)]
        self.assertEqual(sorted(Path(ln[len(t.ON_COMPLETE_MOVED_HEAD):]) for ln in moved), sorted(archive.glob("*.md")))
        self.assertEqual(len(moved), 2)
        for ln in moved:  # どれも、その音声の exit 0 の行のすぐ後
            self.assertTrue(lines[lines.index(ln) - 1].startswith(t.ON_COMPLETE_OK_HEAD))

    def test_unusable_moved_line_does_not_fail_the_transcription(self):
        script = _write(self.base / "lie.py", f"print('moved: ' + {str(self.base / '無い.md')!r})\n")
        self.config({"command": [PY, "-I", str(script)]})
        code, out, _ = self.main()
        self.assertEqual(code, 0)
        self.assertEqual(len(self.transcripts()), 1)  # md は置き場に残っている
        lines = out.splitlines()
        warn = f"{t.ON_COMPLETE_WARN_HEAD}{t.ON_COMPLETE_MOVED_UNUSABLE}（ファイルがありません: {self.base / '無い.md'}）"
        self.assertIn(warn, lines)
        self.assertTrue(lines[lines.index(warn) - 1].startswith(t.ON_COMPLETE_OK_HEAD))
        self.assertNotIn(t.ON_COMPLETE_MOVED_HEAD, out)
        self.assertIn("完了: 1/1 件", lines)

    def test_no_moved_line_without_the_mark(self):
        self.config({"command": self.hook_command()})
        _, out, _ = self.main()
        self.assertNotIn(t.ON_COMPLETE_MOVED_HEAD, out)
        self.assertNotIn(t.ON_COMPLETE_MOVED_UNUSABLE, out)

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

    def test_list_shows_the_setting_and_starts_nothing(self):
        self.config({"command": self.hook_command(), "timeout_sec": 30})
        code, out, _ = self.main("--list")
        self.assertEqual(code, 0)
        self.assertEqual(self.records(), [])
        # PR #17 のレビュー 1 回目の推奨 1: 待つ時間を前もって見積もれるように、設定の行は --list でも出す
        lines = out.splitlines()
        start = f"[E] 終わりの知らせ: md を書くたびに走らせます（上限 30 秒）: {subprocess.list2cmdline(self.hook_command())}"
        self.assertIn(start, lines)
        self.assertLess(lines.index("対象 1 件"), lines.index(start))
        self.assertNotIn("exit", out)
        code, out, _ = self.main("--list", "--no-on-complete")
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
        for word in ("on_complete", '"command"', '"timeout_sec"', "--no-on-complete", ".bat", ".cmd",
                     "PATH のフォルダだけで引く", "PATH に見つかりません", "powershell -Command", '"-File"',
                     "秘密は環境変数で渡す", str(t.ON_COMPLETE_MAX_TIMEOUT_SEC)):
            with self.subTest(word=word):
                self.assertIn(word, self.readme)

    def test_skill_tells_the_user_about_the_warning(self):
        self.assertIn("[WARN] 終わりの知らせ", self.skill)

    def test_skill_counts_the_wait_in_the_estimate(self):
        # PR #17 のレビュー 1 回目の推奨 1
        self.assertIn("[E] 終わりの知らせ: md を書くたびに走らせます", self.skill)
        self.assertIn("対象の数 × 行の上限の秒", self.skill)

    def test_moved_lines_are_the_same_in_code_and_docs(self):
        # #20: コマンドに出してもらう行、キットが出す移り先の行と WARN の行を、README・SKILL.md がコードと同じ文で書く
        for doc in (self.readme, self.skill):
            for text in (t.ON_COMPLETE_MOVED_HEAD.rstrip(), f"[WARN] {t.ON_COMPLETE_MOVED_UNUSABLE}"):
                with self.subTest(text=text):
                    self.assertIn(text, doc)
        self.assertIn(f"`{t.ON_COMPLETE_MOVED_MARK} <新しい絶対パス>`", self.readme)
        self.assertIn("標準出力の最後の空でない行", self.readme)
        self.assertNotIn("`/voice-memo` はランチャーの `->` の行の md を読めなくなる", self.readme)  # 0.9.0 までの文

    def test_skill_reads_the_moved_path_and_asks_when_missing(self):
        # #20: §3 は移り先の行のパスを読み、md が見つからなければ推測で探さず利用者に聞く
        section3 = self.skill.split("## 3.", 1)[1].split("\n## ", 1)[0]
        self.assertIn(t.ON_COMPLETE_MOVED_HEAD.rstrip(), section3)
        self.assertIn("移り先の行のパスを読む", section3)
        self.assertIn("推測で探し回らず", section3)
        self.assertIn("利用者に聞く", section3)
        # §2 の終わりの確かめは、md を動かす設定ではフォルダの数で見ない
        self.assertIn("フォルダの md の数では見ずに `完了: m/n 件` の行で見る", self.skill)

    def test_skill_section2_tells_what_the_moved_lines_mean(self):
        # PR #21 のレビュー 1 回目の推奨 2・任意 2: §2 は、移り先の行の意味と、WARN の行のときは先に -> のパスを読むことを書く
        section2 = self.skill.split("## 2.", 1)[1].split("\n## ", 1)[0]
        moved_item = section2.split(f"- `{t.ON_COMPLETE_MOVED_HEAD}<パス>` の行は", 1)[1].split("\n- ", 1)[0]
        self.assertIn("§3 ではこのパスを読む", moved_item)
        self.assertIn("`  moved: <パス>` の行（音声の退避先）とは別物", moved_item)
        warn_item = section2.split(f"- `[WARN] {t.ON_COMPLETE_MOVED_UNUSABLE}` で始まる行が出たら", 1)[1].split("\n- ", 1)[0]
        self.assertIn("§3 では `->` の行のパスを読む。そこにも md が無ければ", warn_item)
        self.assertIn("md が動いていないこともある", warn_item)
        # 成否の見方は、md の置き場のフォルダではなく、終了コードと完了の行
        self.assertIn("成否は終了コードと `完了: m/n 件` の行で判断する", section2)
        self.assertNotIn("transcript 生成有無で判断する", self.skill)

    def test_default_config(self):
        cfg = json.loads((Path(t.SKILL_DIR) / "config.default.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["on_complete"], {"command": [], "timeout_sec": t.ON_COMPLETE_DEFAULT_TIMEOUT_SEC})


if __name__ == "__main__":
    unittest.main()
