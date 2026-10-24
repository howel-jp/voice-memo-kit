"""import しただけで起きること (プロセスの起動・ソケット・DLL の読み込み・重い import) を確かめる補助。

tests/test_kit_api.py (キットの scripts) と tests/test_measure.py (tools/measure) の ImportSideEffectTest が使う。
もとは 2 つのファイルに同じ前置きを書いていて、同じ穴 (ctypes の塞ぎが CDLL・WinDLL・cdll・windll だけ) が
2 か所にあった (#12、PR #7 のレビュー 2 回目の任意 3)。塞ぐものは、ここ 1 か所で直す。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# 子プロセスの中で、import の前に実行する文。塞いだものが呼ばれると AssertionError で落ちる
GUARD = (
    "import sys, os, subprocess, socket, ctypes\n"
    "class Boom:\n"
    "    def __init__(self, *a, **k): raise AssertionError('side effect on import')\n"
    "class BoomLoader:\n"
    "    def __getattr__(self, name): raise AssertionError('side effect on import')\n"
    "def boom(*a, **k): raise AssertionError('side effect on import')\n"
    "subprocess.Popen = Boom; socket.socket = Boom; socket.create_connection = boom\n"
    # os のプロセスの起動
    "for n in dir(os):\n"
    "    if n.startswith(('spawn', 'exec')) or n in ('system', 'startfile', 'popen'):\n"
    "        setattr(os, n, boom)\n"
    # ctypes の DLL の読み込み。CDLL.__init__ は _dlopen を呼ぶので、_dlopen を塞ぐと CDLL の派生
    # (WinDLL・OleDLL・PyDLL) と LibraryLoader (cdll・windll・oledll・pydll) のすべてに効く
    "ctypes._dlopen = boom\n"
    "ctypes.CDLL = Boom; ctypes.WinDLL = Boom\n"
    "ctypes.cdll = ctypes.windll = BoomLoader()\n"
)

# import しただけでは読み込まない重いモジュール
HEAVY = ("torch", "whisperx", "faster_whisper", "ctranslate2", "tokenizers", "huggingface_hub", "numpy", "pyannote")


def import_in_child(path: Path, modules: list[str], heavy: tuple[str, ...] = HEAVY) -> subprocess.CompletedProcess:
    """`python -I` の子プロセスで、GUARD の後に path から modules を import する。

    標準出力は `loaded:` に続けて、読み込まれた重いモジュールの名前 (無ければ `loaded:` だけ)。
    """
    code = (
        GUARD
        + f"sys.path.insert(0, {str(path)!r})\n"
        + f"import {', '.join(modules)}\n"
        + f"heavy = {list(heavy)!r}\n"
        + "print('loaded:' + ','.join(m for m in heavy if m in sys.modules))\n"
    )
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, encoding="utf-8",
                          timeout=120)
