"""時間切れ (subprocess.run(timeout=...)) の後に、Windows で待ち続けることがあるかを確かめる (#4 の U5)。

Python 3.10 の Windows の subprocess.run は、時間切れのとき kill() の後に timeout なしの communicate() を待つ。
kill() は子だけを止めるので、孫プロセスが stderr の pipe を持っていると、孫が終わるまで返らない (コードの読み)。

    # (a) 仕組み: 代わりの子プロセスで (Claude CLI は呼ばない)
    python tools/measure/timeout_wait.py stand-in --timeout 3 --sleep 20
    # (b) 本物: キットの proofread._call_claude を、timeout だけ替えて呼ぶ (合成の文だけを送る)
    python tools/measure/timeout_wait.py claude --call-claude --timeout 15 --chars 8000

子孫のプロセスは Toolhelp のスナップショット (ctypes) で見る。記録するのは実行ファイル名だけ (引数は記録しない)。
時間切れの後も残っていた子孫は、grace 秒を過ぎたら、その測りで起きた PID だけを止める
(実行ファイル名と作成時刻が、見たときと同じものだけ。PID の再利用で別のプロセスを止めないため)。
import しただけでは、プロセスの起動も CLI の呼び出しもしない。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kit  # noqa: E402

_kit.use_kit_scripts()


# --- プロセスの一覧 (Windows) ------------------------------------------------

def snapshot() -> list[tuple[int, int, str]]:
    """(pid, 親 pid, 実行ファイル名) の一覧。Windows 以外では空。"""
    if os.name != "nt":
        return []
    import ctypes
    from ctypes import wintypes

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260),
        ]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    k32.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    k32.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    if not h or h == wintypes.HANDLE(-1).value:
        return []
    out = []
    try:
        e = PROCESSENTRY32W()
        e.dwSize = ctypes.sizeof(PROCESSENTRY32W)
        ok = k32.Process32FirstW(h, ctypes.byref(e))
        while ok:
            out.append((int(e.th32ProcessID), int(e.th32ParentProcessID), str(e.szExeFile)))
            ok = k32.Process32NextW(h, ctypes.byref(e))
    finally:
        k32.CloseHandle(h)
    return out


def process_create_time(pid: int) -> int | None:
    """プロセスの作成時刻 (FILETIME、100ns 単位)。取れなければ None。PID の再利用を見分けるために使う。"""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return None
    try:
        times = [wintypes.FILETIME() for _ in range(4)]
        if not k32.GetProcessTimes(wintypes.HANDLE(h), *[ctypes.byref(t) for t in times]):
            return None
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime
    finally:
        k32.CloseHandle(h)


def verified_descendants(procs: list[tuple[int, int, str]], root: int, root_ctime: int | None,
                         ctime_of) -> dict[int, tuple[str, int]]:
    """root の子孫 (root 自身は除く) を {pid: (実行ファイル名, 作成時刻)} で返す。

    Windows の親 PID は、親が終わった後も書き換わらず、PID は再利用される。そこで、親子の組ごとに
    「子の作成時刻 >= 親の作成時刻」を確かめ、親より前に作られたもの (別のプロセスが同じ PID を使っていた頃の子)
    と、作成時刻が取れないものは含めない。root の作成時刻が分からなければ空を返す。
    root が終わってその PID を別のプロセスが使っているときは、そのプロセスより後に作られた子も含めない。
    """
    if root_ctime is None:
        return {}
    children: dict[int, list[tuple[int, str]]] = {}
    for pid, ppid, exe in procs:
        if pid != ppid:
            children.setdefault(ppid, []).append((pid, exe))
    root_now = ctime_of(root)
    root_upper = root_now if (root_now is not None and root_now != root_ctime) else None
    out: dict[int, tuple[str, int]] = {}
    stack = [(root, root_ctime, root_upper)]
    while stack:
        cur, cur_ct, upper = stack.pop()
        for pid, exe in children.get(cur, []):
            if pid in out or pid == root:
                continue
            ct = ctime_of(pid)
            if ct is None or ct < cur_ct or (upper is not None and ct >= upper):
                continue
            out[pid] = (exe, ct)
            stack.append((pid, ct, None))
    return out


def terminate(pid: int) -> bool:
    if os.name != "nt":
        return False
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    h = k32.OpenProcess(0x0001, False, pid)  # PROCESS_TERMINATE
    if not h:
        return False
    try:
        return bool(k32.TerminateProcess(wintypes.HANDLE(h), 1))
    finally:
        k32.CloseHandle(h)


# --- 見張り -----------------------------------------------------------------

class Watch(threading.Thread):
    """子 (root) とその子孫を見張る。deadline を過ぎても呼び出しが返らなければ、残った子孫を止める。

    止めるのは、見張りが子孫として見た PID のうち、実行ファイル名と作成時刻が、見たときと今とで同じものだけ
    (PID が別のプロセスに再利用されていたら止めない。PR #7 のレビュー 1 回目の推奨 1)。
    snap・ctime_of・kill は、テストで差し替えるための口。
    """

    def __init__(self, root_pids: list[int], interval: float = 0.25, *, snap=None, ctime_of=None, kill=None):
        super().__init__(daemon=True)
        self.root_pids = root_pids
        self.root_ctime: int | None = None
        self.interval = interval
        self.snap = snap or snapshot
        self.ctime_of = ctime_of or process_create_time
        self.kill = kill or terminate
        self.seen: dict[int, tuple[str, int]] = {}
        self.max_alive = 0
        self.stop_flag = threading.Event()
        self.deadline: float | None = None
        self.killed: dict[int, tuple[str, int]] = {}
        self.killed_at: float | None = None

    def alive_descendants(self) -> dict[int, tuple[str, int]]:
        if not self.root_pids:
            return {}
        if self.root_ctime is None:
            self.root_ctime = self.ctime_of(self.root_pids[0])
        procs = self.snap()
        alive = {pid: exe for pid, _pp, exe in procs}
        found = verified_descendants(procs, self.root_pids[0], self.root_ctime, self.ctime_of)
        # 親が先に消えた子孫: 見たときと同じ PID・同じ名前・同じ作成時刻で生きているものだけ
        for pid, (exe, ct) in self.seen.items():
            if pid not in found and alive.get(pid) == exe and self.ctime_of(pid) == ct:
                found[pid] = (exe, ct)
        return found

    def kill_verified(self, targets: dict[int, tuple[str, int]]) -> dict[int, tuple[str, int]]:
        """止める直前にもう一度、作成時刻が同じことを確かめてから止める。"""
        done = {}
        for pid, (exe, ct) in targets.items():
            if self.ctime_of(pid) == ct and self.kill(pid):
                done[pid] = (exe, ct)
        return done

    def run(self):
        while not self.stop_flag.is_set():
            d = self.alive_descendants()
            self.seen.update(d)
            self.max_alive = max(self.max_alive, len(d))
            if self.deadline is not None and time.perf_counter() > self.deadline and not self.killed:
                self.killed = self.kill_verified(d)
                self.killed_at = time.perf_counter()
            time.sleep(self.interval)


def _record_popen(root_pids: list[int], on_start=None):
    real = subprocess.Popen

    class RecPopen(real):  # type: ignore[misc, valid-type]
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            if not root_pids:
                root_pids.append(self.pid)
                if on_start is not None:
                    on_start(self.pid)

    return RecPopen


def _names(d: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in d.values():
        exe = v[0] if isinstance(v, tuple) else v
        out[exe] = out.get(exe, 0) + 1
    return out


def timed_call(fn, timeout: float, grace: float) -> dict:
    """fn() (中で subprocess.run(timeout=...) を 1 回呼ぶもの) を測る。"""
    root_pids: list[int] = []
    watch = Watch(root_pids)

    def on_start(pid: int) -> None:
        # 子が起きた直後に作成時刻を取る (子孫の確かめの起点)
        watch.root_ctime = watch.ctime_of(pid)

    with mock.patch.object(subprocess, "Popen", _record_popen(root_pids, on_start)):
        watch.start()
        t0 = time.perf_counter()
        watch.deadline = t0 + timeout + grace
        status = "returned"
        try:
            fn()
        except subprocess.TimeoutExpired:
            status = "TimeoutExpired"
        except Exception as e:  # noqa: BLE001
            status = type(e).__name__
        t1 = time.perf_counter()
    watch.stop_flag.set()
    watch.join(timeout=5)
    left_after = watch.alive_descendants()
    # 測りで起きて残ったものは止める (この測りの子孫で、作成時刻が見たときと同じものだけ)
    cleaned = watch.kill_verified(left_after)
    return {
        "status": status,
        "timeout": timeout,
        "elapsed_sec": round(t1 - t0, 2),
        "wait_after_timeout_sec": round((t1 - t0) - timeout, 2),
        "hung_past_grace": bool(watch.killed),
        "killed_at_grace": _names(watch.killed),
        "descendants_seen": _names(watch.seen),
        "max_descendants_alive": watch.max_alive,
        "left_after_return": _names(left_after),
        "cleaned_after_return": _names(cleaned),
    }


# --- (a) 代わりの子プロセス ----------------------------------------------------

_CHILD = r"""
import subprocess, sys, time
mode, sleep = sys.argv[1], float(sys.argv[2])
code = "import time; time.sleep(%r)" % sleep
if mode == "grandchild-stderr":
    subprocess.Popen([sys.executable, "-c", code], stderr=sys.stderr)
elif mode == "grandchild-default":
    subprocess.Popen([sys.executable, "-c", code])
time.sleep(sleep)
"""


def stand_in(mode: str, timeout: float, sleep: float) -> dict:
    """_call_claude と同じ形 (stdin・stdout はファイル、stderr=PIPE) で、代わりの子を呼ぶ。"""
    with tempfile.TemporaryDirectory() as d:
        fin_p, fout_p = Path(d) / "in.txt", Path(d) / "out.txt"
        fin_p.write_text("x", encoding="utf-8")

        def call():
            with open(fin_p, "r", encoding="utf-8") as fin, open(fout_p, "w", encoding="utf-8") as fout:
                subprocess.run([sys.executable, "-c", _CHILD, mode, str(sleep)], stdin=fin, stdout=fout,
                               stderr=subprocess.PIPE, text=True, encoding="utf-8", env=os.environ.copy(),
                               timeout=timeout)

        row = timed_call(call, timeout, grace=sleep + 30)
    row["mode"] = mode
    row["child_sleep"] = sleep
    return row


# --- (b) 本物の Claude CLI ----------------------------------------------------

def claude(timeout: float, chars: int, seed: int, model: str, grace: float) -> dict:
    import proofread
    import synth

    names = synth.memo_names(seed)
    body = synth.synth_memo_text(chars, seed, names)
    real_run = subprocess.run

    def run_with_timeout(cmd, *a, **k):
        k["timeout"] = timeout
        return real_run(cmd, *a, **k)

    def call():
        # キットの経路そのまま (proofread_body → _call_claude)。替えるのは timeout だけ
        with mock.patch.object(proofread.subprocess, "run", run_with_timeout):
            proofread.proofread_body(body, names, model=model)

    row = timed_call(call, timeout, grace)
    row.update({"mode": "claude", "input_chars": len(body), "model": model})
    return row


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="時間切れの後の待ち (#4 U5)")
    ap.add_argument("--json", type=Path, default=None, help="結果を 1 行の JSON で追記する先")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stand-in", help="(a) 代わりの子プロセス")
    s.add_argument("--timeout", type=float, default=3.0)
    s.add_argument("--sleep", type=float, default=20.0)
    s.add_argument("--modes", default="none,grandchild-default,grandchild-stderr")
    c = sub.add_parser("claude", help="(b) 本物の Claude CLI (合成の文だけ)")
    c.add_argument("--call-claude", action="store_true", help="これを付けたときだけ CLI を呼ぶ")
    c.add_argument("--timeout", type=float, required=True)
    c.add_argument("--chars", type=int, default=8000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--model", default="claude-sonnet-4-6")
    c.add_argument("--grace", type=float, default=120.0)
    args = ap.parse_args(argv)
    _kit.utf8_stdout()

    rows = []
    if args.cmd == "stand-in":
        for mode in [m for m in args.modes.split(",") if m]:
            rows.append(stand_in(mode, args.timeout, args.sleep))
    else:
        if not args.call_claude:
            print("[dry-run] --call-claude を付けたときだけ Claude CLI を呼ぶ")
            return 0
        rows.append(claude(args.timeout, args.chars, args.seed, args.model, args.grace))
    for r in rows:
        line = json.dumps(r, ensure_ascii=False)
        print(line)
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            with args.json.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
