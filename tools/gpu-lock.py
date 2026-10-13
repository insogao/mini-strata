#!/usr/bin/env python3
"""gpu_lock.py - bounded Windows GPU job lock + persistent cleanup fence (task 258, copy of 253 + owner-state fix).

Owner scope: results/phase2_mtp/gpu-lock258/**. Copy of task 253 plus the
single owner liveness fix from the 253 review. No network, no model, no GPU,
no database, no service, no FIFO scheduling, no auto-wake.

Exclusive primitive (unchanged from 246/249)
  msvcrt.locking (Win32 LockFile) on byte 0 of an explicit absolute lock file.
  The OS releases the region when the holding process exits, normally or not.

Cleanup fence (task 253, the only new mechanism)
  Marker file: <lock path>.cleanup-required.json
  - run mode checks the marker right after acquiring the OS lock. If it exists,
    the run is refused (exit 9); the tool never clears the marker by itself.
  - A new marker is written BEFORE any child process may start, and only when
    the Job Object exists. It carries the lock path, job id, owner PID, owner
    process creation identity and reason.
  - The marker is deleted ONLY on the fully clean path: every root exited,
    Job-Object ActiveProcesses confirmed 0, no timeout / nonzero / lifecycle
    error. Anything else keeps the marker as a persistent block for the next
    agent, even though the OS lock is already free.
  - The fence is never auto-expired and never deleted because a PID vanished.
    Explicit release is a separate, manual action (see below) after a human /
    supervisor confirmed the process/lease state.

Release (explicit supervisor action)
  python gpu_lock.py release --lock <abs> --expect-job <marker job id> --by <who>
  Acquires the same OS lock, then deletes the marker only if:
    - a marker exists and its job id equals --expect-job, and
    - the recorded owner is confirmed exited: OpenProcess is opened with
      SYNCHRONIZE|QUERY_LIMITED, then either the creation identity differs
      (PID reused -> old owner gone) or WaitForSingleObject(h, 0) returns
      WAIT_OBJECT_0. WAIT_TIMEOUT (alive) and unknown states refuse; merely
      being able to open the process object is never treated as alive.
    No other command deletes markers.

Direct-binary policy / cleanup confirmation / bounded wait / exit codes
  are inherited from task 249:
  0 ok | 2 usage/validation | 3 busy | 4 not authorized | 5 hash mismatch
  6 spawn failure | 7 command timed out or exited nonzero | 8 lifecycle or
  cleanup NO-GO | 9 cleanup fence active (run refused, marker kept)

Usage example (10 lines):
  # acquire-only
  python gpu_lock.py hold --lock D:\\abs\\gpu.lock --hold-seconds 0 --wait-seconds 30
  # validated direct-binary launch (single command, argv[0] == binary, sha checked)
  python gpu_lock.py run --lock D:\\abs\\gpu.lock --job D:\\abs\\job.json \\
      --result D:\\abs\\result.json --wait-seconds 60
  # explicit fence release after supervisor confirmation
  python gpu_lock.py release --lock D:\\abs\\gpu.lock --expect-job <id> --by supervisor
  # job.json: authorized=true, job, task, binary(abs)+binary_sha256,
  # commands=[[binary,...]], max_launch=1, runtime_limit_seconds
"""

import argparse
import ctypes
import hashlib
import json
import msvcrt
import os
import subprocess
import sys
import time
from ctypes import wintypes

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_BUSY = 3
EXIT_NOT_AUTHORIZED = 4
EXIT_HASH = 5
EXIT_SPAWN = 6
EXIT_RUNTIME = 7
EXIT_LIFECYCLE = 8
EXIT_FENCE = 9

MAX_WAIT_SECONDS = 60.0
DEFAULT_RETRY_SECONDS = 10.0
MIN_RETRY_SECONDS = 0.2
MAX_RETRY_SECONDS = 30.0
MIN_CLEANUP_SECONDS = 0.5
MAX_CLEANUP_SECONDS = 30.0

CLEANUP_MARKER_SUFFIX = ".cleanup-required.json"

WAIT_TIMEOUT = 0x00000102

CREATE_SUSPENDED = 0x00000004
CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000
STARTF_USESTDHANDLES = 0x00000100

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
FILE_SHARE_READ = 0x00000001
FILE_SHARE_WRITE = 0x00000002
OPEN_EXISTING = 3
CREATE_ALWAYS = 2
FILE_ATTRIBUTE_NORMAL = 0x00000080
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

SYNCHRONIZE = 0x00100000
ERROR_INVALID_PARAMETER = 87

JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION = 1

DENY_BASENAMES = {
    "python.exe", "pythonw.exe", "py.exe",
    "cmd.exe", "powershell.exe", "pwsh.exe",
    "wscript.exe", "cscript.exe", "mshta.exe",
    "bash.exe", "sh.exe", "wsl.exe",
    "node.exe", "java.exe",
}

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
        ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class JOBOBJECT_BASIC_ACCOUNTING_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("TotalUserTime", wintypes.LARGE_INTEGER),
        ("TotalKernelTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalUserTime", wintypes.LARGE_INTEGER),
        ("ThisPeriodTotalKernelTime", wintypes.LARGE_INTEGER),
        ("TotalPageFaultCount", wintypes.DWORD),
        ("TotalProcesses", wintypes.DWORD),
        ("ActiveProcesses", wintypes.DWORD),
        ("TotalTerminatedProcesses", wintypes.DWORD),
    ]


kernel32.CreateJobObjectW.restype = wintypes.HANDLE
kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
kernel32.SetInformationJobObject.restype = wintypes.BOOL
kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                             ctypes.c_void_p, wintypes.DWORD]
kernel32.QueryInformationJobObject.restype = wintypes.BOOL
kernel32.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD,
                                               ctypes.c_void_p]
kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
kernel32.TerminateJobObject.restype = wintypes.BOOL
kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel32.TerminateProcess.restype = wintypes.BOOL
kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
kernel32.CreateProcessW.restype = wintypes.BOOL
kernel32.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR,
                                    ctypes.c_void_p, ctypes.c_void_p,
                                    wintypes.BOOL, wintypes.DWORD,
                                    ctypes.c_void_p, wintypes.LPCWSTR,
                                    ctypes.POINTER(STARTUPINFOW),
                                    ctypes.POINTER(PROCESS_INFORMATION)]
kernel32.ResumeThread.restype = wintypes.DWORD
kernel32.ResumeThread.argtypes = [wintypes.HANDLE]
kernel32.WaitForSingleObject.restype = wintypes.DWORD
kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
kernel32.GetExitCodeProcess.restype = wintypes.BOOL
kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE,
                                        ctypes.POINTER(wintypes.DWORD)]
kernel32.CloseHandle.restype = wintypes.BOOL
kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
kernel32.CreateFileW.restype = wintypes.HANDLE
kernel32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
                                 wintypes.DWORD,
                                 ctypes.POINTER(SECURITY_ATTRIBUTES),
                                 wintypes.DWORD, wintypes.DWORD,
                                 wintypes.HANDLE]
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL,
                                 wintypes.DWORD]
kernel32.GetProcessTimes.restype = wintypes.BOOL
kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE,
                                     ctypes.POINTER(wintypes.FILETIME),
                                     ctypes.POINTER(wintypes.FILETIME),
                                     ctypes.POINTER(wintypes.FILETIME),
                                     ctypes.POINTER(wintypes.FILETIME)]


def now_iso():
    t = time.time()
    ms = int((t - int(t)) * 1000)
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t)) + ".%03d" % ms


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def process_creation_filetime(pid):
    h = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return None
    try:
        created = wintypes.FILETIME()
        exited = wintypes.FILETIME()
        kernel_t = wintypes.FILETIME()
        user_t = wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(h, ctypes.byref(created),
                                      ctypes.byref(exited),
                                      ctypes.byref(kernel_t),
                                      ctypes.byref(user_t))
        if not ok:
            return None
        return (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel32.CloseHandle(h)


def owner_process_state(pid, creation_filetime):
    """Classify the recorded owner: 'alive', 'exited' or 'unknown'.

    A process object that merely can be opened is not proof of life: after
    OpenProcess with SYNCHRONIZE|QUERY_LIMITED, WaitForSingleObject(h, 0)
    decides. The creation identity guards against PID reuse. 'unknown' must
    be treated as still running (refuse the release).
    """
    h = kernel32.OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                             False, pid)
    if not h:
        if ctypes.get_last_error() == ERROR_INVALID_PARAMETER:
            return "exited"
        return "unknown"
    try:
        created = wintypes.FILETIME()
        exited_t = wintypes.FILETIME()
        kernel_t = wintypes.FILETIME()
        user_t = wintypes.FILETIME()
        ok = kernel32.GetProcessTimes(h, ctypes.byref(created),
                                      ctypes.byref(exited_t),
                                      ctypes.byref(kernel_t),
                                      ctypes.byref(user_t))
        if not ok:
            return "unknown"
        cur = (created.dwHighDateTime << 32) | created.dwLowDateTime
        if creation_filetime is not None and cur != creation_filetime:
            return "exited"
        rc = kernel32.WaitForSingleObject(h, 0)
        if rc == 0:
            return "exited"
        if rc == WAIT_TIMEOUT:
            return "alive"
        return "unknown"
    finally:
        kernel32.CloseHandle(h)


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)
    os.replace(tmp, path)


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def marker_path_for(lock_path):
    return os.path.abspath(lock_path) + CLEANUP_MARKER_SUFFIX


def marker_exists(lock_path):
    return os.path.isfile(marker_path_for(lock_path))


def write_cleanup_marker(lock_path, job_id, task):
    path = marker_path_for(lock_path)
    write_json(path, {
        "state": "cleanup_required",
        "lock_path": os.path.abspath(lock_path),
        "job": job_id,
        "task": task,
        "owner_pid": os.getpid(),
        "owner_creation_filetime": process_creation_filetime(os.getpid()),
        "created_at": now_iso(),
        "reason": "launch_pending",
        "outcome": None,
        "note": "persistent fence; only explicit `release` after supervisor confirmation removes it",
    })
    return path


def update_marker_outcome(lock_path, outcome):
    path = marker_path_for(lock_path)
    data = read_json(path)
    if not isinstance(data, dict):
        return
    data["outcome"] = outcome
    data["updated_at"] = now_iso()
    write_json(path, data)


def delete_cleanup_marker(lock_path):
    path = marker_path_for(lock_path)
    if os.path.isfile(path):
        os.remove(path)
    return path


class FileRegionLock:
    """Win32 file region lock on byte 0. OS released on process exit."""

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.fd = None

    def try_acquire(self):
        parent = os.path.dirname(self.path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o666)
        try:
            if os.fstat(fd).st_size < 1:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            self.fd = fd
            return True, None
        except OSError as e:
            os.close(fd)
            return False, e

    def release(self):
        if self.fd is None:
            return
        try:
            os.lseek(self.fd, 0, os.SEEK_SET)
            msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        os.close(self.fd)
        self.fd = None


def acquire_with_wait(lock, wait_seconds, retry_interval):
    start = time.monotonic()
    deadline = start + wait_seconds
    attempts = 0
    while True:
        attempts += 1
        ok, err = lock.try_acquire()
        if ok:
            return {"attempts": attempts,
                    "wait_elapsed_seconds": round(time.monotonic() - start, 3)}
        now = time.monotonic()
        if now >= deadline:
            return None
        sleep_s = min(retry_interval, max(0.0, deadline - now))
        if sleep_s > 0:
            time.sleep(sleep_s)


class JobObject:
    def __init__(self):
        self.handle = kernel32.CreateJobObjectW(None, None)
        if not self.handle:
            raise OSError("CreateJobObjectW failed err=%d" %
                          ctypes.get_last_error())
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        ok = kernel32.SetInformationJobObject(
            self.handle, JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info))
        if not ok:
            err = ctypes.get_last_error()
            kernel32.CloseHandle(self.handle)
            self.handle = None
            raise OSError("SetInformationJobObject failed err=%d" % err)

    def assign(self, hprocess):
        if not kernel32.AssignProcessToJobObject(self.handle, hprocess):
            raise OSError("AssignProcessToJobObject failed err=%d" %
                          ctypes.get_last_error())

    def active_processes(self):
        info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION()
        ok = kernel32.QueryInformationJobObject(
            self.handle, JOB_OBJECT_BASIC_ACCOUNTING_INFORMATION,
            ctypes.byref(info), ctypes.sizeof(info), None)
        if not ok:
            raise OSError("QueryInformationJobObject failed err=%d" %
                          ctypes.get_last_error())
        return int(info.ActiveProcesses)

    def terminate(self, code=1):
        if self.handle:
            kernel32.TerminateJobObject(self.handle, code)

    def close(self):
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None


def confirm_cleanup(job, timeout_seconds):
    """Bounded poll until ActiveProcesses == 0. Returns (confirmed, seconds)."""
    start = time.monotonic()
    while True:
        n = job.active_processes()
        if n == 0:
            return True, round(time.monotonic() - start, 3)
        if time.monotonic() - start >= timeout_seconds:
            return False, round(time.monotonic() - start, 3)
        time.sleep(0.1)


def _open_inheritable(path, access, creation):
    sa = SECURITY_ATTRIBUTES()
    sa.nLength = ctypes.sizeof(sa)
    sa.lpSecurityDescriptor = None
    sa.bInheritHandle = True
    h = kernel32.CreateFileW(path, access, FILE_SHARE_READ | FILE_SHARE_WRITE,
                             ctypes.byref(sa), creation,
                             FILE_ATTRIBUTE_NORMAL, None)
    if h is None or h == INVALID_HANDLE_VALUE:
        raise OSError("CreateFileW failed for %s err=%d" %
                      (path, ctypes.get_last_error()))
    return h


def spawn_suspended(argv, cwd, log_path):
    cmdline = subprocess.list2cmdline([str(a) for a in argv])
    nul = _open_inheritable("NUL", GENERIC_READ, OPEN_EXISTING)
    logf = _open_inheritable(log_path, GENERIC_WRITE, CREATE_ALWAYS)
    si = STARTUPINFOW()
    si.cb = ctypes.sizeof(si)
    si.dwFlags = STARTF_USESTDHANDLES
    si.hStdInput = nul
    si.hStdOutput = logf
    si.hStdError = logf
    pi = PROCESS_INFORMATION()
    ok = kernel32.CreateProcessW(
        None, ctypes.create_unicode_buffer(cmdline), None, None, True,
        CREATE_SUSPENDED | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
        None, os.path.abspath(cwd) if cwd else None,
        ctypes.byref(si), ctypes.byref(pi))
    kernel32.CloseHandle(nul)
    kernel32.CloseHandle(logf)
    if not ok:
        raise OSError("CreateProcessW failed err=%d" % ctypes.get_last_error())
    return pi


def run_one_command(argv, job, cwd, runtime_limit, log_path):
    rec = {"argv": [str(a) for a in argv], "started_at": now_iso(),
           "timed_out": False, "lifecycle_error": None, "exited": False}
    t0 = time.monotonic()
    pi = spawn_suspended(argv, cwd, log_path)
    rec["pid"] = pi.dwProcessId
    try:
        job.assign(pi.hProcess)
    except OSError as e:
        kernel32.TerminateProcess(pi.hProcess, 1)
        kernel32.WaitForSingleObject(pi.hProcess, 5000)
        kernel32.CloseHandle(pi.hThread)
        kernel32.CloseHandle(pi.hProcess)
        rec["lifecycle_error"] = str(e)
        rec["elapsed_seconds"] = round(time.monotonic() - t0, 3)
        rec["finished_at"] = now_iso()
        return rec
    kernel32.ResumeThread(pi.hThread)
    kernel32.CloseHandle(pi.hThread)
    ms = int(max(1.0, runtime_limit) * 1000)
    wait_rc = kernel32.WaitForSingleObject(pi.hProcess, ms)
    if wait_rc == WAIT_TIMEOUT:
        rec["timed_out"] = True
        job.terminate(1)
        wait_rc = kernel32.WaitForSingleObject(pi.hProcess, 10000)
    rec["exited"] = (wait_rc == 0)
    code = wintypes.DWORD(0)
    kernel32.GetExitCodeProcess(pi.hProcess, ctypes.byref(code))
    rec["exit_code"] = code.value
    rec["elapsed_seconds"] = round(time.monotonic() - t0, 3)
    rec["finished_at"] = now_iso()
    kernel32.CloseHandle(pi.hProcess)
    return rec


ALLOWED_JOB_KEYS = {"authorized", "job", "task", "source", "binary",
                    "binary_sha256", "commands", "max_launch",
                    "runtime_limit_seconds", "note"}


def _norm(path):
    return os.path.normcase(os.path.abspath(path))


def validate_job(data, expect_job=None):
    errs = []
    if not isinstance(data, dict):
        return ["job file must be a JSON object"]
    unknown = sorted(set(data.keys()) - ALLOWED_JOB_KEYS)
    if unknown:
        errs.append("unknown fields: %s" % ", ".join(unknown))
    if data.get("authorized") is not True:
        errs.append("missing explicit authorized=true")
    job_id = data.get("job")
    if not isinstance(job_id, str) or not job_id:
        errs.append("missing job id")
    if expect_job is not None and job_id != expect_job:
        errs.append("job id %r does not match expected %r" % (job_id, expect_job))
    if data.get("task") is None:
        errs.append("missing task")

    cmds = data.get("commands")
    if not isinstance(cmds, list) or not cmds:
        errs.append("commands must be a non-empty list")
    elif len(cmds) != 1:
        errs.append("exactly one direct binary command is allowed (got %d)" % len(cmds))
    else:
        for i, c in enumerate(cmds):
            if not isinstance(c, list) or not c or not all(isinstance(x, str) and x for x in c):
                errs.append("command %d must be an argv list of non-empty strings" % i)

    max_launch = data.get("max_launch")
    if not isinstance(max_launch, int) or isinstance(max_launch, bool) or not (1 <= max_launch <= 16):
        errs.append("max_launch must be an integer in [1, 16]")
    elif isinstance(cmds, list) and len(cmds) > max_launch:
        errs.append("commands count %d exceeds max_launch %d" % (len(cmds), max_launch))

    rt = data.get("runtime_limit_seconds")
    if not isinstance(rt, (int, float)) or isinstance(rt, bool) or not (0 < rt <= 3600):
        errs.append("runtime_limit_seconds must be a number in (0, 3600]")

    binary = data.get("binary")
    bsha = data.get("binary_sha256")
    binary_ok = isinstance(binary, str) and bool(binary)
    if not binary_ok:
        errs.append("binary is required (absolute path of the frozen binary)")
    else:
        if not os.path.isabs(binary):
            errs.append("binary must be an absolute path")
        if not os.path.isfile(binary):
            errs.append("binary not found: %s" % binary)
        base = os.path.basename(binary).lower()
        if base in DENY_BASENAMES:
            errs.append("binary basename %s is a shell/interpreter/script host - rejected" % base)
    if not isinstance(bsha, str) or len(bsha) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in bsha):
        errs.append("binary_sha256 is missing or not a sha256 hex")

    if binary_ok and isinstance(cmds, list) and len(cmds) == 1:
        c0 = cmds[0]
        if isinstance(c0, list) and c0 and isinstance(c0[0], str):
            a0 = c0[0]
            if not os.path.isabs(a0):
                errs.append("argv[0] must be an absolute path")
            elif _norm(a0) != _norm(binary):
                errs.append("argv[0] must equal the declared binary (no wrappers)")
    return errs


def check_wait_args(wait_seconds, retry_interval):
    errs = []
    if not (0.0 <= wait_seconds <= MAX_WAIT_SECONDS):
        errs.append("--wait-seconds must be within [0, %.0f]" % MAX_WAIT_SECONDS)
    if not (MIN_RETRY_SECONDS <= retry_interval <= MAX_RETRY_SECONDS):
        errs.append("--retry-interval-seconds must be within [%.1f, %.1f]" % (MIN_RETRY_SECONDS, MAX_RETRY_SECONDS))
    return errs


def cmd_hold(args):
    res = {"mode": "hold", "started_at": now_iso(), "argv": sys.argv[1:],
           "lock_path": os.path.abspath(args.lock),
           "task": args.task, "source": args.source,
           "wait_seconds": args.wait_seconds,
           "retry_interval_seconds": args.retry_interval_seconds,
           "hold_seconds": args.hold_seconds}
    errs = []
    if not os.path.isabs(args.lock):
        errs.append("--lock must be an absolute path")
    errs += check_wait_args(args.wait_seconds, args.retry_interval_seconds)
    if errs:
        res["errors"] = errs
        res["exit_code"] = EXIT_USAGE
        if args.result:
            write_json(args.result, res)
        print("REJECTED: %s" % "; ".join(errs))
        return EXIT_USAGE

    lock = FileRegionLock(args.lock)
    acquired = False
    try:
        a = acquire_with_wait(lock, args.wait_seconds, args.retry_interval_seconds)
        if a is None:
            res["status"] = "busy"
            res["exit_code"] = EXIT_BUSY
            print("BUSY: lock held, deadline reached")
            return EXIT_BUSY
        acquired = True
        res["acquired_at"] = now_iso()
        res["wait"] = a
        if args.audit:
            write_json(args.audit, {
                "owner_pid": os.getpid(),
                "owner_creation_filetime": process_creation_filetime(os.getpid()),
                "task": args.task,
                "source": args.source,
                "lock_path": res["lock_path"],
                "acquired_at": res["acquired_at"],
                "mode": "hold",
                "budget": {"wait_seconds": args.wait_seconds,
                           "retry_interval_seconds": args.retry_interval_seconds},
            })
        if args.ready_file:
            with open(args.ready_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "at": now_iso()}))
        if args.hold_seconds > 0:
            time.sleep(args.hold_seconds)
        return EXIT_OK
    finally:
        if acquired:
            lock.release()
            res["released_at"] = now_iso()
        res.setdefault("exit_code", EXIT_OK)
        if args.result:
            write_json(args.result, res)


def cmd_release(args):
    """Explicit supervisor release of a cleanup fence. Never called by run()."""
    res = {"mode": "release", "started_at": now_iso(), "argv": sys.argv[1:],
           "lock_path": os.path.abspath(args.lock),
           "marker_path": marker_path_for(args.lock),
           "expect_job": args.expect_job, "by": args.by}
    errs = []
    if not os.path.isabs(args.lock):
        errs.append("--lock must be an absolute path")
    if not args.expect_job:
        errs.append("--expect-job is required")
    errs += check_wait_args(args.wait_seconds, args.retry_interval_seconds)
    if errs:
        res["errors"] = errs
        res["exit_code"] = EXIT_USAGE
        if args.result:
            write_json(args.result, res)
        print("REJECTED: %s" % "; ".join(errs))
        return EXIT_USAGE

    lock = FileRegionLock(args.lock)
    acquired = False
    try:
        a = acquire_with_wait(lock, args.wait_seconds, args.retry_interval_seconds)
        if a is None:
            res["status"] = "busy"
            res["exit_code"] = EXIT_BUSY
            print("BUSY: lock held, deadline reached")
            return EXIT_BUSY
        acquired = True
        res["wait"] = a
        res["acquired_at"] = now_iso()

        marker = read_json(res["marker_path"])
        if not isinstance(marker, dict):
            res["released"] = False
            res["reason"] = "no_marker"
            print("NO MARKER: nothing to release")
            return EXIT_OK
        res["marker"] = marker
        if marker.get("job") != args.expect_job:
            res["released"] = False
            res["reason"] = "job_mismatch"
            res["exit_code"] = EXIT_USAGE
            print("REFUSED: marker job %r != expected %r" % (marker.get("job"), args.expect_job))
            return EXIT_USAGE
        pid = marker.get("owner_pid")
        ct = marker.get("owner_creation_filetime")
        state = "exited"
        if pid is not None:
            state = owner_process_state(int(pid), ct)
            res["owner_state"] = state
            if state != "exited":
                res["released"] = False
                res["reason"] = "owner_still_running" if state == "alive" else "owner_state_unknown"
                res["owner_pid"] = pid
                res["exit_code"] = EXIT_USAGE
                print("REFUSED: marker owner pid %s state=%s" % (pid, state))
                return EXIT_USAGE
        delete_cleanup_marker(args.lock)
        res["released"] = True
        res["reason"] = "explicit_release"
        res["released_at"] = now_iso()
        print("RELEASED: cleanup fence removed by explicit action")
        return EXIT_OK
    finally:
        if acquired:
            lock.release()
        res.setdefault("exit_code", EXIT_OK)
        if args.result:
            write_json(args.result, res)


def cmd_run(args):
    res = {"mode": "run", "started_at": now_iso(), "argv": sys.argv[1:],
           "lock_path": os.path.abspath(args.lock),
           "marker_path": marker_path_for(args.lock),
           "wait_seconds": args.wait_seconds,
           "retry_interval_seconds": args.retry_interval_seconds,
           "cleanup_wait_seconds": args.cleanup_wait_seconds,
           "commands": [],
           "marker": {"path": marker_path_for(args.lock), "written_at": None,
                      "removed_at": None, "retained": False, "outcome": None},
           "child_cleanup": {"all_exited": False, "roots_exited": False,
                             "active_processes_zero": False,
                             "confirm_seconds": None,
                             "confirm_after_terminate_seconds": None,
                             "no_go": False, "processes": []}}
    rc = _run_body(args, res)
    res["exit_code"] = rc
    if args.result:
        write_json(args.result, res)
    return rc


def _run_body(args, res):
    lock = FileRegionLock(args.lock)
    acquired = False
    job = None
    try:
        errs = []
        if not os.path.isabs(args.lock):
            errs.append("--lock must be an absolute path")
        errs += check_wait_args(args.wait_seconds, args.retry_interval_seconds)
        if not (MIN_CLEANUP_SECONDS <= args.cleanup_wait_seconds <= MAX_CLEANUP_SECONDS):
            errs.append("--cleanup-wait-seconds must be within [%.1f, %.1f]" % (MIN_CLEANUP_SECONDS, MAX_CLEANUP_SECONDS))
        if not os.path.isfile(args.job):
            errs.append("job file not found: %s" % args.job)
        if errs:
            res["errors"] = errs
            print("REJECTED: %s" % "; ".join(errs))
            return EXIT_USAGE

        try:
            with open(args.job, "r", encoding="utf-8") as f:
                job_data = json.load(f)
        except Exception as e:
            res["errors"] = ["job file parse failed: %s" % e]
            print("REJECTED: job file parse failed")
            return EXIT_USAGE

        res["job"] = job_data.get("job") if isinstance(job_data, dict) else None
        res["task"] = job_data.get("task") if isinstance(job_data, dict) else None
        verrs = validate_job(job_data, args.expect_job)
        if verrs:
            res["errors"] = verrs
            print("NOT AUTHORIZED: %s" % "; ".join(verrs))
            return EXIT_NOT_AUTHORIZED

        job_sha_before = sha256_file(args.job)

        a = acquire_with_wait(lock, args.wait_seconds, args.retry_interval_seconds)
        if a is None:
            res["status"] = "busy"
            print("BUSY: lock held, deadline reached")
            return EXIT_BUSY
        acquired = True
        res["acquired_at"] = now_iso()
        res["wait"] = a

        if marker_exists(args.lock):
            res["status"] = "cleanup_fence_active"
            res["marker"]["retained"] = True
            res["fence"] = read_json(res["marker_path"])
            print("REFUSED: cleanup-required marker present; explicit supervisor release needed")
            return EXIT_FENCE

        if args.audit:
            write_json(args.audit, {
                "owner_pid": os.getpid(),
                "owner_creation_filetime": process_creation_filetime(os.getpid()),
                "task": job_data.get("task"),
                "source": job_data.get("source", args.source),
                "job": job_data.get("job"),
                "binary": job_data.get("binary"),
                "binary_sha256_declared": job_data.get("binary_sha256"),
                "lock_path": res["lock_path"],
                "acquired_at": res["acquired_at"],
                "mode": "run",
                "budget": {"wait_seconds": args.wait_seconds,
                           "retry_interval_seconds": args.retry_interval_seconds,
                           "runtime_limit_seconds": job_data.get("runtime_limit_seconds"),
                           "max_launch": job_data.get("max_launch"),
                           "cleanup_wait_seconds": args.cleanup_wait_seconds},
            })

        if sha256_file(args.job) != job_sha_before:
            res["errors"] = ["job file changed after lock acquisition"]
            return EXIT_HASH
        binary = job_data.get("binary")
        actual = sha256_file(binary)
        res["binary_sha256_actual"] = actual
        if actual != job_data["binary_sha256"].lower():
            res["errors"] = ["binary sha256 mismatch after lock acquisition"]
            return EXIT_HASH

        try:
            job = JobObject()
        except OSError as e:
            res["status"] = "lifecycle_unavailable"
            res["errors"] = [str(e)]
            print("LIFECYCLE UNAVAILABLE: launcher NO-GO, use acquire-only mode")
            return EXIT_LIFECYCLE

        write_cleanup_marker(args.lock, job_data.get("job"), job_data.get("task"))
        res["marker"]["written_at"] = now_iso()

        if args.ready_file:
            with open(args.ready_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"pid": os.getpid(), "at": now_iso()}))

        child_log_dir = args.child_log_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.result)) if args.result else os.getcwd(),
            "child_logs")
        os.makedirs(child_log_dir, exist_ok=True)

        rt = float(job_data["runtime_limit_seconds"])
        stop = False
        rc = EXIT_OK
        for i, argv in enumerate(job_data["commands"]):
            if stop:
                res["commands"].append({"argv": argv, "skipped": True})
                continue
            logp = os.path.join(child_log_dir, "child_%02d.log" % i)
            try:
                rec = run_one_command(argv, job, args.cwd or None, rt, logp)
            except OSError as e:
                rec = {"argv": [str(x) for x in argv], "spawn_error": str(e),
                       "exited": False, "timed_out": False,
                       "lifecycle_error": None, "pid": None}
            rec["log_path"] = logp
            res["commands"].append(rec)
            res["child_cleanup"]["processes"].append(rec.get("pid"))
            if rec.get("spawn_error"):
                stop = True
                rc = EXIT_SPAWN
                continue
            if rec.get("lifecycle_error"):
                stop = True
                rc = EXIT_LIFECYCLE
                continue
            if not rec.get("exited"):
                stop = True
                rc = EXIT_RUNTIME
                continue
            if rec["timed_out"] or rec.get("exit_code", 1) != 0:
                stop = True
                rc = EXIT_RUNTIME

        roots_exited = True
        for rec in res["commands"]:
            if rec.get("skipped"):
                continue
            if (not rec.get("exited") or rec.get("lifecycle_error")
                    or rec.get("spawn_error")):
                roots_exited = False
        res["child_cleanup"]["roots_exited"] = roots_exited

        confirmed, secs = confirm_cleanup(job, args.cleanup_wait_seconds)
        res["child_cleanup"]["confirm_seconds"] = secs
        if not confirmed:
            print("CLEANUP NOT CONFIRMED: ActiveProcesses != 0, terminating job")
            job.terminate(1)
            confirmed2, secs2 = confirm_cleanup(job, args.cleanup_wait_seconds)
            res["child_cleanup"]["confirm_after_terminate_seconds"] = secs2
            confirmed = confirmed2
        res["child_cleanup"]["active_processes_zero"] = bool(confirmed)
        res["child_cleanup"]["all_exited"] = bool(roots_exited and confirmed)
        if not confirmed:
            res["child_cleanup"]["no_go"] = True
            res["status"] = "cleanup_unconfirmed_no_go"
            print("NO_GO: job ActiveProcesses did not reach 0")
            rc = EXIT_LIFECYCLE

        command_trouble = False
        for rec in res["commands"]:
            if rec.get("skipped"):
                continue
            if rec.get("timed_out") or rec.get("exit_code", 1) != 0 \
                    or rec.get("spawn_error") or rec.get("lifecycle_error"):
                command_trouble = True
        cleanup_clean = (rc == EXIT_OK and roots_exited and confirmed
                         and not command_trouble)
        if cleanup_clean:
            delete_cleanup_marker(args.lock)
            res["marker"]["removed_at"] = now_iso()
            res["marker"]["retained"] = False
        else:
            outcome = "cleanup_unconfirmed" if not confirmed else (
                "command_timeout" if any(r.get("timed_out") for r in res["commands"]) else
                "command_failed" if command_trouble else "incomplete")
            res["marker"]["retained"] = True
            res["marker"]["outcome"] = outcome
            update_marker_outcome(args.lock, outcome)
        return rc
    finally:
        if job is not None:
            res["job_closed_at"] = now_iso()
            job.close()
        if acquired:
            lock.release()
            res["released_at"] = now_iso()


def build_parser():
    p = argparse.ArgumentParser(
        prog="gpu_lock.py",
        description="bounded Windows GPU job lock + cleanup fence (task 253)")
    sub = p.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("hold", help="acquire-only: hold the lock for N seconds")
    h.add_argument("--lock", required=True, help="absolute lock file path")
    h.add_argument("--audit", default=None)
    h.add_argument("--result", default=None)
    h.add_argument("--ready-file", default=None)
    h.add_argument("--hold-seconds", type=float, default=1.0)
    h.add_argument("--wait-seconds", type=float, default=60.0)
    h.add_argument("--retry-interval-seconds", type=float, default=DEFAULT_RETRY_SECONDS)
    h.add_argument("--task", default=None)
    h.add_argument("--source", default=None)

    r = sub.add_parser("run", help="validated direct-binary launcher under the lock + fence")
    r.add_argument("--lock", required=True, help="absolute lock file path")
    r.add_argument("--job", required=True, help="authorized job JSON path")
    r.add_argument("--audit", default=None)
    r.add_argument("--result", default=None)
    r.add_argument("--ready-file", default=None)
    r.add_argument("--child-log-dir", default=None)
    r.add_argument("--cwd", default=None)
    r.add_argument("--wait-seconds", type=float, default=60.0)
    r.add_argument("--retry-interval-seconds", type=float, default=DEFAULT_RETRY_SECONDS)
    r.add_argument("--cleanup-wait-seconds", type=float, default=5.0)
    r.add_argument("--expect-job", default=None)
    r.add_argument("--source", default=None)

    rel = sub.add_parser("release", help="explicit supervisor release of the cleanup fence")
    rel.add_argument("--lock", required=True, help="absolute lock file path")
    rel.add_argument("--expect-job", required=True, help="marker job id to confirm")
    rel.add_argument("--by", default=None, help="operator identity note")
    rel.add_argument("--wait-seconds", type=float, default=5.0)
    rel.add_argument("--retry-interval-seconds", type=float, default=DEFAULT_RETRY_SECONDS)
    rel.add_argument("--result", default=None)
    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "hold":
        rc = cmd_hold(args)
    elif args.cmd == "release":
        rc = cmd_release(args)
    else:
        rc = cmd_run(args)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
