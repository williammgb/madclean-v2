"""Run one command so that every process it starts ends with it.

The smoke gate starts madclean-ui through this. Reflex starts a tree of processes
(reflex, the backend, its worker processes, node); stopping the top one used to leave
the rest running, holding ports and files. On Windows this script puts itself in a
job object that ends every process in it when the script ends, for any reason.

It ends when the stop file appears, when the command exits, or after --max-seconds,
so a gate whose caller died cannot leave the app running.

usage: python scripts/run_in_job.py --stop-file F --max-seconds N -- command [args]
"""

import argparse
import ctypes
import os
import subprocess
import sys
import time
from ctypes import wintypes


def _kill_tree_on_exit() -> None:
    """Put this process in a job whose processes all end when the job's handle closes."""

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount")]

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimits),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kill_on_job_close = 0x2000
    extended_limit_information = 9
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]
    kernel32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
    limits = ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = kill_on_job_close
    if not kernel32.SetInformationJobObject(job, extended_limit_information, ctypes.byref(limits), ctypes.sizeof(limits)):
        raise OSError(ctypes.get_last_error(), "SetInformationJobObject failed")
    if not kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()):
        raise OSError(ctypes.get_last_error(), "AssignProcessToJobObject failed")
    # The handle stays open for this process's life; Windows closes it when the
    # process ends, which ends every process in the job.


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-file", required=True)
    parser.add_argument("--max-seconds", type=float, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given")

    if sys.platform == "win32":
        _kill_tree_on_exit()
        child = subprocess.Popen(command)
    else:
        child = subprocess.Popen(command, start_new_session=True)

    deadline = time.monotonic() + args.max_seconds
    while child.poll() is None:
        if os.path.exists(args.stop_file) or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    if child.poll() is None and sys.platform != "win32":
        os.killpg(child.pid, 9)
    # On Windows, returning ends this process and the job ends everything it started.
    return child.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
