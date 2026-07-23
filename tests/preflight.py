#!/usr/bin/env python3
"""Benchmark preflight: verify no stray llama processes, no running Docker
containers, and a clean GPU baseline before starting a measured run.

Report-only by default; --kill terminates stray llama/bench/install processes
and stops ALL running Docker containers (the historical results in
docs/RESULTS.md were only reproducible after doing exactly that). Exits 0 when
clean, 1 when anything is still in the way, so scripts can gate on it:

    python3 tests/preflight.py --kill && python3 tests/benchmark.py ...

Linux-only, stdlib-only. GPU baseline uses nvidia-smi when present; the
default 300 MiB allowance covers persistent desktop/remote-desktop processes
that always hold a little VRAM.
"""
import argparse
import re
import subprocess
import sys

STRAY_PATTERNS = ("llama-server", "llama-bench", "llama-cli", "install.py")


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def stray_processes():
    out = _run(["pgrep", "-af", "|".join(STRAY_PATTERNS)])
    me = str(__import__("os").getpid())
    procs = []
    for line in out.splitlines():
        pid, _, cmd = line.partition(" ")
        if pid != me and "preflight" not in cmd:
            procs.append((pid, cmd.strip()))
    return procs


def running_containers():
    out = _run(["docker", "ps", "--format", "{{.ID}} {{.Names}} {{.Image}}"])
    return [l for l in out.splitlines() if l.strip()]


def gpu_used_mib():
    out = _run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    m = re.search(r"\d+", out or "")
    return int(m.group()) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kill", action="store_true",
                    help="terminate stray processes and stop all running containers")
    ap.add_argument("--max-gpu-mib", type=int, default=300,
                    help="GPU baseline allowance in MiB (default 300)")
    args = ap.parse_args()

    clean = True

    procs = stray_processes()
    if procs and args.kill:
        for pid, cmd in procs:
            print(f"killing stray pid {pid}: {cmd[:90]}")
            _run(["kill", pid])
        import time
        time.sleep(3)
        procs = stray_processes()
    for pid, cmd in procs:
        print(f"STRAY process pid {pid}: {cmd[:110]}")
        clean = False

    containers = running_containers()
    if containers and args.kill:
        for line in containers:
            cid = line.split()[0]
            print(f"stopping container {line}")
            _run(["docker", "stop", cid])
        containers = running_containers()
    for line in containers:
        print(f"RUNNING container: {line}")
        clean = False

    used = gpu_used_mib()
    if used is None:
        print("gpu: nvidia-smi unavailable, skipping baseline check")
    elif used > args.max_gpu_mib:
        print(f"GPU baseline {used} MiB exceeds allowance {args.max_gpu_mib} MiB")
        clean = False
    else:
        print(f"gpu baseline ok: {used} MiB")

    print("preflight:", "CLEAN" if clean else "NOT CLEAN")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
