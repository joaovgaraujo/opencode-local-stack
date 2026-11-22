#!/usr/bin/env python3
"""opencode-local: start the local model server (if needed) and open OpenCode
in the current folder. Installed on PATH by install.py / setup.cmd.

    opencode-local                      server + OpenCode here
    opencode-local --think off          this run without thinking
    opencode-local --think limited --think-budget 2048
    opencode-local --think on           unlimited thinking
    opencode-local --ctx 131072         different context size
    opencode-local --idle-minutes 30    unload the model after 30 idle minutes (0 = never)
    opencode-local --think off --save   make it the default
    opencode-local --server-only        just the server (for other tools)
    opencode-local --status             what is running
    opencode-local --stop               stop the server
    opencode-local run "fix the tests"  anything after the options goes to OpenCode

The server keeps running after OpenCode closes, so the next start is instant.
After 10 idle minutes it unloads the model to free GPU memory and RAM; the
next request loads it again (seconds when the file is still in Windows' cache).
Settings come from settings.json, written by the installer.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from installer import opencode_setup, rapidmlx_setup, server, settings  # noqa: E402

STATE_FILE = os.path.join(ROOT, ".server-state.json")


def parse_args():
    p = argparse.ArgumentParser(prog="opencode-local", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--think", choices=settings.THINK_MODES,
                   help="limited = capped thinking (default), off = none, on = unlimited")
    p.add_argument("--think-budget", type=int, metavar="TOKENS",
                   help="cap for --think limited (default 4096; ~1 min on a 12 GB GPU)")
    p.add_argument("--ctx", type=int, metavar="TOKENS", help="context size, e.g. 65536")
    p.add_argument("--idle-minutes", type=int, metavar="MIN",
                   help="unload the model after MIN idle minutes; the next request loads it "
                        "again (default 10; 0 = keep it loaded)")
    p.add_argument("--save", action="store_true", help="keep these options as the default")
    p.add_argument("--server-only", action="store_true", help="start the server, not OpenCode")
    p.add_argument("--restart", action="store_true", help="restart the server even if running")
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    # Used by the opencode-local launcher (see opencode_setup.install_launcher):
    # --prepare writes the OpenCode command to FILE instead of running it, so
    # the launcher runs OpenCode itself and no Python process is alive while
    # it works; --after restores the terminal once OpenCode has exited.
    p.add_argument("--prepare", metavar="FILE", help=argparse.SUPPRESS)
    p.add_argument("--after", metavar="FILE", help=argparse.SUPPRESS)
    return p.parse_known_args()


DONE = 3  # --prepare exit code: handled, nothing for the launcher to run


def write_handoff(path, exe, passthrough):
    """Command file the launcher runs: OpenCode with the user's arguments."""
    if os.name == "nt":
        line = subprocess.list2cmdline([exe] + passthrough).replace("%", "%%")
        body = "@echo off\r\ncall " + line + "\r\n"
    else:
        import shlex
        body = "exec " + " ".join(shlex.quote(a) for a in [exe] + passthrough) + "\n"
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(body)
    with open(path + ".json", "w", encoding="utf-8") as f:
        json.dump({"modes": save_console_modes(), "lock": folder_lock_path()}, f)


def after(path):
    """Restore the terminal and clear the folder lock after OpenCode exits."""
    try:
        with open(path + ".json", encoding="utf-8") as f:
            info = json.load(f)
    except (OSError, ValueError):
        info = {}
    modes = info.get("modes")
    restore_terminal([tuple(m) if m else None for m in modes] if modes else None)
    if info.get("lock"):
        clear_lock(info["lock"])
    for p in (path, path + ".json"):
        try:
            os.remove(p)
        except OSError:
            pass
    print_closing()
    return 0


def print_closing():
    print("OpenCode closed. The model server is still running (fast next start).\n"
          "  continue the last session:  opencode-local -c\n"
          "  continue a given session:   opencode-local -s <session id>\n"
          "  stop the model server:      opencode-local --stop")


def healthy(port, path="/health"):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def health_path(s):
    return "/v1/models" if settings.is_mlx(s) else "/health"


def read_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def stop():
    server.kill_existing_server(ROOT)
    rapidmlx_setup.kill_existing(ROOT)
    try:
        os.remove(STATE_FILE)
    except OSError:
        pass


def tail(path, n=15):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def ensure_server(s):
    """Start llama-server with settings s unless the same one is already up."""
    if not settings.is_mlx(s) and s.get("backend") != "cpu":
        exe = server.find_server_binary(settings.resolve_path(ROOT, s["bin_dir"]))
        if exe:
            s["fit_target_runtime"] = server.fit_target_for(exe, s["fit_margin_mib"])
    args = settings.server_args(s, ROOT)
    port = s["port"]
    state = read_state()
    hpath = health_path(s)
    if healthy(port, hpath):
        if state and settings.comparable(state.get("args") or []) == settings.comparable(args):
            return True
        if not state:
            print(f"Something else is already serving port {port} (not started by "
                  f"opencode-local). Stop it, or change \"port\" in {settings.path(ROOT)}.")
            return False
        print("Settings changed; restarting the model server ...")
    stop()
    log_out, log_err = os.path.join(ROOT, "server.log"), os.path.join(ROOT, "server.err")
    if settings.is_mlx(s):
        exe = rapidmlx_setup.find_rapidmlx(ROOT)
        if not exe:
            print("rapid-mlx not found. Re-run setup.sh / install.py.")
            return False
        proc = rapidmlx_setup.start_rapidmlx(exe, args, log_out, log_err)
        rapidmlx_setup.write_pid(ROOT, proc.pid)
    else:
        bin_path = server.find_server_binary(settings.resolve_path(ROOT, s["bin_dir"]))
        if not bin_path:
            print(f"llama-server not found in {s['bin_dir']}. Re-run setup.cmd / install.py.")
            return False
        model_path = settings.resolve_path(ROOT, s["model_path"])
        if not os.path.exists(model_path):
            print(f"Model file missing: {model_path}. Re-run setup.cmd / install.py.")
            return False
        proc = server.start_server(bin_path, args, log_out, log_err, root=ROOT)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump({"pid": proc.pid, "args": args}, f)
    print(f"Loading {s['display_name']} (first start takes about a minute) ", end="", flush=True)
    t0 = time.time()
    while time.time() - t0 < 900:
        if proc.poll() is not None:
            print("\nThe model server stopped while loading. Last lines of server.err:\n")
            print(tail(log_err))
            try:
                os.remove(STATE_FILE)
            except OSError:
                pass
            return False
        if healthy(port, hpath):
            print(f" ready ({time.time() - t0:.0f} s)")
            return True
        print(".", end="", flush=True)
        time.sleep(2)
    print("\nTimed out waiting for the server; see server.err.")
    return False


def folder_lock_path():
    """One lock file per project folder, holding the OpenCode process id."""
    import hashlib
    key = hashlib.sha1(os.path.normcase(os.path.abspath(os.getcwd())).encode()).hexdigest()[:16]
    d = os.path.join(ROOT, ".opencode-open")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, key + ".pid")


def _pid_alive(pid):
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def running_pid(lock):
    try:
        with open(lock, encoding="ascii") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return None
    return pid if _pid_alive(pid) else None


def write_lock(lock, pid):
    try:
        with open(lock, "w", encoding="ascii") as f:
            f.write(str(pid))
    except OSError:
        pass


def clear_lock(lock):
    try:
        os.remove(lock)
    except OSError:
        pass


def kill_tree(pid):
    """End OpenCode and its children (on Windows, opencode.cmd -> opencode.exe)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        # Not killpg: OpenCode shares this wrapper's (and the shell's) group.
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def save_console_modes():
    """Windows: the console input/output modes before OpenCode changes them."""
    if os.name != "nt":
        return None
    import ctypes
    k32 = ctypes.windll.kernel32
    modes = []
    for std in (-10, -11):  # STD_INPUT_HANDLE, STD_OUTPUT_HANDLE
        h = k32.GetStdHandle(std)
        m = ctypes.c_uint32()
        modes.append((h, m.value) if k32.GetConsoleMode(h, ctypes.byref(m)) else None)
    return modes


def restore_terminal(saved):
    """Put the terminal back the way it was, however OpenCode exited.

    OpenCode's interface turns on mouse tracking; if it is aborted or crashes
    without switching it off, every mouse move then prints codes such as
    "[<35;78;1M" at the prompt. Restore the saved Windows console modes, then
    send the "mouse tracking off" sequences, show the cursor, reset colors."""
    if os.name == "nt" and saved:
        import ctypes
        k32 = ctypes.windll.kernel32
        for entry in saved:
            if entry:
                k32.SetConsoleMode(entry[0], entry[1])
        out = saved[1]
        if out:  # make sure the sequences below are interpreted, not printed
            k32.SetConsoleMode(out[0], out[1] | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    if sys.stdout.isatty():
        sys.stdout.write("\x1b[?1000l\x1b[?1002l\x1b[?1003l\x1b[?1006l\x1b[?1015l"
                         "\x1b[?2004l\x1b[?25h\x1b[0m")
        sys.stdout.flush()
    if os.name == "nt" and saved and saved[1]:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(saved[1][0], saved[1][1])


def write_opencode_config(s):
    """Point OpenCode at this server with the current context size."""
    model = {"id": s["model_id"], "display_name": s["display_name"],
             "reasoning": s.get("reasoning", True)}
    path = os.path.join(ROOT, "opencode.json")
    if settings.is_mlx(s):
        server.write_opencode_json(path, model, s["port"], s["ctx"], provider_key="rapidmlx",
                                   provider_label="rapid-mlx (local)")
    else:
        server.write_opencode_json(path, model, s["port"], s["ctx"])
    opencode_setup.install_config(path)


def main():
    opts, passthrough = parse_args()
    if opts.after:
        return after(opts.after)
    handled = DONE if opts.prepare else 0
    s = settings.load(ROOT)
    if s is None:
        print(f"No settings found ({settings.path(ROOT)}). Run setup.cmd (Windows) or "
              f"python install.py first.")
        return 1
    if opts.stop:
        stop()
        print("Model server stopped.")
        return handled
    if opts.think:
        s["thinking"] = opts.think
    if opts.think_budget:
        s["think_budget"] = opts.think_budget
        if not opts.think:
            s["thinking"] = "limited"
    if opts.ctx:
        s["ctx"] = opts.ctx
        s["ctx_explicit"] = opts.ctx
    if opts.idle_minutes is not None:
        s["idle_minutes"] = max(0, opts.idle_minutes)
    if opts.save:
        settings.save(ROOT, s)
        print("Saved as default.")
    if opts.status:
        state = read_state()
        up = healthy(s["port"], health_path(s))
        print(("Running: " if up else "Not running. Default: ") + settings.describe(s))
        if up and state and state.get("args") != settings.server_args(s, ROOT):
            print("  (the running server uses different settings; the next start restarts it)")
        return handled
    if opts.restart:
        stop()
    if not ensure_server(s):
        return 1
    print(settings.describe(s))
    write_opencode_config(s)
    if opts.server_only:
        print("Server is running in the background. Stop it with: opencode-local --stop")
        return handled
    exe = opencode_setup.find_opencode()
    if not exe:
        print("OpenCode is not installed. Install it with:\n"
              f"  npm install -g opencode-ai@{opencode_setup.OPENCODE_VERSION}")
        return 1
    lock = folder_lock_path()
    other = running_pid(lock)
    if other and opts.prepare and other == os.getppid():
        other = None
    if other:
        print(f"OpenCode is already open for this folder in another window (process {other}).\n"
              "Two windows in one folder share the session and draw over each other's\n"
              "changes; with -c both continue the SAME session. Close the other one first.")
        try:
            if input("Open a second one anyway? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        except EOFError:
            return 1
    if opts.prepare:
        # The launcher (our parent) runs OpenCode and stays alive while it
        # works, so its process id marks the folder as open.
        write_lock(lock, os.getppid())
        write_handoff(opts.prepare, exe, passthrough)
        return 0
    sys.stdout.flush()
    saved = save_console_modes()
    # Ctrl+C reaches every process in the terminal. OpenCode handles it
    # itself; this wrapper must keep waiting. Before, it exited on Ctrl+C
    # while OpenCode kept running, so the shell prompt came back under a
    # still-drawing OpenCode and mouse codes flooded the window.
    import signal
    old_handler = signal.signal(signal.SIGINT, lambda *_: None)
    proc = None
    rc = 1
    try:
        proc = subprocess.Popen([exe] + passthrough)
        write_lock(lock, proc.pid)
        rc = proc.wait()
    finally:
        if proc is not None and proc.poll() is None:
            # Leaving early for any reason: never orphan OpenCode.
            kill_tree(proc.pid)
        restore_terminal(saved)
        clear_lock(lock)
        signal.signal(signal.SIGINT, old_handler)
    print_closing()
    return rc


if __name__ == "__main__":
    sys.exit(main())
