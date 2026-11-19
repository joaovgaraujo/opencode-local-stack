#!/usr/bin/env python3
"""opencode-local: start the local model server (if needed) and open OpenCode
in the current folder. Installed on PATH by install.py / setup.cmd.

    opencode-local                      server + OpenCode here
    opencode-local --think off          this run without thinking
    opencode-local --think limited --think-budget 2048
    opencode-local --think on           unlimited thinking
    opencode-local --ctx 131072         different context size
    opencode-local --think off --save   make it the default
    opencode-local --server-only        just the server (for other tools)
    opencode-local --status             what is running
    opencode-local --stop               stop the server
    opencode-local run "fix the tests"  anything after the options goes to OpenCode

The server keeps running after OpenCode closes, so the next start is instant.
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

from installer import opencode_setup, server, settings  # noqa: E402

STATE_FILE = os.path.join(ROOT, ".server-state.json")


def parse_args():
    p = argparse.ArgumentParser(prog="opencode-local", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--think", choices=settings.THINK_MODES,
                   help="limited = capped thinking (default), off = none, on = unlimited")
    p.add_argument("--think-budget", type=int, metavar="TOKENS",
                   help="cap for --think limited (default 4096; ~1 min on a 12 GB GPU)")
    p.add_argument("--ctx", type=int, metavar="TOKENS", help="context size, e.g. 65536")
    p.add_argument("--save", action="store_true", help="keep these options as the default")
    p.add_argument("--server-only", action="store_true", help="start the server, not OpenCode")
    p.add_argument("--restart", action="store_true", help="restart the server even if running")
    p.add_argument("--status", action="store_true")
    p.add_argument("--stop", action="store_true")
    return p.parse_known_args()


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
            return r.status == 200
    except OSError:
        return False


def read_state():
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def stop():
    server.kill_existing_server(ROOT)
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
    args = settings.server_args(s, ROOT)
    port = s["port"]
    state = read_state()
    if healthy(port):
        if state and state.get("args") == args:
            return True
        if not state:
            print(f"Something else is already serving port {port} (not started by "
                  f"opencode-local). Stop it, or change \"port\" in {settings.path(ROOT)}.")
            return False
        print("Settings changed; restarting the model server ...")
    stop()
    bin_path = server.find_server_binary(settings.resolve_path(ROOT, s["bin_dir"]))
    if not bin_path:
        print(f"llama-server not found in {s['bin_dir']}. Re-run setup.cmd / install.py.")
        return False
    model_path = settings.resolve_path(ROOT, s["model_path"])
    if not os.path.exists(model_path):
        print(f"Model file missing: {model_path}. Re-run setup.cmd / install.py.")
        return False
    log_out, log_err = os.path.join(ROOT, "server.log"), os.path.join(ROOT, "server.err")
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
        if healthy(port):
            print(f" ready ({time.time() - t0:.0f} s)")
            return True
        print(".", end="", flush=True)
        time.sleep(2)
    print("\nTimed out waiting for the server; see server.err.")
    return False


def write_opencode_config(s):
    """Point OpenCode at this server with the current context size."""
    model = {"id": s["model_id"], "display_name": s["display_name"],
             "reasoning": s.get("reasoning", True)}
    path = os.path.join(ROOT, "opencode.json")
    server.write_opencode_json(path, model, s["port"], s["ctx"])
    opencode_setup.install_config(path)


def main():
    opts, passthrough = parse_args()
    s = settings.load(ROOT)
    if s is None:
        print(f"No settings found ({settings.path(ROOT)}). Run setup.cmd (Windows) or "
              f"python install.py first.")
        return 1
    if opts.stop:
        stop()
        print("Model server stopped.")
        return 0
    if opts.think:
        s["thinking"] = opts.think
    if opts.think_budget:
        s["think_budget"] = opts.think_budget
        if not opts.think:
            s["thinking"] = "limited"
    if opts.ctx:
        s["ctx"] = opts.ctx
    if opts.save:
        settings.save(ROOT, s)
        print("Saved as default.")
    if opts.status:
        state = read_state()
        up = healthy(s["port"])
        print(("Running: " if up else "Not running. Default: ") + settings.describe(s))
        if up and state and state.get("args") != settings.server_args(s, ROOT):
            print("  (the running server uses different settings; the next start restarts it)")
        return 0
    if opts.restart:
        stop()
    if not ensure_server(s):
        return 1
    print(settings.describe(s))
    write_opencode_config(s)
    if opts.server_only:
        print("Server is running in the background. Stop it with: opencode-local --stop")
        return 0
    exe = opencode_setup.find_opencode()
    if not exe:
        print("OpenCode is not installed. Install it with:\n"
              f"  npm install -g opencode-ai@{opencode_setup.OPENCODE_VERSION}")
        return 1
    sys.stdout.flush()
    rc = subprocess.call([exe] + passthrough)
    print("OpenCode closed. The model server is still running (fast next start); "
          "stop it with: opencode-local --stop")
    return rc


if __name__ == "__main__":
    sys.exit(main())
