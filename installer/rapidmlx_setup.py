"""rapid-mlx lifecycle: install, start, health-check - the macOS/Apple Silicon
counterpart to server.py's llama-server path.

rapid-mlx (https://github.com/raullenchai/Rapid-MLX, PyPI: rapid-mlx) is an
OpenAI-compatible inference server built on Apple's MLX framework. Verified
against the PyPI project metadata as the canonical source - a second GitHub
org (bitandmortar/rapid-mlx) makes an identical claim, but PyPI's own
Homepage/Repository/Documentation links all point at raullenchai/Rapid-MLX,
so that's what this installer trusts and what gets installed (`pip install
rapid-mlx`, the normal PyPI channel - not the project's own `curl | bash`
one-liner, which this installer deliberately never runs automatically for
the same reason TurboQuant's third-party binaries aren't auto-run; see
docs/TURBOQUANT.md for that precedent).

Unlike llama.cpp, rapid-mlx manages its own model download - `rapid-mlx
serve <hf-repo-id>` fetches the MLX-format weights from Hugging Face itself
on first run. This module never touches installer/download.py.

Not exercised on real Apple Silicon hardware while writing this (the dev
environment for this repo is Windows) - the CLI shape, PyPI package name,
and mlx-community repo ids are all verified against real sources (see
docs/MODELS.md), but the actual serve/health-check behavior should be
treated as unverified until someone runs it on a Mac. Please open an issue
with `python install.py`'s output if something here doesn't match reality.
"""
import platform
import subprocess
import sys


def find_rapidmlx():
    import shutil
    return shutil.which("rapid-mlx")


def ensure_rapidmlx(log):
    """Install rapid-mlx via pip (the canonical PyPI package) if the CLI
    isn't already on PATH. Returns the resolved path, or None on failure."""
    exe = find_rapidmlx()
    if exe:
        log(f"Using existing rapid-mlx: {exe}")
        return exe
    log("Installing rapid-mlx (pip install rapid-mlx) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "rapid-mlx"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout or "").strip().splitlines()[-5:]
    if tail:
        log("\n".join(tail))
    if r.returncode != 0:
        log((r.stderr or "").strip()[-2000:])
        return None
    return find_rapidmlx()


def build_serve_args(repo_id, port):
    return ["serve", repo_id, "--port", str(port), "--host", "127.0.0.1"]


def start_rapidmlx(exe, args, log_out_path, log_err_path):
    out_f = open(log_out_path, "wb")
    err_f = open(log_err_path, "wb")
    proc = subprocess.Popen([exe] + args, stdout=out_f, stderr=err_f)
    return proc


def kill_existing():
    """Best-effort: stop any rapid-mlx server left running from a previous run."""
    try:
        subprocess.run(["pkill", "-f", "rapid-mlx"], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass
    import time
    time.sleep(1)


def is_apple_silicon_macos():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def write_run_script(root, repo_id, port):
    """Write run.sh so the server can be restarted later (with an overridable
    port) without going through the installer again. repo_id is always a
    catalog-defined mlx-community repo string here (not user/archive input),
    but it's still spliced in via shlex.quote rather than bare - same
    reasoning as server.py's write_run_scripts: cheap to do right, and it
    means this code stays safe if the catalog ever becomes user-extensible.
    """
    import os
    import shlex

    sh = f"""#!/usr/bin/env bash
# run.sh - restart the rapid-mlx server configured by install.py.
# Regenerated each install; re-run install.py to change model/quant.
set -euo pipefail
PORT="${{PORT:-{port}}}"
REPO={shlex.quote(repo_id)}

command -v rapid-mlx >/dev/null 2>&1 || {{
    echo "rapid-mlx not found on PATH. Install it with: pip install rapid-mlx" >&2
    exit 1
}}

echo "==> rapid-mlx serve $REPO --port $PORT --host 127.0.0.1"
exec rapid-mlx serve "$REPO" --port "$PORT" --host 127.0.0.1
"""
    sh_path = os.path.join(root, "run.sh")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
