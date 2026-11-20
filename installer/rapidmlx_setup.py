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
the same reason a project's own bootstrap script isn't auto-run).

Unlike llama.cpp, rapid-mlx manages its own model download - `rapid-mlx
serve <hf-repo-id>` fetches the MLX-format weights from Hugging Face itself
on first run. This module never touches installer/download.py.

Validated end-to-end on a 16 GB Apple M4 with rapid-mlx 0.10.15: the serve
flags, health check, validation suite, and OpenCode agentic smoke test all
pass with qwen3.5-9b-4bit. See docs/MACOS.md for the measured footprints,
tok/s, and context ceilings. tok/s and the fit verdicts depend on the exact
machine, so re-measure on yours.
"""
import os
import platform
import shutil
import signal
import subprocess
import sys
import time


RAPIDMLX_VERSION = "1.0.5"
PYTHON_VERSION = (3, 12)   # rapid-mlx 1.x requires Python >= 3.12
# Newest Python known to have wheels for all of rapid-mlx's dependencies. On
# Python 3.15 (Homebrew, 2026-11) tiktoken has no wheel and fails to build.
PYTHON_MAX = (3, 13)


def _python_ok(version):
    return PYTHON_VERSION <= tuple(version) <= PYTHON_MAX
VENV_DIR_NAME = ".rapidmlx-venv"
UV_DIR_NAME = ".uv"
PID_FILE_NAME = ".rapidmlx.pid"
PFLASH_MODES = {"off", "auto", "always"}


def _local_rapidmlx(root):
    return os.path.join(root, VENV_DIR_NAME, "bin", "rapid-mlx")


def rapidmlx_version(exe):
    """Return the ``rapid-mlx --version`` number (e.g. "0.10.15") or None."""
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True,
                             timeout=15, encoding="utf-8", errors="replace").stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for tok in (out or "").split():
        if tok[:1].isdigit() and "." in tok:
            return tok.strip()
    return None


def _version_tuple(v):
    parts = []
    for p in (v or "").split("."):
        num = "".join(ch for ch in p if ch.isdigit())
        parts.append(int(num) if num else 0)
    return tuple(parts)


def _is_compatible(exe):
    """A PATH rapid-mlx is only usable if it's new enough to understand the
    serve flags this installer emits. The ``--pflash`` flag landed in the
    0.10.x line; older builds (e.g. Homebrew's 0.6.x) don't understand it.
    Rather than probe every flag, gate on the pinned major.minor so behavior
    stays reproducible."""
    ver = rapidmlx_version(exe)
    if not ver:
        return False, None
    return _version_tuple(ver) >= _version_tuple(RAPIDMLX_VERSION), ver


def find_rapidmlx(root=None):
    """Prefer the installer-managed venv CLI, then a compatible PATH install.

    A PATH ``rapid-mlx`` is only returned when its version is >= the pinned
    ``RAPIDMLX_VERSION``; an older Homebrew/pipx build is ignored so the
    installer can fall back to its own pinned venv (see ensure_rapidmlx)."""
    if root:
        local = _local_rapidmlx(root)
        if os.path.isfile(local) and os.access(local, os.X_OK):
            return local
    path_exe = shutil.which("rapid-mlx")
    if path_exe and _is_compatible(path_exe)[0]:
        return path_exe
    return None


def ensure_rapidmlx(root, log):
    """Return rapid-mlx, installing a pinned copy in a project venv if needed.

    Homebrew Python follows PEP 668 and rejects global ``pip install`` calls.
    A repository-local venv avoids modifying the system interpreter, is used by
    the generated run.sh, and keeps the tested Rapid-MLX version reproducible.
    """
    exe = find_rapidmlx(root)
    if exe:
        log(f"Using existing rapid-mlx: {exe}")
        return exe

    # A PATH install that's too old to understand our serve flags is ignored
    # by find_rapidmlx; say so before falling back to the pinned venv, or the
    # skip looks like a bug (issue #: Homebrew 0.6.x + --pflash).
    path_exe = shutil.which("rapid-mlx")
    if path_exe:
        ok, ver = _is_compatible(path_exe)
        if not ok:
            log(f"Ignoring {path_exe} (rapid-mlx {ver or '?'} < pinned "
                f"{RAPIDMLX_VERSION}; too old for this installer's serve flags); "
                f"installing the pinned version in {VENV_DIR_NAME} instead.")

    venv_dir = os.path.join(root, VENV_DIR_NAME)
    venv_python = os.path.join(venv_dir, "bin", "python")
    if os.path.isfile(venv_python) and not _python_ok(_python_version(venv_python)):
        log(f"{VENV_DIR_NAME} uses a Python outside 3.12-3.13 (rapid-mlx and its "
            "dependencies need that range); recreating it ...")
        shutil.rmtree(venv_dir, ignore_errors=True)
    uv = None
    if not os.path.isfile(venv_python):
        py = _find_python()
        if py:
            log(f"Creating {VENV_DIR_NAME} with {py} ...")
            created = subprocess.run([py, "-m", "venv", venv_dir], capture_output=True,
                                     text=True, encoding="utf-8", errors="replace")
        else:
            # macOS ships Python 3.9; get a private 3.12 via uv instead of
            # asking for Homebrew or an admin install.
            uv = _ensure_uv(root, log)
            if not uv:
                return None
            log("Creating the rapid-mlx environment with uv's Python 3.12 (downloads ~30 MB) ...")
            created = subprocess.run([uv, "venv", "--python", "3.12", venv_dir],
                                     capture_output=True, text=True, encoding="utf-8",
                                     errors="replace")
        if created.returncode != 0:
            log((created.stderr or created.stdout or "venv creation failed").strip()[-2000:])
            return None

    log(f"Installing rapid-mlx=={RAPIDMLX_VERSION} in {VENV_DIR_NAME} ...")
    uv = uv or _local_uv(root)
    if uv:
        cmd = [uv, "pip", "install", "--python", venv_python, f"rapid-mlx=={RAPIDMLX_VERSION}"]
    else:
        cmd = [venv_python, "-m", "pip", "install", f"rapid-mlx=={RAPIDMLX_VERSION}"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = (r.stdout or "").strip().splitlines()[-5:]
    if tail:
        log("\n".join(tail))
    if r.returncode != 0:
        log((r.stderr or "").strip()[-2000:])
        return None
    return find_rapidmlx(root)


def _python_version(exe):
    try:
        out = subprocess.run([exe, "-c", "import sys; print(sys.version_info[0], "
                              "sys.version_info[1])"], capture_output=True, text=True,
                             timeout=30).stdout.split()
        return int(out[0]), int(out[1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return (0, 0)


def _find_python():
    """A Python 3.12-3.13 already on this Mac (Homebrew, python.org), or None;
    then uv provides one."""
    cands = [shutil.which(n) for n in ("python3.13", "python3.12")]
    cands += ["/opt/homebrew/bin/python3.13", "/opt/homebrew/bin/python3.12"]
    cands += ["/opt/homebrew/bin/python3", "/usr/local/bin/python3",
              "/Library/Frameworks/Python.framework/Versions/Current/bin/python3", sys.executable]
    for c in cands:
        if c and os.path.exists(c) and _python_ok(_python_version(c)):
            return c
    return None


def _local_uv(root):
    for dirpath, _dirs, files in os.walk(os.path.join(root, UV_DIR_NAME)):
        if "uv" in files:
            return os.path.join(dirpath, "uv")
    return shutil.which("uv")


def _ensure_uv(root, log):
    """uv (astral-sh/uv), unpacked into <repo>/.uv: no admin rights needed.
    It installs a private Python 3.12 for rapid-mlx."""
    found = _local_uv(root)
    if found:
        return found
    import tarfile
    from . import download
    arch = "aarch64" if platform.machine() == "arm64" else "x86_64"
    name = f"uv-{arch}-apple-darwin.tar.gz"
    dest = os.path.join(root, UV_DIR_NAME)
    os.makedirs(dest, exist_ok=True)
    archive = os.path.join(dest, name)
    log("Downloading uv (to install Python 3.12 for rapid-mlx) ...")
    try:
        download.download_with_retries(
            f"https://github.com/astral-sh/uv/releases/latest/download/{name}", archive)
        download.extract_archive(archive, dest)
    except (OSError, RuntimeError, tarfile.TarError) as e:
        log(f"  could not get uv: {e}. Install Python 3.12+ (https://www.python.org/downloads/) "
            "and re-run.")
        return None
    finally:
        if os.path.exists(archive):
            os.remove(archive)
    return _local_uv(root)


def effort_for(thinking, budget):
    """rapid-mlx has no token-budget flag; --default-reasoning-effort maps to
    a thinking-token cap for templates without native levels."""
    if thinking == "off":
        return "none"
    if thinking == "limited":
        return "low" if budget <= 1024 else "medium" if budget <= 4096 else "high"
    return None


def build_serve_args(repo_id, port, pflash="off", thinking="on", think_budget=4096,
                     idle_minutes=0, ctx=None):
    # The catalog deliberately contains text models only. Rapid-MLX's
    # filename heuristic can mistake Qwen3.5 text checkpoints for MLLMs.
    #
    # --pflash off: Rapid-MLX auto-enables PFlash ("always") for verified
    # Qwen3.5/3.6 aliases, but its PFlash path refuses to run once the model is
    # (mis)classified as multimodal - and that classification ignores our
    # --no-mllm override, so serve hard-exits with "--pflash is not supported
    # for multimodal models" before the server ever binds. Forcing it off keeps
    # startup reliable across the catalog; the models here are text-only, so the
    # long-prompt prefill compression PFlash would add is not worth a crash.
    if pflash not in PFLASH_MODES:
        raise ValueError(f"Unsupported Rapid-MLX PFlash mode: {pflash}")
    args = ["--no-telemetry", "serve", repo_id, "--no-mllm",
            "--pflash", pflash,
            "--port", str(port), "--host", "127.0.0.1"]
    effort = effort_for(thinking, think_budget)
    if effort:
        args += ["--default-reasoning-effort", effort]
    if idle_minutes and idle_minutes > 0:
        args += ["--idle-unload-seconds", str(int(idle_minutes) * 60)]
    if ctx:
        args += ["--context-length", str(int(ctx))]
    return args


def start_rapidmlx(exe, args, log_out_path, log_err_path):
    out_f = open(log_out_path, "wb")
    err_f = open(log_err_path, "wb")
    try:
        # own session: keeps running after the terminal that started it closes
        return subprocess.Popen([exe] + args, stdout=out_f, stderr=err_f,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    finally:
        out_f.close()
        err_f.close()


def _pid_path(root):
    return os.path.join(root, PID_FILE_NAME)


def write_pid(root, pid):
    with open(_pid_path(root), "w", encoding="ascii") as f:
        f.write(f"{pid}\n")


def kill_existing(root):
    """Stop only the Rapid-MLX process previously started in this repository.

    The old ``pkill -f rapid-mlx`` killed every user's Rapid-MLX server,
    including unrelated projects. A pid file lets repeat installer runs clean
    up their own process without reaching outside the project.
    """
    pid_path = _pid_path(root)
    try:
        with open(pid_path, encoding="ascii") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        return

    try:
        command = subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                                 capture_output=True, text=True, timeout=5).stdout
        if "rapid-mlx" in command:
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.2)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        try:
            os.remove(pid_path)
        except OSError:
            pass


def is_apple_silicon_macos():
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def write_run_script(root, repo_id, port, pflash="off"):
    """Write a restart script that also works with the project-local venv."""
    import shlex

    if pflash not in PFLASH_MODES:
        raise ValueError(f"Unsupported Rapid-MLX PFlash mode: {pflash}")

    sh = f"""#!/usr/bin/env bash
# run.sh - restart the rapid-mlx server configured by install.py.
# Regenerated each install; re-run install.py to change model/quant.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PORT="${{PORT:-{port}}}"
REPO={shlex.quote(repo_id)}
LOCAL_RAPID_MLX="$SCRIPT_DIR/{VENV_DIR_NAME}/bin/rapid-mlx"

if [ -x "$LOCAL_RAPID_MLX" ]; then
    RAPID_MLX="$LOCAL_RAPID_MLX"
elif command -v rapid-mlx >/dev/null 2>&1; then
    RAPID_MLX="$(command -v rapid-mlx)"
else
    echo "rapid-mlx is not installed. Re-run install.py to create {VENV_DIR_NAME}." >&2
    exit 1
fi

echo "==> $RAPID_MLX --no-telemetry serve $REPO --no-mllm --pflash {pflash} --port $PORT --host 127.0.0.1"
exec "$RAPID_MLX" --no-telemetry serve "$REPO" --no-mllm --pflash {pflash} --port "$PORT" --host 127.0.0.1
"""
    sh_path = os.path.join(root, "run.sh")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
