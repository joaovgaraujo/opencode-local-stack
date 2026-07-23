"""llama-server lifecycle: build args, start/stop, health check, and write the
generated artifacts (opencode.json, run.ps1/run.sh) that let you restart the
server later without re-running the installer."""
import json
import os
import platform
import subprocess
import time
import urllib.error
import urllib.request


def server_exe_name():
    return "llama-server.exe" if platform.system() == "Windows" else "llama-server"


def find_server_binary(bin_dir):
    """The extracted release archive may place the binary at the root or
    under build/bin/ (layout has changed across llama.cpp releases) - search
    for it instead of assuming a fixed path."""
    target = server_exe_name()
    for root, _dirs, files in os.walk(bin_dir):
        if target in files:
            return os.path.join(root, target)
    return None


def build_server_args(model, quant, model_path, port, profile, kv_type="q8_0", backend=None,
                      extra_args=None):
    from . import catalog
    if backend not in {"cuda", "vulkan", "rocm", "cpu"}:
        raise ValueError(f"backend must be resolved before building server arguments: {backend!r}")
    ctx = catalog.ctx_sizes(model)[profile]
    args = [
        "-m", model_path,
        "--alias", model["id"],
        "--ctx-size", str(ctx),
        "--flash-attn", "on",
        "-ctk", kv_type,
        "-ctv", kv_type,
        "--jinja",
        "--host", "127.0.0.1",
        "--port", str(port),
    ]
    if backend == "cpu":
        args += ["--n-gpu-layers", "0"]
    else:
        args += ["--n-gpu-layers", "999"]
        if model["arch"] == "moe":
            args += ["--cpu-moe"]
    return args + list(extra_args or [])


PID_FILE_NAME = ".llama-server.pid"


def _pid_path(root):
    return os.path.join(root, PID_FILE_NAME)


def _remove_pid_file(root):
    try:
        os.remove(_pid_path(root))
    except OSError:
        pass


def _running_executable(pid):
    """Return the executable path for pid, or None if it cannot be verified."""
    if platform.system() == "Windows":
        command = ("(Get-CimInstance Win32_Process -Filter 'ProcessId = "
                   f"{pid}' -ErrorAction SilentlyContinue).ExecutablePath")
        try:
            result = subprocess.run(["powershell", "-NoProfile", "-Command", command],
                                    capture_output=True, text=True, encoding="utf-8",
                                    errors="replace", timeout=5)
            return result.stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None
    proc_exe = f"/proc/{pid}/exe"
    try:
        return os.path.realpath(proc_exe) if os.path.exists(proc_exe) else None
    except OSError:
        return None


def kill_existing_server(root):
    """Stop only the llama-server previously started by this repository.

    A PID alone can be stale and reused, so the PID file also records the
    resolved executable path. The process is terminated only when its current
    executable still matches; unrelated llama-server instances are untouched.
    """
    try:
        with open(_pid_path(root), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        pid = int(lines[0])
        expected_executable = os.path.realpath(lines[1])
    except (OSError, ValueError, IndexError):
        return

    actual_executable = _running_executable(pid)
    same_executable = (actual_executable is not None and
                       os.path.normcase(os.path.realpath(actual_executable)) ==
                       os.path.normcase(expected_executable))
    if not same_executable:
        _remove_pid_file(root)
        return

    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True, timeout=15)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
            for _ in range(25):
                if _running_executable(pid) is None:
                    break
                time.sleep(0.2)
            else:
                os.kill(pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    finally:
        _remove_pid_file(root)
    time.sleep(1)


def start_server(server_bin, args, log_out_path, log_err_path, root=None):
    out_f = open(log_out_path, "wb")
    err_f = open(log_err_path, "wb")
    popen_kwargs = {}
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        proc = subprocess.Popen([server_bin] + args, stdout=out_f, stderr=err_f,
                                **popen_kwargs)
    finally:
        out_f.close()
        err_f.close()
    if root:
        with open(_pid_path(root), "w", encoding="utf-8") as handle:
            handle.write(f"{proc.pid}\n{os.path.realpath(server_bin)}\n")
    return proc


def wait_for_health(port, timeout_s=360, interval_s=2, path="/health", process=None):
    """Poll path until it returns HTTP 200, stopping if a child server exits.

    ``process`` is optional so existing llama.cpp callers retain their
    behavior. The Rapid-MLX launcher supplies its Popen object: a failed model
    initialization should report its log error immediately, not consume the
    30-minute first-download grace period.
    """
    url = f"http://127.0.0.1:{port}{path}"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(interval_s)
    return False


def get_served_alias(port):
    url = f"http://127.0.0.1:{port}/v1/models"
    with urllib.request.urlopen(url, timeout=10) as r:
        data = json.load(r)
    return data["data"][0]["id"]


def write_opencode_json(path, model, port, ctx, provider_key="llamacpp",
                         provider_label="llama-server (local)", served_model_id=None):
    """Write a provider config using the exact model identifier the server accepts.

    llama.cpp normally serves the catalog ID, while Rapid-MLX returns its full
    Hugging Face repository ID. ``served_model_id`` keeps the generated
    OpenCode configuration aligned with either backend.
    """
    model_id = served_model_id or model["id"]
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"{provider_key}/{model_id}",
        "provider": {
            provider_key: {
                "npm": "@ai-sdk/openai-compatible",
                "name": provider_label,
                "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                "models": {
                    model_id: {
                        "name": f"{model['display_name']} (local)",
                        "tools": True,
                        "reasoning": bool(model.get("reasoning")),
                        "limit": {"context": ctx, "output": min(ctx // 2, 32768)},
                    }
                },
            }
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def _ps1_single_quoted(s):
    """A PowerShell single-quoted string literal is fully inert - no `$(...)`
    subexpression evaluation, no variable interpolation - only an embedded
    single quote needs escaping (by doubling it)."""
    return "'" + s.replace("'", "''") + "'"


def _sh_quoted(s):
    """A shlex-quoted (single-quoted, shell-escaped) literal, safe to splice
    directly into generated shell source with no re-evaluation risk."""
    import shlex
    return shlex.quote(s)


def write_run_scripts(root, bin_dir, model_path, model_id, arch, port, ctx, kv_type, backend,
                      extra_args=None):
    """Write run.ps1 (Windows) and run.sh (Linux) so the server can be
    restarted later (with overridable -Port/-CtxSize/-KvType) without going
    through the installer/GUI again. Both are generated from the same
    resolved args so they can never drift apart.

    rel_bin/rel_model come from the actual extracted directory layout of a
    downloaded archive (see find_server_binary), not a fixed constant - so
    they're spliced in as properly-quoted literals (_ps1_single_quoted /
    _sh_quoted), never bare inside a double-quoted string. A double-quoted
    PowerShell or bash string still expands `$(...)`/backticks, so a path
    segment containing one would otherwise execute when the script runs.
    """
    n_gpu_layers = "0" if backend == "cpu" else "999"
    moe_flag = (backend != "cpu" and arch == "moe")
    extra_args = list(extra_args or [])
    extra_ps_entries = "".join(f",\n    {_ps1_single_quoted(arg)}" for arg in extra_args)
    extra_sh_words = " ".join(_sh_quoted(arg) for arg in extra_args)

    # Forward slashes work in both PowerShell (Windows and pwsh-on-Linux) and
    # bash - avoids writing a run.sh full of backslashes when install.py runs
    # on Windows (os.path.relpath uses the host's native separator).
    rel_bin = os.path.relpath(bin_dir, root).replace(os.sep, "/")
    rel_model = os.path.relpath(model_path, root).replace(os.sep, "/")

    ps1 = f"""# run.ps1 - restart the llama-server configured by install.py.
# Regenerated each install; re-run install.py to change model/quant/profile.
param(
    [string]$BinDir    = (Join-Path $PSScriptRoot {_ps1_single_quoted(rel_bin)}),
    [string]$ModelPath = (Join-Path $PSScriptRoot {_ps1_single_quoted(rel_model)}),
    [int]$Port         = {port},
    [int]$CtxSize      = {ctx},
    [string]$KvType    = "{kv_type}"
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $ModelPath)) {{ throw "Model not found: $ModelPath" }}
$Server = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $Server)) {{ throw "llama-server.exe not found in: $BinDir" }}

$serverArgs = @(
    "-m", $ModelPath,
    "--alias", "{model_id}",
    "--n-gpu-layers", "{n_gpu_layers}",
{'    "--cpu-moe",' if moe_flag else ''}
    "--ctx-size", "$CtxSize",
    "--flash-attn", "on",
    "-ctk", $KvType,
    "-ctv", $KvType,
    "--jinja",
    "--host", "127.0.0.1",
    "--port", "$Port"{extra_ps_entries}
)
Write-Host "==> llama-server $($serverArgs -join ' ')"
& $Server @serverArgs
"""
    with open(os.path.join(root, "run.ps1"), "w", encoding="utf-8", newline="\n") as f:
        f.write(ps1)

    moe_line = '    "--cpu-moe" \\\n' if moe_flag else ""
    # Deliberately spelled out as if/else rather than a compact "${1:-...}"
    # default: rel_bin/rel_model reflect the archive's own directory layout
    # (see the write_run_scripts docstring), so they're spliced in through
    # _sh_quoted() as an unambiguous single-quoted literal - simpler to
    # verify correct than nesting quote-switches inside a parameter expansion.
    sh = f"""#!/usr/bin/env bash
# run.sh - restart the llama-server configured by install.py.
# Regenerated each install; re-run install.py to change model/quant/profile.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
if [ -n "${{1:-}}" ]; then
    BIN_DIR="$1"
else
    BIN_DIR="$SCRIPT_DIR/"{_sh_quoted(rel_bin)}
fi
if [ -n "${{2:-}}" ]; then
    MODEL_PATH="$2"
else
    MODEL_PATH="$SCRIPT_DIR/"{_sh_quoted(rel_model)}
fi
PORT="${{PORT:-{port}}}"
CTX_SIZE="${{CTX_SIZE:-{ctx}}}"
KV_TYPE="${{KV_TYPE:-{kv_type}}}"
EXTRA_ARGS=({extra_sh_words})

[ -f "$MODEL_PATH" ] || {{ echo "Model not found: $MODEL_PATH" >&2; exit 1; }}
SERVER="$BIN_DIR/llama-server"
[ -x "$SERVER" ] || SERVER="$(find "$BIN_DIR" -name llama-server -type f | head -n1)"
[ -x "$SERVER" ] || {{ echo "llama-server not found under: $BIN_DIR" >&2; exit 1; }}

echo "==> $SERVER -m $MODEL_PATH --alias {model_id} --n-gpu-layers {n_gpu_layers} ..."
exec "$SERVER" \\
    -m "$MODEL_PATH" \\
    --alias "{model_id}" \\
    --n-gpu-layers {n_gpu_layers} \\
{moe_line}    --ctx-size "$CTX_SIZE" \\
    --flash-attn on \\
    -ctk "$KV_TYPE" -ctv "$KV_TYPE" \\
    --jinja \\
    --host 127.0.0.1 --port "$PORT" \\
    "${{EXTRA_ARGS[@]}}"
"""
    sh_path = os.path.join(root, "run.sh")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
