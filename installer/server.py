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


def build_server_args(model, quant, model_path, port, profile, kv_type="q8_0", backend="cuda"):
    from . import catalog
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
    return args


def kill_existing_server():
    """Best-effort: stop any llama-server left running from a previous run."""
    try:
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/IM", "llama-server.exe", "/F"],
                            capture_output=True, timeout=15)
        else:
            subprocess.run(["pkill", "-f", "llama-server"], capture_output=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass
    time.sleep(1)


def start_server(server_bin, args, log_out_path, log_err_path):
    out_f = open(log_out_path, "wb")
    err_f = open(log_err_path, "wb")
    popen_kwargs = {}
    if platform.system() == "Windows":
        popen_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    proc = subprocess.Popen([server_bin] + args, stdout=out_f, stderr=err_f, **popen_kwargs)
    return proc


def wait_for_health(port, timeout_s=360, interval_s=2):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
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


def write_opencode_json(path, model, port, profile):
    from . import catalog
    ctx = catalog.ctx_sizes(model)[profile]
    cfg = {
        "$schema": "https://opencode.ai/config.json",
        "model": f"llamacpp/{model['id']}",
        "provider": {
            "llamacpp": {
                "npm": "@ai-sdk/openai-compatible",
                "name": "llama-server (local)",
                "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
                "models": {
                    model["id"]: {
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


def write_run_scripts(root, bin_dir, model_path, model_id, arch, port, ctx, kv_type, backend):
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
    "--port", "$Port"
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
    --host 127.0.0.1 --port "$PORT"
"""
    sh_path = os.path.join(root, "run.sh")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
