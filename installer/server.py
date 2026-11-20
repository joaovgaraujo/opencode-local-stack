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
    popen_kwargs = {"stdin": subprocess.DEVNULL}
    if platform.system() == "Windows":
        # Own hidden console: survives the terminal that started it closing,
        # and Ctrl+C in that terminal (e.g. quitting OpenCode) doesn't reach it.
        popen_kwargs["creationflags"] = (subprocess.CREATE_NO_WINDOW |
                                         subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        popen_kwargs["start_new_session"] = True
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


def device_memory(server_bin):
    """[(total_mib, free_mib), ...] per GPU from `llama-server --list-devices`
    ("CUDA0: NAME (11956 MiB, 10924 MiB free)"); [] when unknown."""
    import re
    try:
        r = subprocess.run([server_bin, "--list-devices"], capture_output=True, text=True,
                           timeout=60, encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return []
    out = (r.stdout or "") + (r.stderr or "")
    return [(int(t), int(f)) for t, f in re.findall(r"\((\d+) MiB, (\d+) MiB free\)", out)]


def fit_target_for(server_bin, margin_mib):
    """--fit-target value: the margin itself, unless the runtime reports more
    free memory than the card has. Seen with a CUDA 13.3 source build on a
    laptop (55421 MiB "free" on a 11956 MiB GPU): --fit then plans the whole
    model onto the GPU and loading fails with cudaMalloc out of memory. In
    that case raise the margin so --fit uses at most total - margin."""
    mems = device_memory(server_bin)
    if not mems or all(free <= total for total, free in mems):
        return str(margin_mib)
    return ",".join(str(max(margin_mib, free - total + margin_mib)) for total, free in mems)


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


def write_run_scripts(root, python_exe):
    """run.ps1 / run.sh start the server alone (no OpenCode). They call the
    same start command as `opencode-local`, so they always use settings.json
    and can't drift from it. The paths are spliced in as inert quoted literals
    (_ps1_single_quoted / _sh_quoted)."""
    start_py = os.path.join(root, "start.py")
    ps1 = ("# run.ps1 - start the llama-server set up by install.py (no OpenCode).\n"
           "# Same as: opencode-local --server-only. Settings live in settings.json.\n"
           f"& {_ps1_single_quoted(python_exe)} {_ps1_single_quoted(start_py)} "
           "--server-only @args\n")
    with open(os.path.join(root, "run.ps1"), "w", encoding="utf-8", newline="\n") as f:
        f.write(ps1)
    sh = ("#!/usr/bin/env bash\n"
          "# run.sh - start the llama-server set up by install.py (no OpenCode).\n"
          f'exec {_sh_quoted(python_exe)} {_sh_quoted(start_py)} --server-only "$@"\n')
    sh_path = os.path.join(root, "run.sh")
    with open(sh_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(sh)
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
