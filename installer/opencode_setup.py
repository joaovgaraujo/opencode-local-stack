"""OpenCode install + config + smoke test, cross-platform.

Node/npm are never installed silently - we print the right command and stop.
When the caller opts in via install_node=True we use a first-party, no-sudo
installer on every platform (see try_install_node): winget on Windows,
Homebrew on macOS, and on Linux the official nodejs.org tarball extracted
into a project-local ./node directory (same pattern as the llama.cpp
prebuilt download - nothing touches the system).
"""
import os
import platform
import shutil
import subprocess
import tarfile


def find_node():
    return shutil.which("node")


def find_npm():
    return shutil.which("npm")


def node_install_hint():
    system = platform.system()
    if system == "Windows":
        return "winget install -e --id OpenJS.NodeJS.LTS"
    if system == "Darwin":
        return ("one of:\n"
                "    brew install node                  # Homebrew\n"
                "    (or install via nvm: https://github.com/nvm-sh/nvm)")
    return ("one of:\n"
            "    python install.py --install-node   # official tarball into ./node, no sudo\n"
            "    sudo apt install nodejs npm        # Debian/Ubuntu\n"
            "    sudo dnf install nodejs npm        # Fedora\n"
            "    sudo pacman -S nodejs npm          # Arch\n"
            "    (or install via nvm: https://github.com/nvm-sh/nvm)")


def refresh_windows_path():
    """winget and the Node installer write PATH to the registry; a running
    process (this one, or a terminal opened earlier) keeps its old copy.
    Reload machine + user PATH so node, npm and opencode are found now."""
    if platform.system() != "Windows":
        return
    import winreg
    parts = []
    for hive, key in ((winreg.HKEY_LOCAL_MACHINE,
                       r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
                      (winreg.HKEY_CURRENT_USER, "Environment")):
        try:
            with winreg.OpenKey(hive, key) as k:
                parts.append(os.path.expandvars(winreg.QueryValueEx(k, "Path")[0]))
        except OSError:
            pass
    parts.append(os.environ.get("PATH", ""))
    seen, merged = set(), []
    for entry in os.pathsep.join(parts).split(os.pathsep):
        key = entry.strip().rstrip("\\").lower()
        if key and key not in seen:  # keep it short: cmd.exe chokes on a huge PATH
            seen.add(key)
            merged.append(entry.strip())
    os.environ["PATH"] = os.pathsep.join(merged)


def install_node_windows():
    subprocess.run(["winget", "install", "-e", "--id", "OpenJS.NodeJS.LTS",
                     "--accept-source-agreements", "--accept-package-agreements"])
    refresh_windows_path()
    return find_node()


def install_node_macos():
    brew = shutil.which("brew")
    if not brew:
        return None
    subprocess.run([brew, "install", "node"], check=False)
    return find_node()


NODE_VERSION = "26.13.1"  # current LTS (Lithium)

_LINUX_NODE_ARCHES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def local_node_bins():
    """bin folders of a Node the installer unpacked into <repo>/node."""
    import glob
    return sorted(glob.glob(os.path.join(ROOT, "node", "node-v*", "bin")), reverse=True)


def add_local_tools_to_path():
    """Linux/macOS: make the installer's own Node and ~/.local/bin (where npm
    puts OpenCode for a system Node) visible to this process and its children;
    OpenCode's launcher needs `node` on PATH."""
    if platform.system() == "Windows":
        return
    extra = local_node_bins() + [os.path.join(os.path.expanduser("~"), ".local", "bin")]
    parts = os.environ.get("PATH", "").split(os.pathsep)
    os.environ["PATH"] = os.pathsep.join([d for d in extra if d not in parts] + parts)


def install_node_linux(log):
    """Download the official nodejs.org tarball (Linux or macOS) into
    <repo>/node and put its bin/ on this process's PATH. Project-local and
    sudo-free, like the llama.cpp prebuilt download; the system is never
    modified."""
    from . import download

    arch = _LINUX_NODE_ARCHES.get(platform.machine().lower())
    if not arch:
        log(f"No official Node build for architecture {platform.machine()!r}.")
        return None
    root = ROOT
    node_dir = os.path.join(root, "node")
    os_tag = "darwin" if platform.system() == "Darwin" else "linux"
    dist = f"node-v{NODE_VERSION}-{os_tag}-{arch}"
    bin_dir = os.path.join(node_dir, dist, "bin")
    if not os.path.exists(os.path.join(bin_dir, "node")):
        url = f"https://nodejs.org/dist/v{NODE_VERSION}/{dist}.tar.gz"
        archive = os.path.join(node_dir, f"{dist}.tar.gz")
        os.makedirs(node_dir, exist_ok=True)
        log(f"Downloading Node v{NODE_VERSION} ({arch}) from nodejs.org ...")
        download.download_with_retries(url, archive)
        with tarfile.open(archive) as tf:
            try:
                # the tarball ships bin -> ../lib symlinks, which
                # download.extract_archive's stricter policy rejects
                tf.extractall(node_dir, filter="data")
            except TypeError:  # Python < 3.12: no extraction filters
                tf.extractall(node_dir)
        os.remove(archive)
    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    node = find_node()
    if node:
        log(f"Node v{NODE_VERSION} ready in {bin_dir}")
        log(f"  (for your shell: export PATH=\"{bin_dir}:$PATH\")")
    return node


def try_install_node(log):
    """Install Node when the caller opts in, using a first-party, no-sudo
    installer per platform: winget on Windows, Homebrew on macOS (if present),
    the official nodejs.org tarball into ./node on Linux. Returns the node
    path on success, else None (caller falls back to node_install_hint())."""
    system = platform.system()
    if system == "Windows":
        log("Installing Node LTS via winget ...")
        return install_node_windows()
    if system == "Darwin":
        if shutil.which("brew"):
            log("Installing Node via Homebrew (brew install node) ...")
            node = install_node_macos()
            if node:
                return node
        return install_node_linux(log)   # nodejs.org macOS tarball, no Homebrew needed
    if system == "Linux":
        return install_node_linux(log)
    return None


OPENCODE_VERSION = "1.19.6"
OPENAI_COMPAT_VERSION = "3.0.93"


def _npm_prefix_writable():
    """False when `npm install -g` would need sudo (a system Node from apt or
    a distro package installs globals under /usr)."""
    if platform.system() == "Windows":
        return True
    bin_dir = npm_global_bin()
    if not bin_dir:
        return True
    prefix = os.path.dirname(bin_dir)
    return os.access(prefix, os.W_OK) and os.access(os.path.join(prefix, "lib"), os.W_OK)


def npm_install_opencode(log):
    cmd = ["npm", "install", "-g", f"opencode-ai@{OPENCODE_VERSION}",
           f"@ai-sdk/openai-compatible@{OPENAI_COMPAT_VERSION}"]
    if not _npm_prefix_writable():
        # No sudo: install into ~/.local (opencode lands in ~/.local/bin)
        user_prefix = os.path.join(os.path.expanduser("~"), ".local")
        log(f"  npm's global folder needs root; installing OpenCode into {user_prefix}")
        cmd += ["--prefix", user_prefix]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                       shell=(platform.system() == "Windows"))
    tail = (r.stdout or "").strip().splitlines()[-3:]
    if tail:
        log("\n".join(tail))
    if r.returncode != 0:
        log("npm install failed:\n" + "\n".join((r.stderr or "").strip().splitlines()[-5:]))
    return r.returncode == 0


def npm_global_bin():
    """Folder where `npm install -g` puts the opencode launcher."""
    npm = find_npm()
    if not npm:
        return None
    try:
        r = subprocess.run([npm, "prefix", "-g"], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60,
                           shell=(platform.system() == "Windows"))
    except (OSError, subprocess.TimeoutExpired):
        return None
    prefix = (r.stdout or "").strip()
    if not prefix:
        return None
    return prefix if platform.system() == "Windows" else os.path.join(prefix, "bin")


def ensure_opencode_on_path(log):
    """Make `opencode` callable from this process and tell the user how to
    reach it from their own terminal. Returns True when it was found."""
    if shutil.which("opencode"):
        on_path = True
    else:
        on_path = False
        bin_dir = npm_global_bin()
        if bin_dir and os.path.isdir(bin_dir):
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
        if not shutil.which("opencode"):
            log("  OpenCode was not found after npm install; run "
                "`npm install -g opencode-ai` by hand and check its output.")
            return False
    exe = shutil.which("opencode")
    if not on_path:
        folder = os.path.dirname(exe)
        log(f"  NOTE: OpenCode is in {folder}, which your terminal's PATH may not include.")
        if platform.system() == "Windows":
            log("  Open a NEW PowerShell window before running `opencode`. If it is still "
                "not recognized, run it once as:")
            log(f'    & "{exe}"')
            log("  or add the folder to your user PATH:")
            log(f"    [Environment]::SetEnvironmentVariable('Path', "
                f"[Environment]::GetEnvironmentVariable('Path','User') + ';{folder}', 'User')")
        else:
            log(f'  Add it to your shell: export PATH="{folder}:$PATH"')
    return True


def opencode_config_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, ".config", "opencode")


def install_config(opencode_json_path):
    """Merge this stack's provider + default model into the user's global
    OpenCode config, keeping anything else they configured (other providers,
    agents, MCP servers...). An unreadable existing file is kept as .bak."""
    import json
    cfg_dir = opencode_config_dir()
    os.makedirs(cfg_dir, exist_ok=True)
    dest = os.path.join(cfg_dir, "opencode.json")
    with open(opencode_json_path, encoding="utf-8") as f:
        ours = json.load(f)
    merged = {}
    if os.path.exists(dest):
        try:
            with open(dest, encoding="utf-8") as f:
                merged = json.load(f)
            if not isinstance(merged, dict):
                raise ValueError("not an object")
        except ValueError:
            shutil.copyfile(dest, dest + ".bak")
            merged = {}
    merged["$schema"] = ours["$schema"]
    merged["model"] = ours["model"]
    providers = merged.get("provider") if isinstance(merged.get("provider"), dict) else {}
    providers.update(ours["provider"])
    merged["provider"] = providers
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")
    return dest


def find_opencode():
    """Path of the OpenCode launcher to run, or None.

    On Windows npm installs three launchers: opencode.ps1, opencode.cmd and a
    shell script. PowerShell prefers the .ps1, which the default execution
    policy blocks ("running scripts is disabled"), so OpenCode looks missing
    in every folder. The .cmd launcher is never blocked; use it."""
    if platform.system() == "Windows":
        refresh_windows_path()
        exe = shutil.which("opencode.cmd")
        if not exe:
            bin_dir = npm_global_bin()
            cand = os.path.join(bin_dir, "opencode.cmd") if bin_dir else None
            exe = cand if cand and os.path.exists(cand) else None
        return exe
    add_local_tools_to_path()
    exe = shutil.which("opencode")
    if not exe:
        bin_dir = npm_global_bin()
        cand = os.path.join(bin_dir, "opencode") if bin_dir else None
        exe = cand if cand and os.path.exists(cand) else None
    return exe


def powershell_scripts_blocked():
    """True when PowerShell's effective execution policy blocks unsigned local
    scripts (Restricted / AllSigned / Undefined), which is what makes a plain
    `opencode` fail in PowerShell on Windows."""
    if platform.system() != "Windows":
        return False
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", "Get-ExecutionPolicy"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return False
    return r.stdout.strip() in ("Restricted", "AllSigned", "Undefined")


def allow_powershell_scripts(log):
    """Set the CurrentUser execution policy to RemoteSigned: local scripts
    (npm's opencode.ps1) run, downloaded unsigned scripts still don't. No
    admin rights needed; affects only this Windows user."""
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force"],
                   capture_output=True, text=True)
    ok = not powershell_scripts_blocked()
    log("  PowerShell scripts allowed for this user (RemoteSigned)." if ok else
        "  Could not change the PowerShell policy; `opencode-local` works anyway.")
    return ok


def _ensure_local_bin_on_path(log):
    """~/.local/bin holds opencode-local (and OpenCode for a system Node).
    Ubuntu's ~/.profile adds it at login; macOS's zsh doesn't. Add one guarded
    line to the login profile when it is missing, so a new terminal finds it."""
    local_bin = os.path.join(os.path.expanduser("~"), ".local", "bin")
    if local_bin in os.environ.get("PATH", "").split(os.pathsep):
        return
    profile = os.path.join(os.path.expanduser("~"),
                           ".zprofile" if platform.system() == "Darwin" else ".profile")
    line = 'export PATH="$HOME/.local/bin:$PATH"  # added by opencode-local-stack'
    try:
        existing = open(profile, encoding="utf-8").read() if os.path.exists(profile) else ""
        if ".local/bin" in existing:
            return
        with open(profile, "a", encoding="utf-8") as f:
            f.write(("\n" if existing and not existing.endswith("\n") else "") + line + "\n")
        log(f"  added ~/.local/bin to PATH in {profile} (open a new terminal to use it)")
    except OSError as e:
        log(f"  add ~/.local/bin to your PATH to use opencode-local ({e})")


def install_launcher(root, python_exe, log):
    """Install the `opencode-local` command so it works from any folder.

    Windows: opencode-local.cmd in npm's global folder (on PATH: it is where
    opencode.cmd lives), else the per-user WindowsApps folder, which Windows
    puts on PATH by default. A .cmd is never blocked by PowerShell's execution
    policy. Linux: ~/.local/bin/opencode-local. A copy also goes in the repo."""
    start_py = os.path.join(root, "start.py")
    windows = platform.system() == "Windows"
    if windows:
        # No setlocal on purpose: when this runs in a cmd window whose PATH
        # predates the Node install (e.g. one opened by Windows Terminal), the
        # npm folder is added to that window, so a plain `opencode` (such as
        # the "opencode -s <session>" hint OpenCode prints) works there too.
        body = ('@echo off\r\n'
                'where opencode >nul 2>nul || set "PATH=%APPDATA%\\npm;%PATH%"\r\n'
                f'"{python_exe}" "{start_py}" %*\r\n')
        targets = [npm_global_bin(),
                   os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WindowsApps")]
        name = "opencode-local.cmd"
    else:
        body = f'#!/usr/bin/env sh\nexec "{python_exe}" "{start_py}" "$@"\n'
        targets = [os.path.join(os.path.expanduser("~"), ".local", "bin")]
        name = "opencode-local"
        _ensure_local_bin_on_path(log)
    with open(os.path.join(root, name), "w", encoding="utf-8", newline="") as f:
        f.write(body)
    if not windows:
        os.chmod(os.path.join(root, name), 0o755)
    for folder in targets:
        if not folder or (windows and not os.path.isdir(folder)):
            continue
        try:
            os.makedirs(folder, exist_ok=True)
            dest = os.path.join(folder, name)
            with open(dest, "w", encoding="utf-8", newline="") as f:
                f.write(body)
            if not windows:
                os.chmod(dest, 0o755)
            log(f"  start command installed: {dest}")
            return dest
        except OSError:
            continue
    log(f"  could not put opencode-local on PATH; run it as {os.path.join(root, name)}")
    return None


def _run_opencode(args, cwd, timeout_s):
    exe = find_opencode()
    if not exe:
        return None
    windows = platform.system() == "Windows"
    p = subprocess.Popen([exe] + args, cwd=cwd, stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         encoding="utf-8", errors="replace", shell=windows,
                         **({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if windows
                            else {"start_new_session": True}))
    try:
        out, err = p.communicate(timeout=timeout_s)
        return subprocess.CompletedProcess(p.args, p.returncode, out, err)
    except subprocess.TimeoutExpired:
        # Kill the whole tree: killing only the opencode.cmd wrapper leaves
        # opencode.exe running with our pipes open, and communicate() then
        # waits forever.
        if windows:
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            import signal
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
        try:
            p.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            pass
        return None


def warm_up(scratch_dir, model_alias, timeout_s=180):
    os.makedirs(scratch_dir, exist_ok=True)
    return _run_opencode(["run", "--auto", "--dir", scratch_dir, "-m", model_alias,
                          "reply with: ready"], scratch_dir, timeout_s)


def agentic_smoke_test(scratch_dir, model_alias, timeout_s=360):
    """Ask OpenCode to write + run calc.py, then verify it actually printed 5."""
    os.makedirs(scratch_dir, exist_ok=True)
    for f in ("calc.py",):
        p = os.path.join(scratch_dir, f)
        if os.path.exists(p):
            os.remove(p)
    _run_opencode(
        ["run", "--auto", "--dir", scratch_dir, "-m", model_alias,
         "Create a Python file calc.py that prints the sum of 2 and 3, then run it "
         "with 'python calc.py' and report the exact stdout."],
        scratch_dir, timeout_s,
    )
    calc_path = os.path.join(scratch_dir, "calc.py")
    if not os.path.exists(calc_path):
        return False
    python_exe = shutil.which("python") or shutil.which("python3")
    if not python_exe:
        return False
    r = subprocess.run([python_exe, calc_path], capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=30)
    return r.stdout.strip().startswith("5")
