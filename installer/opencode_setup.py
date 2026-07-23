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


def install_node_windows():
    subprocess.run(["winget", "install", "-e", "--id", "OpenJS.NodeJS.LTS",
                     "--accept-source-agreements", "--accept-package-agreements"])
    return find_node()


def install_node_macos():
    brew = shutil.which("brew")
    if not brew:
        return None
    subprocess.run([brew, "install", "node"], check=False)
    return find_node()


NODE_VERSION = "22.22.1"  # LTS; matches the version this stack was validated with

_LINUX_NODE_ARCHES = {"x86_64": "x64", "amd64": "x64", "aarch64": "arm64", "arm64": "arm64"}


def install_node_linux(log):
    """Download the official nodejs.org Linux tarball into <repo>/node and put
    its bin/ on this process's PATH. Project-local and sudo-free, like the
    llama.cpp prebuilt download; the system is never modified."""
    from . import download

    arch = _LINUX_NODE_ARCHES.get(platform.machine().lower())
    if not arch:
        log(f"No official Node build for architecture {platform.machine()!r}.")
        return None
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    node_dir = os.path.join(root, "node")
    dist = f"node-v{NODE_VERSION}-linux-{arch}"
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
            return install_node_macos()
        log("Homebrew not found - cannot auto-install Node. "
            "Install Homebrew (https://brew.sh) or Node directly.")
        return None
    if system == "Linux":
        return install_node_linux(log)
    return None


OPENCODE_VERSION = "1.18.4"
OPENAI_COMPAT_VERSION = "3.0.14"


def npm_install_opencode(log):
    r = subprocess.run(
        ["npm", "install", "-g", f"opencode-ai@{OPENCODE_VERSION}",
         f"@ai-sdk/openai-compatible@{OPENAI_COMPAT_VERSION}"],
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                        shell=(platform.system() == "Windows"))
    tail = (r.stdout or "").strip().splitlines()[-3:]
    if tail:
        log("\n".join(tail))
    return r.returncode == 0


def opencode_config_dir():
    home = os.path.expanduser("~")
    return os.path.join(home, ".config", "opencode")


def install_config(opencode_json_path):
    cfg_dir = opencode_config_dir()
    os.makedirs(cfg_dir, exist_ok=True)
    dest = os.path.join(cfg_dir, "opencode.json")
    shutil.copyfile(opencode_json_path, dest)
    return dest


def _run_opencode(args, cwd, timeout_s):
    exe = shutil.which("opencode")
    if not exe:
        return None
    try:
        r = subprocess.run([exe] + args, cwd=cwd, capture_output=True, text=True,
                            encoding="utf-8", errors="replace",
                            timeout=timeout_s, shell=(platform.system() == "Windows"))
        return r
    except subprocess.TimeoutExpired:
        return None


def warm_up(scratch_dir, model_alias, timeout_s=180):
    os.makedirs(scratch_dir, exist_ok=True)
    return _run_opencode(["run", "--auto", "--dir", scratch_dir, "-m", model_alias,
                          "reply with: ready"], scratch_dir, timeout_s)


def agentic_smoke_test(scratch_dir, model_alias, timeout_s=360):
    """Ask OpenCode to write + run calc.py, then verify it actually printed 5."""
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
