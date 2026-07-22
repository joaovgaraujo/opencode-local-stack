"""OpenCode install + config + smoke test, cross-platform.

Node/npm are never installed silently on Linux (too many package managers,
usually needs sudo) - we print the right command and stop. On Windows we
optionally shell out to winget, which is first-party and requires no sudo
equivalent, if the caller opts in via install_node=True.
"""
import os
import platform
import shutil
import subprocess


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
            "    sudo apt install nodejs npm        # Debian/Ubuntu\n"
            "    sudo dnf install nodejs npm        # Fedora\n"
            "    sudo pacman -S nodejs npm          # Arch\n"
            "    (or install via nvm: https://github.com/nvm-sh/nvm)")


def install_node_windows():
    subprocess.run(["winget", "install", "-e", "--id", "OpenJS.NodeJS.LTS",
                     "--accept-source-agreements", "--accept-package-agreements"])
    return find_node()


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
