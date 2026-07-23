"""Tkinter preflight and macOS/Homebrew bootstrap for the GUI wizard."""
import os
import platform
import shutil
import subprocess
import sys


def _probe(create_window=False):
    code = "import tkinter as tk"
    if create_window:
        code += "; root=tk.Tk(); root.withdraw(); root.update_idletasks(); root.destroy()"
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=20)


def install_hint():
    system = platform.system()
    if system == "Darwin":
        version = f"{sys.version_info.major}.{sys.version_info.minor}"
        return f"brew install python-tk@{version}  # must match this Python"
    if system == "Windows":
        return "re-run the python.org installer and enable 'tcl/tk and IDLE'"
    return "install python3-tk with your distro package manager (for example: sudo apt install python3-tk)"


def ensure_tkinter(log=print):
    """Return whether Tk can open a window; repair matching Homebrew Python on macOS."""
    imported = _probe()
    if imported.returncode and platform.system() == "Darwin":
        brew = shutil.which("brew")
        if brew:
            prefix = subprocess.run([brew, "--prefix"], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=20).stdout.strip()
            if prefix and os.path.realpath(sys.executable).startswith(prefix):
                formula = f"python-tk@{sys.version_info.major}.{sys.version_info.minor}"
                log(f"Tkinter is missing; installing matching Homebrew formula {formula} ...")
                subprocess.run([brew, "install", formula], check=False)
                imported = _probe()
    if imported.returncode:
        return False, f"Tkinter import failed. Fix: {install_hint()}"
    window = _probe(create_window=True)
    if window.returncode:
        detail = (window.stderr or window.stdout).strip().splitlines()[-1:]
        return False, f"Tkinter cannot open a window ({detail[0] if detail else 'unknown error'})."
    return True, "Tkinter GUI ready"
