#!/usr/bin/env bash
# setup.sh - one-step setup for Linux and macOS. Runs install.py's wizard
# (model, context, thinking) with Node.js installed if missing.
#
#   ./setup.sh                                  the wizard
#   ./setup.sh --hf https://huggingface.co/OWNER/NAME
#   ./setup.sh --model auto --think off --non-interactive
#   ./setup.sh --build-llamacpp                 Linux: compile llama.cpp with CUDA
#
# Needs Python 3.9+ (preinstalled on Ubuntu and macOS). Nothing here uses sudo:
# llama.cpp, Node.js and the model go into this folder, OpenCode and the
# opencode-local command into ~/.local.
set -euo pipefail
cd "$(dirname "$0")"

PY=""
for c in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1 &&
       "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
        PY="$c"
        break
    fi
done
if [ -z "$PY" ]; then
    echo "Python 3.9 or newer is needed."
    echo "  Ubuntu/Debian: sudo apt install python3"
    echo "  Fedora:        sudo dnf install python3"
    echo "  macOS:         xcode-select --install"
    exit 1
fi
exec "$PY" install.py --install-node "$@"
