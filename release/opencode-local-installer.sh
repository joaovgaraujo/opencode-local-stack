#!/usr/bin/env bash
# opencode-local-installer.sh - install or update OpenCode Local on Linux/macOS.
#
#   curl -fsSL https://github.com/joaovgaraujo/opencode-local-stack/releases/latest/download/opencode-local-installer.sh | bash
#
# Downloads the latest release into ~/opencode-local (or $OCL_DEST), keeping
# models, llama.cpp and settings already there, then runs setup.sh. Pass setup
# options after `bash -s --`, e.g.  ... | bash -s -- --model auto --think off --non-interactive
set -euo pipefail
REPO="joaovgaraujo/opencode-local-stack"
DEST="${OCL_DEST:-$HOME/opencode-local}"

echo "== OpenCode Local installer =="
echo "Installs into $DEST (set OCL_DEST to change). The model needs 5-25 GB of disk."
if [ -n "${OCL_REF:-}" ]; then
    URL="https://github.com/$REPO/archive/$OCL_REF.tar.gz"; LABEL="$OCL_REF"
else
    LABEL=$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
            sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)
    URL="https://github.com/$REPO/archive/refs/tags/$LABEL.tar.gz"
fi
echo "Downloading opencode-local-stack $LABEL ..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -fsSL "$URL" | tar xz -C "$TMP"
SRC=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)
mkdir -p "$DEST"
# Copy over the release's files; models/, llama.cpp/, settings.json stay.
cp -R "$SRC"/. "$DEST"/
chmod +x "$DEST/setup.sh"
echo "Files are in $DEST"
echo
# Piped into bash, stdin is the script itself: give the wizard the terminal.
if [ -t 0 ]; then
    exec "$DEST/setup.sh" "$@"
elif [ -r /dev/tty ] && (: < /dev/tty) 2>/dev/null; then
    exec "$DEST/setup.sh" "$@" < /dev/tty
else
    exec "$DEST/setup.sh" "$@"
fi
