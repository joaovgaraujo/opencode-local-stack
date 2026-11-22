Download **OpenCodeLocal-Setup.exe** (Windows) or run the Linux/macOS one-line installer; existing installs keep their models and settings.

## Fix

- **Garbled OpenCode window with two windows open in one folder.** Starting `opencode-local -c` in two terminals made both continue the same session at once: each window redrew changes made by the other, and both queued work on the same model server. `opencode-local` now remembers which folders already have OpenCode open, and warns and asks before opening a second one.
