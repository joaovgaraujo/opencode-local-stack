Download **OpenCodeLocal-Setup.exe** (Windows) or run the Linux/macOS one-line installer; existing installs keep their models and settings.

## Fix

- **Garbled window after Ctrl+C in OpenCode (Windows).** Ctrl+C reaches every program in the terminal. `opencode-local` exited on it while OpenCode kept running, so the PowerShell prompt came back under a still-drawing OpenCode (after a "Terminate batch job (Y/N)?" prompt), and mouse moves flooded the window with codes such as `[<35;78;1M` and `[555;163;26M`. `opencode-local` now ignores Ctrl+C itself, waits until OpenCode has exited, and only then restores the terminal. If it ever has to exit first, it ends OpenCode rather than leaving it running.

OpenCode has a related upstream bug, where an interrupted OpenCode can leave mouse tracking on ([anomalyco/opencode#6912](https://github.com/anomalyco/opencode/issues/6912)). The terminal restore that runs after OpenCode exits covers that case.
