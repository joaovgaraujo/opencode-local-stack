Download **OpenCodeLocal-Setup.exe** (Windows) or run the Linux/macOS one-line installer; existing installs keep their models and settings.

## Fix: garbled OpenCode window while the agent works

The model was restarting a project's Python dev server by killing every Python process on the PC: `taskkill /F /IM python.exe` and `Get-Process python | Stop-Process -Force`. The `opencode-local` wrapper is a Python process, so it died too. OpenCode was left running without it, the shell prompt came back underneath, and mouse moves filled the window with codes such as `[<35;78;1M`. It happened without anyone pressing Ctrl+C. Those commands also killed any other Python program running on the PC.

- **No Python process while OpenCode runs.** The `opencode-local` launcher now uses Python only to prepare (server, config) and to clean up afterwards. While you work, OpenCode runs directly under the launcher, so killing every Python process can't take its window down. Tested: no `start.py` process exists while OpenCode is running.
- **Killing processes by name is denied.** The installer adds rules to your global OpenCode config that deny `taskkill /IM ...`, `Stop-Process -Name ...`, `... | Stop-Process`, `pkill` and `killall`, including when wrapped in `powershell -Command "..."`. Stopping a process by its ID (`Stop-Process -Id 1234`) stays allowed, so the agent can still restart its own server. Your own permission rules are kept. Tested with OpenCode 1.19.6.

To also steer the model, add a project `AGENTS.md` that asks it to start dev servers with `Start-Process ... -PassThru`, save the PID, and stop only that PID.
