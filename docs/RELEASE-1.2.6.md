Download **OpenCodeLocal-Setup.exe** (Windows); existing installs keep their models and settings.

## Fix

- **`opencode-local` not recognized in some terminals.** The command was installed only into npm's folder. A terminal that started with a PATH from before Node.js was installed (seen with Windows Terminal) couldn't find it. It is now also installed into the per-user WindowsApps folder, which has been on PATH since Windows was set up. The uninstaller already removes both.
