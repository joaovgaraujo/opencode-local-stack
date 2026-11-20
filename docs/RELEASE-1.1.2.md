Download **OpenCodeLocal-Setup.exe** below; it updates an existing install and keeps its models.

## Fixes

- **"'opencode' is not recognized" after closing OpenCode.** Terminals opened from the right-click menu or the Start menu could start with a PATH from before Node.js was installed, so the `opencode -s <session>` line OpenCode prints on exit failed there. `opencode-local` now adds npm's folder to the window's PATH when `opencode` isn't found, so that line works.
- When OpenCode closes, `opencode-local` now prints how to continue:
  - `opencode-local -c` continues the last session
  - `opencode-local -s <session id>` continues a given one
