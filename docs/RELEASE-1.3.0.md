Download **OpenCodeLocal-Setup.exe** (Windows) or run the Linux/macOS one-line installer; existing installs keep their models and settings.

## OpenCode 2.3.2

OpenCode 2.x is released only as programs on GitHub, not on npm, where `opencode-ai` stops at 1.19.6. The installer now downloads OpenCode 2.3.2 from its GitHub release, checks it against the release's SHA256SUMS, and puts it in the install folder.

- **No Node.js needed for OpenCode.** If the download fails, the installer falls back to npm `opencode-ai@1.19.6`.
- **One OpenCode.** A plain `opencode` command now runs 2.3.2 too. The old npm `opencode-ai` 1.x is removed when it isn't running, because 1.x can't see sessions created by 2.x.
- **Tested with OpenCode 2.3.2:**
  - The local model and the existing OpenCode config work.
  - The kill-by-name deny rules from 1.2.5 are still enforced.
  - Sessions from 1.19 are listed and can be continued.

What changed in OpenCode 2.3 is in its release notes: https://github.com/anomalyco/opencode/releases/tag/v2.3.0

## Fixes since 1.2.5

- `opencode-local` is also installed into the per-user WindowsApps folder, so it is found in terminals that started before Node.js was installed.
