Download **OpenCodeLocal-Setup.exe** (Windows) or run the Linux/macOS one-line installer below; existing installs keep their models and settings.

## Fixes

- **Mouse codes at the prompt after OpenCode closes.** If OpenCode was aborted or crashed, it could leave the terminal's mouse tracking on, so every mouse move printed text such as `[<35;78;1M` at the prompt. `opencode-local` now saves the terminal's state before starting OpenCode and restores it afterwards, however OpenCode exits.

## Updates

- `@ai-sdk/openai-compatible` 3.0.94 (OpenCode 1.19.6, llama.cpp b12359 and Node.js 26.13.1 were already the newest).
- rapid-mlx stays at 1.0.5 on macOS. 1.1.0, released 2026-11-21, changes `rapid-mlx serve` into a supervised parent and runner process, and hasn't been tested with this installer yet.

## Tip

Run one OpenCode per project folder. Two OpenCode windows in the same folder share the model server and the project's session data, and get in each other's way.
