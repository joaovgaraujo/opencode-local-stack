Fixes found by a full test install of 1.1.0. Download **OpenCodeLocal-Setup.exe** below; it updates an existing install and keeps its models.

## Fixes

- **OpenCode no longer gets stuck retrying.** llama.cpp's default of 4 parallel slots shares one context cache. When OpenCode sent its title and summary requests next to the main one, the server ran out of cache space ("failed to find N available cells in kv cache") and OpenCode retried without end. The server now runs one slot, so the extra requests wait their turn.
- **The installer no longer freezes when an OpenCode test runs too long.** On timeout it killed only the `opencode.cmd` wrapper, and the still-running `opencode.exe` kept the installer waiting. It now ends the whole process tree.
- **The long-context test now fits the context you chose.** It always sent a 30K-token prompt, which failed with HTTP 400 at contexts below 32K.
- **32K is now the smallest context offered.** OpenCode's own instructions and tool definitions take about 10K tokens. Asking for less prints a warning.

## Known limit

Very small models (Qwen3.5-4B) pass the installer's tests but can't finish real OpenCode tasks: they keep calling tools without stopping. The installer only recommends them for GPUs with 6 GB or less; use 8 GB or more of GPU memory for coding with OpenCode.
