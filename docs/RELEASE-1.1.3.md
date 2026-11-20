Download **OpenCodeLocal-Setup.exe** below; it updates an existing install and keeps its models.

## The model unloads when idle

The model server keeps running after OpenCode closes, so the next start is fast. Until now it also kept the model in GPU memory and RAM until you ran `opencode-local --stop`.

It now unloads the model after 10 idle minutes, using llama.cpp's `--sleep-idle-seconds`. The next request loads it again. Measured with Qwen3.6-35B-A3B at 262K context on an RTX 4090:

| | GPU memory in use | Server RAM |
|---|---|---|
| Loaded | 23.6 GB | 6.0 GB |
| Asleep | 2.1 GB (desktop only) | 0.6 GB |

Waking up and answering took 20 seconds with the model file still in Windows' file cache; from disk it takes longer.

Change it with `opencode-local --idle-minutes N`, and add `--save` to keep it. `0` keeps the model loaded. The installer checks that the llama.cpp build supports the option and leaves it out when it doesn't.
