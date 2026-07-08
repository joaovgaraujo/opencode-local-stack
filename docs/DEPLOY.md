# DEPLOY — standing this up on the 8 GB-class laptop GPU (8 GB)

This repo was built and validated on a different Windows PC (a 24 GB desktop GPU), with the config
**constrained to the 8 GB budget**. The heavy artifacts are NOT in git — the installer
downloads them: the llama.cpp CUDA binary is a fat build (arch-native + PTX fallback), and
the GGUF is identical bytes on any machine.

## Fastest path: one command

From this folder on the laptop (after step 1 below installs Node if needed):
```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```
`install.ps1` does every step in this doc automatically — pre-flight, binary/model
check + integrity, start server, run all validation tests + measure VRAM/RAM, set up
OpenCode, run an agentic smoke test, and write `RESULTS.md`.

**It is a self-contained single file.** You can drop *just* `install.ps1` on a bare
laptop with nothing else: it downloads the llama.cpp CUDA binary (GitHub) and the ~20GB
GGUF (Hugging Face) if they're absent, and generates `opencode.json` and `tests\validate.py`
itself. The rest of this bundle (pre-downloaded binary + `models\` slot) just lets it skip
the big downloads — copy the GGUF into `models\` first and it won't re-fetch 20GB. It's idempotent (safe to
re-run) and won't install system packages silently (pass `-InstallNode` / `-InstallPython`
to opt in, else it prints the winget command and stops). Use `-Profile conservative` for the
32768-ctx low-memory profile. The manual steps below are the same thing by hand / for
reference and troubleshooting.

## What's in the repo

```
install.ps1              # ONE-SHOT installer + validator (does everything below)
run.ps1                # PRIMARY profile: 65536 ctx, q8_0 KV, --cpu-moe  (the target)
run-conservative.ps1   # LOW-MEMORY profile: 32768 ctx, more VRAM headroom
opencode.json                 # OpenCode config -> local endpoint, alias qwen3.6-35b-a3b
models/                       # <- DROP THE GGUF HERE (or let the installer download it)
tests/validate.py             # the 4 functional tests
tests/vram_logger.ps1         # sample card VRAM + llama-server RAM to a CSV
docs/RESULTS.md               # build-rig results: flags, pass/fail, VRAM/RAM
docs/DEPLOY.md                # this file
```

Downloaded by the installer (git-ignored, NOT in the repo):
```
llama.cpp/                    # prebuilt llama.cpp CUDA binary (~1.1GB, from GitHub)
models/*.gguf                 # the ~20GB Q4_K_M model (from Hugging Face)
```

Both run scripts take paths as `$PSScriptRoot`-relative **parameters** at the top — nothing
is hardcoded to a user profile. Keep the folder layout and they work with zero edits, or
override `-BinDir` / `-ModelPath`.

## Steps on the 8 GB target

### 1. Install Node.js (only if `node -v` fails) — needed for OpenCode
```powershell
winget install OpenJS.NodeJS.LTS
```
Close and reopen the terminal afterward so `node`/`npm` land on PATH. (Python 3 is only
needed if you want to run `tests\validate.py`: `winget install Python.Python.3.12`.)

### 2. Drop in the model
Copy the ~20.6 GB GGUF you already downloaded on the main PC into the repo's `models\`:
```
models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
```
(Copying it avoids re-downloading 20 GB. If you'd rather re-fetch:
`curl -L -o models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`)

### 3. Start the server
```powershell
cd opencode-local-stack
powershell -ExecutionPolicy Bypass -File .\run.ps1
```
The script prints the full flag line first. Confirm it's up (new terminal):
```powershell
curl http://127.0.0.1:8080/health          # {"status":"ok"}
curl http://127.0.0.1:8080/v1/models        # id/alias: qwen3.6-35b-a3b
```
If 8 GB ever feels tight (busy desktop, other GPU apps), use the low-memory profile:
```powershell
powershell -ExecutionPolicy Bypass -File .\run-conservative.ps1   # 32768 ctx
```

### 4. Point OpenCode at it
```powershell
npm install -g opencode-ai@latest @ai-sdk/openai-compatible
mkdir "$env:USERPROFILE\.config\opencode" -Force
copy .\opencode.json "$env:USERPROFILE\.config\opencode\opencode.json"
```
**First `opencode` run needs internet** — it downloads ripgrep once from GitHub. (This
silently hung a headless run during testing until ripgrep finished; if the first launch
seems stuck, it's the ripgrep fetch, not the model.) Then in a project folder:
```powershell
opencode                                   # interactive; /models -> pick the local one
# or headless:
opencode run --auto -m llamacpp/qwen3.6-35b-a3b "create hello.py that prints hi, run it"
```

### 5. (Optional) re-verify on the laptop
```powershell
python .\tests\validate.py                 # all 4 tests should PASS
```

## What transfers from the rig — and what doesn't

**WILL transfer (same 8GB-class card, same-arch kernels, same bytes):**
- Correctness: short completion, code-gen, tool calling, 30k needle retrieval — all PASS.
- Tool calling / `--jinja` chat template behavior.
- **VRAM fit**: ~4.0 GB (primary) / ~3.3 GB (conservative) llama-server footprint. With the
  8 GB target's ~1–1.5 GB desktop compositor, expect ~5.5 GB total — inside 8 GB.

**WON'T transfer — RE-MEASURE on the 8 GB target:**
- **tok/s.** The rig hit ~47 tok/s fresh / ~44 @30k, prompt eval ~586 tok/s. The laptop has
  different GPU clocks, thermal/power limits, and RAM bandwidth (experts run on CPU, so RAM
  bandwidth matters a lot). Your numbers will differ — measure with `tests\vram_logger.ps1`
  running during `tests\validate.py`.

## Re-tune expert offload ON THE 8 GB target (not on any other machine)

The scripts ship with `--cpu-moe` (ALL experts in RAM = lowest VRAM, safe first boot).
Because the primary profile only uses ~4.0 GB of the 8 GB card, there's headroom to move
some experts onto the GPU for more speed. **Do this on the real 8 GB target — the optimal offload
depends on its actual 8 GB, not any other GPU.**

1. Start from safe: `--cpu-moe` (edit nothing).
2. Binary-search `-NCpuMoe N` **upward** (fewer layers forced to CPU = more experts on GPU):
   ```powershell
   .\run.ps1 -NCpuMoe 36      # then 32, then 28, ...
   ```
   Lower N = more experts on GPU = more VRAM used + faster.
3. While generating at ~full context, watch VRAM:
   ```powershell
   nvidia-smi --query-gpu=memory.used --format=csv -l 2
   ```
   Stop when peak card usage reaches **~6.5 GB** (leaves ~1.5 GB for the desktop). Keep the
   last stable value.
4. If it OOMs or spills to shared memory (Task Manager "Shared GPU memory" > 0, throughput
   craters): back off — raise N toward `--cpu-moe`, then if needed drop ctx (65536→32768),
   then KV to q4_0 (`-KvType q4_0`), in that order.
5. Recommended: NVIDIA Control Panel → Manage 3D settings → **CUDA - Sysmem Fallback
   Policy = Prefer No Sysmem Fallback**, so it OOMs loudly instead of silently running at
   1/10 speed off shared memory.

## Notes
- The load log warns *"tensor overrides to CPU with mmap enabled — consider --no-mmap"*.
  mmap is fine and safer with sufficient RAM (experts stay page-cached). If prompt ingestion
  feels I/O-stalled, add `--no-mmap` to the run script's arg list (forces experts fully
  resident, ~20 GB anonymous RAM).
- One server instance at a time — kill the old one before restarting with new flags:
  `Get-Process llama-server | Stop-Process -Force`.
- Frontend is **OpenCode only**. Open WebUI was intentionally not set up.
