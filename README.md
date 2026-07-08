# Qwen3.6-35B-A3B on an 8 GB-class GPU (8 GB) — local llama.cpp + OpenCode

Run the **Qwen3.6-35B-A3B** MoE model (35B total, ~3B active) fully locally on an **8 GB
laptop GPU**, with the coding agent **OpenCode** as the frontend. The trick: all the MoE
expert weights live in system RAM (`--cpu-moe`) while attention + the KV cache stay on the
GPU, so the whole thing fits in ~4 GB of VRAM at a 64k context.

One PowerShell script (`install.ps1`) sets up and validates everything.

---

## Target hardware

| | |
|---|---|
| GPU | 8 GB-class laptop GPU, **8 GB** (the target compute cap) — any recent NVIDIA card works |
| RAM | sufficient RAM (needs ~24 GB free; ~20 GB holds the experts) |
| Disk | ~25 GB free (for the model) |
| OS | Windows 10/11 |
| Model | `unsloth/Qwen3.6-35B-A3B-GGUF` → `Q4_K_M` (~20.6 GB) |
| Server | llama.cpp `llama-server` (prebuilt CUDA binary, auto-downloaded) |
| Frontend | **OpenCode** only |

> Built and validated on a stand-in test GPU rig, with the config **deliberately
> constrained to the 8 GB budget**. See [`docs/RESULTS.md`](docs/RESULTS.md).

---

## Quick start

```powershell
git clone <this-repo-url> opencode-local-stack
cd opencode-local-stack

# (optional) drop an already-downloaded GGUF into .\models\ to skip the 20 GB download
# otherwise the installer downloads it for you

powershell -ExecutionPolicy Bypass -File .\install.ps1
```

That's it. The installer is a **self-contained single file** — on a bare machine it
downloads the llama.cpp binary + the model and generates its own config/tests. When it
finishes you'll have a local OpenAI-compatible endpoint at `http://127.0.0.1:8080/v1`
(model alias `qwen3.6-35b-a3b`) and OpenCode wired to it.

### What `install.ps1` does

1. **Pre-flight** — NVIDIA GPU, VRAM, free RAM, free disk.
2. **Binary** — uses `.\llama.cpp\` if present, else downloads the prebuilt CUDA release.
3. **Model** — uses `.\models\*.gguf` (or `-ModelPath`), else downloads it; verifies GGUF magic + size.
4. **Server** — starts `llama-server` with the target profile; waits for `/health`; checks the alias.
5. **Validation** — short completion, code-gen (`ast.parse`), tool-calling, 30k-token needle-in-haystack; logs **peak VRAM/RAM**.
6. **OpenCode** — installs it, writes `opencode.json`, runs an agentic smoke test (writes + runs a Python file).
7. **Report** — writes `RESULTS.md` and prints a pass/fail summary.

Idempotent — safe to re-run. It will **not** install Node/Python silently; if they're
missing it prints the `winget` command and stops (or pass `-InstallNode` / `-InstallPython`).

### Useful flags

```powershell
.\install.ps1 -Profile conservative   # 32768 ctx, lower VRAM/RAM headroom
.\install.ps1 -InstallNode            # let it winget-install Node LTS
.\install.ps1 -SkipTests              # just stand up the server
.\install.ps1 -ModelPath D:\path\to\model.gguf   # reuse an existing GGUF
```

---

## Manual usage (without the installer)

```powershell
# start the server (primary 65536-ctx profile)
powershell -ExecutionPolicy Bypass -File .\run.ps1
# ...or the low-memory profile
powershell -ExecutionPolicy Bypass -File .\run-conservative.ps1

# point OpenCode at it
npm install -g opencode-ai@latest @ai-sdk/openai-compatible
mkdir "$env:USERPROFILE\.config\opencode" -Force
copy .\opencode.json "$env:USERPROFILE\.config\opencode\opencode.json"
opencode      # then /models -> Qwen3.6-35B-A3B (local)
```

The run scripts take `-BinDir` / `-ModelPath` / `-Port` / `-CtxSize` overrides; defaults are
relative to the script folder so the whole thing is portable.

---

## Validated results (on the build rig)

| Test | Result |
|---|---|
| Short completion | ✅ PASS |
| Code gen + `ast.parse` | ✅ PASS |
| Tool calling (`--jinja`) | ✅ PASS |
| 30k-token needle @ 60% | ✅ PASS |
| VRAM fit ≤ 6.5 GB | ✅ **~4.0 GB** (primary), ~3.3 GB (conservative) |
| OpenCode agentic | ✅ PASS |

Full numbers, flags, and methodology: [`docs/RESULTS.md`](docs/RESULTS.md).
Laptop stand-up, what transfers, and `--n-cpu-moe` re-tuning: [`docs/DEPLOY.md`](docs/DEPLOY.md).

---

## Good to know

- **Qwen3.6 is a reasoning model.** It emits `<think>` tokens (in a separate
  `reasoning_content` field) before the final answer — give it generous `max_tokens`
  (≥1024) or `content` can come back empty. OpenCode handles this natively.
- **VRAM fit transfers, tok/s does not.** The 8 GB footprint is card-independent (experts
  are on CPU), but generation speed depends on the laptop's clocks/thermals/RAM bandwidth —
  re-measure there. On the real 8 GB target you can push `--n-cpu-moe N` upward to use the spare
  VRAM headroom for more speed (see `docs/DEPLOY.md`).
- **First OpenCode run needs internet** — it downloads `ripgrep` once.

## Repo layout

```
install.ps1              one-shot installer + validator (self-contained)
run.ps1                primary server profile (65536 ctx, q8_0 KV, --cpu-moe)
run-conservative.ps1   low-memory server profile (32768 ctx)
opencode.json                 OpenCode -> local endpoint config
models/                       drop the GGUF here (git-ignored)
tests/validate.py             the 4 functional tests
tests/vram_logger.ps1         VRAM/RAM sampler
docs/DEPLOY.md                laptop deployment + re-tuning guide
docs/RESULTS.md               full validation results
```

## Credits

- Model: [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp)
- Agent frontend: [OpenCode](https://opencode.ai)
