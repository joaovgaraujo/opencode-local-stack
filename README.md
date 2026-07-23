# opencode-local-stack

Local llama.cpp / rapid-mlx + OpenCode coding-agent stack (Windows, Linux, macOS)

Run a local LLM fully on your own machine, sized to whatever GPU/RAM (or
Apple Silicon unified memory) you actually have, with the coding agent
**OpenCode** as the frontend. Pick from six current small-to-mid models
(35B-A3B down to 4B) across a range of quantizations; a GUI installer
detects your hardware and only shows you the combinations that plausibly fit.

The trick behind the biggest model (Qwen3.6-35B-A3B, 35B total / 3B active):
MoE expert weights live in system RAM (`--cpu-moe`) while attention + the KV
cache stay on the GPU, so a 35B model fits in a few GB of VRAM (measured
2.7–6.0 GB across 8k–256k context on a 12 GB card — see
[`RESULTS.md`](RESULTS.md)). Smaller dense models just run fully on GPU.

One command (`install.py`) detects your hardware, lets you pick a model, and
sets up + validates everything. On Windows/Linux that's llama.cpp serving
GGUF weights; on Apple Silicon Macs it's [rapid-mlx](docs/MACOS.md) serving
MLX weights instead — see [`docs/MACOS.md`](docs/MACOS.md) if that's you,
including an important caveat about what is and isn't verified there.

---

## Quick start

```
git clone https://github.com/joaovgaraujo/opencode-local-stack.git
cd opencode-local-stack
python install.py
```

That's it. `install.py` is cross-platform (Windows, Linux, macOS/Apple
Silicon), stdlib-only (no `pip install` needed for the installer itself), and:

1. **Detects your hardware** — OS, GPU vendor + VRAM (or Apple Silicon
   unified memory), RAM, free disk.
2. **Picks an engine** — llama.cpp (Windows/Linux) or rapid-mlx (macOS/Apple
   Silicon) — and **opens a picker** (GUI if Tkinter is available, else a text
   wizard) showing every model/quantization combination for that engine,
   sorted by whether it *fits*, is a *tight fit*, or *won't fit* on your
   machine.
3. **Downloads** a matching prebuilt llama.cpp release + the GGUF you picked
   (or reuses ones you already have) — on macOS, installs pinned rapid-mlx in
   a project-local virtual environment instead, and rapid-mlx downloads its
   own MLX weights on first run.
4. **Starts the server**, waits for it to become healthy, and runs the validation
   suite (short completion, code-gen, tool-calling, 30k-token
   needle-in-haystack).
5. **Sets up OpenCode** — installs pinned packages, writes `opencode.json`, and
   runs an agentic smoke test (writes + runs a Python file).
6. **Writes `RESULTS.md`** with a pass/fail summary for your machine.

```
python install.py --list-models              # see every model id / quant / size
python install.py --model qwen3.5-9b --profile primary --non-interactive
python install.py --cli                      # force the text wizard (no GUI)
python install.py --skip-tests               # just stand up the server
```

On Linux/NVIDIA, automatic installs use the official Vulkan prebuilt. To use
a CUDA build you compiled yourself, select its real backend explicitly:
```
python install.py --model qwen3.5-4b --non-interactive \
    --backend cuda --bin-dir ./llama.cpp/build/bin
```
Repeat `--extra-server-arg` to preserve fork-specific options in both generated
launchers. Values beginning with `-` must use the equals form, for example
`--extra-server-arg=--cont-batching`; pass a separate value as another token,
for example `--extra-server-arg=--threads --extra-server-arg 16`.

Idempotent — safe to re-run. Node.js (required for OpenCode) is never
installed silently; if it's missing, `install.py` prints the install command
for your OS and stops (or pass `--install-node` on Windows to let it
`winget install` for you).

---

## Pick a model

Sizes below are for each model's Q4_K_M/Q4_K_M-class quant (the installer's
default) at the primary context profile. **RAM/VRAM columns are conservative
estimates from `installer/catalog.py`'s fit heuristic, not measurements**,
except Qwen3.6-35B-A3B, which is the one model actually benchmarked on real
hardware (12 GB-class GPU, 12 GB) — see [`RESULTS.md`](RESULTS.md) for the exact
numbers and [`docs/MODELS.md`](docs/MODELS.md) for how the estimate is
computed. Always re-measure on your own machine with `tests/benchmark.py` or
`tests/vram_logger.ps1` before capacity planning.

| Model | Arch | Default quant | Est. VRAM | Est. RAM | Notes |
|---|---|---:|---:|---:|---|
| **Qwen3.6-35B-A3B** | MoE 35B/3B active | 20.6 GB | ~2.7–6.0 GB (measured, `--cpu-moe`, 8k–256k ctx) | ~21–22 GB (measured RSS) | See [`RESULTS.md`](RESULTS.md) for measured CUDA/Vulkan tok/s and TurboQuant KV-cache results |
| **Gemma 4 26B-A4B** | MoE 26B/4B active | 15.8 GB | ~3.8–4.5 GB (est., by context) | ~19.8 GB (est.) | Same `--cpu-moe` trick as Qwen3.6, lighter RAM footprint |
| **Gemma 4 12B (Unified)** | Dense | 6.6 GB | ~8.1–9.1 GB (est., conservative/primary profile) | ~3 GB (est.) | Fits fully on an 8 GB-class GPU |
| **Qwen3.5-9B** | Dense | 5.3 GB | ~6.8–7.8 GB (est., conservative/primary profile) | ~3 GB (est.) | Largest dense model in this catalog |
| **Gemma 4 E4B** | Dense | 4.6 GB | ~6.1–7.1 GB (est., conservative/primary profile) | ~3 GB (est.) | Smallest Gemma 4 text model |
| **Qwen3.5-4B** | Dense | 2.6 GB | ~4.1–5.1 GB (est., conservative/primary profile) | ~3 GB (est.) | Runs on almost anything, including CPU-only |

MoE VRAM stays roughly constant regardless of quant size (experts live in
system RAM); dense VRAM scales directly with quant size. On macOS each model
also has 4bit/6bit/8bit MLX quants (smaller — e.g. Qwen3.5-4B is 2.8–4.8 GB in
MLX) served by rapid-mlx instead, sharing one unified-memory pool instead of
a separate VRAM/RAM split — see [`docs/MACOS.md`](docs/MACOS.md).

Full quant list, exact file sizes, and how the fit estimate is computed:
[`docs/MODELS.md`](docs/MODELS.md). Standard Q4_K_M weights can also be paired
with TurboQuant's KV-cache compression (`turbo3`/`turbo4`) for a smaller GPU
footprint at long context; a separate experimental TurboQuant *weight* format
(TQ3_1S) is available for Qwen3.6-35B-A3B but was measured far slower for
this MoE model's CPU-offloaded experts — see
[`docs/TURBOQUANT.md`](docs/TURBOQUANT.md) and [`RESULTS.md`](RESULTS.md)
before reaching for either (they need a community llama.cpp fork, not the
stock build the installer downloads).

---

## Good to know

- **These are reasoning models.** They emit `<think>` tokens (in a separate
  `reasoning_content` field) before the final answer — give generous
  `max_tokens` (≥1024) or `content` can come back empty. OpenCode handles this
  natively.
- **VRAM fit is architecture-dependent, tok/s never transfers across
  machines.** MoE models' VRAM footprint is roughly quant-size-independent
  (experts live on CPU); dense models' VRAM scales directly with quant size.
  Generation speed depends on your GPU clocks/thermals/RAM bandwidth —
  re-measure on your own hardware with `tests/validate.py` +
  `tests/vram_logger.ps1` (or an `nvidia-smi -l` loop on Linux).
- **First OpenCode run needs internet** — it downloads `ripgrep` once.
- **Linux + NVIDIA defaults to an official Vulkan prebuilt**, not CUDA,
  because llama.cpp doesn't publish a prebuilt Linux CUDA binary. A custom
  CUDA build is supported with `--backend cuda --bin-dir <build/bin>`; see
  [`docs/DEPLOY.md`](docs/DEPLOY.md#linux--nvidia-vulkan-vs-building-cuda-from-source).
- **`install.py` itself doesn't measure memory.** Use `tests/benchmark.py` (a
  stdlib-only `llama-bench` wrapper that enforces VRAM/RSS caps and reports
  peaks) or `tests/vram_logger.ps1` on Windows, or sample `nvidia-smi` plus
  the server process RSS on Linux while `tests/validate.py` runs; idle
  allocation is not a trustworthy peak.

## Repo layout

```
install.py                    one-shot installer + validator (Windows, Linux, macOS; GUI + CLI)
installer/                    hardware detection, model catalog, download, server lifecycle, wizards
installer/rapidmlx_setup.py   macOS/Apple Silicon engine
opencode.json                 OpenCode -> local endpoint config (regenerated per install)
models/                       drop a GGUF here, or let the installer download it (llama.cpp engine)
tests/validate.py             the 4 functional tests (LLAMA_BASE_URL / LLAMA_MODEL env-driven)
tests/benchmark.py            llama-bench wrapper with VRAM/RSS caps (Linux/Windows, CUDA/Vulkan/etc.)
tests/vram_logger.ps1         Windows VRAM/RAM sampler
docs/MODELS.md                full model catalog + fit-estimate methodology
docs/MACOS.md                 macOS/Apple Silicon (rapid-mlx) — start here if you're on a Mac
docs/TURBOQUANT.md            experimental TurboQuant quant + community fork links
docs/DEPLOY.md                manual usage, GPU backend selection, MoE re-tuning
docs/RESULTS.md               historical validated results (Qwen3.6, primary profile)
```

Generated on first run (git-ignored): `llama.cpp/` (the runtime), `models/*.gguf`
(the weights), `run.ps1` / `run.sh` (restart the last server without the
wizard), `RESULTS.md` (your machine's pass/fail summary).

## Credits

- Models: [unsloth](https://huggingface.co/unsloth) GGUF and
  [mlx-community](https://huggingface.co/mlx-community) MLX quantizations of
  Qwen ([Qwen team, Alibaba](https://github.com/QwenLM)) and
  [Gemma](https://ai.google.dev/gemma) (Google) releases.
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) (Windows/Linux),
  [rapid-mlx](https://github.com/raullenchai/Rapid-MLX) (macOS/Apple Silicon)
- Agent frontend: [OpenCode](https://opencode.ai)
