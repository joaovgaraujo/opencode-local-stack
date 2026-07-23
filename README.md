# opencode-local-stack

Run a local LLM on your own machine with the [OpenCode](https://opencode.ai)
coding agent as the frontend. Works on Windows, Linux, and macOS/Apple Silicon.
One command detects your hardware, downloads a model that fits, serves it, and
validates the whole stack.

The catalog has six small-to-mid models, from Qwen3.6-35B-A3B down to
Qwen3.5-4B, each in several quantizations. The installer only shows the
combinations that fit your GPU or unified memory.

Qwen3.6-35B-A3B is a mixture-of-experts model: 35B total parameters, 3B active.
Its expert weights sit in system RAM (`--cpu-moe`) while attention and the KV
cache stay on the GPU, so it runs in a few GB of VRAM. Measured 2.7 to 6.0 GB
across 8k to 256k context on a 12 GB card (see
[`docs/RESULTS.md`](docs/RESULTS.md)). The smaller dense models run entirely on
the GPU.

On Windows and Linux the engine is llama.cpp serving GGUF weights. On Apple
Silicon it's rapid-mlx serving MLX weights. The macOS path is validated
end-to-end on Apple Silicon and defaults to native TurboQuant K8V4 KV-cache
compression. See [`docs/MACOS.md`](docs/MACOS.md).

## Quick start

```
git clone https://github.com/joaovgaraujo/opencode-local-stack.git
cd opencode-local-stack
python install.py
```

`install.py` is cross-platform and needs no `pip install` of its own. It:

1. Detects your hardware: OS, GPU and VRAM (or Apple Silicon unified memory),
   RAM, free disk.
2. Opens a picker that shows only the model/quant combinations that fit, best
   fit first. The GUI installs its own prerequisites (Tkinter on Homebrew
   Python, and Node.js if you leave the box checked); headless systems fall
   back to a text wizard.
3. Downloads a matching llama.cpp release and the GGUF, or on macOS installs
   pinned rapid-mlx in a project-local venv and lets it fetch MLX weights on
   first run.
4. Starts the server, waits for health, and runs the validation suite: short
   completion, code generation, tool calling, and a 30k-token needle test.
5. Installs OpenCode, writes `opencode.json`, and runs an agentic smoke test
   that writes and runs a Python file.
6. Writes `RESULTS.md` for your machine.

```
python install.py --list-models              # every model id, quant, and size
python install.py --model qwen3.5-9b --non-interactive
python install.py --cli                      # text wizard, no GUI
python install.py --skip-tests               # just start the server
```

Node.js is required for OpenCode and is never installed without your say-so.
The GUI checkbox and the `--install-node` flag opt into a first-party
installer: winget on Windows, Homebrew on macOS, and on Linux the official
nodejs.org tarball extracted into a project-local `./node` directory (no sudo,
nothing touches the system - delete the directory to undo).

`install.py` is idempotent, so re-running it is safe.

## Pick a model

The RAM and VRAM columns below are conservative estimates from the fit
heuristic in `installer/catalog.py`, not measurements. The exception is
Qwen3.6-35B-A3B, which was measured on real hardware. See
[`docs/RESULTS.md`](docs/RESULTS.md) for the llama.cpp `--cpu-moe` run and
[`docs/TURBOQUANT.md`](docs/TURBOQUANT.md) for the 12 GB-class CUDA run, plus
[`docs/MODELS.md`](docs/MODELS.md) for how the estimate is computed.
Re-measure on your own machine with `tests/benchmark.py` before capacity
planning.

| Model | Arch | Default quant | Est. VRAM | Est. RAM | Notes |
|---|---|---:|---:|---:|---|
| **Qwen3.6-35B-A3B** | MoE 35B/3B active | 20.6 GB | 2.7–6.0 GB (measured, `--cpu-moe`, 8k–256k ctx) | 21–22 GB (measured RSS) | See [`docs/RESULTS.md`](docs/RESULTS.md) for measured tok/s and [`docs/TURBOQUANT.md`](docs/TURBOQUANT.md) for TurboQuant results |
| **Gemma 4 26B-A4B** | MoE 26B/4B active | 15.8 GB | 3.8–4.5 GB (est.) | 19.8 GB (est.) | Same `--cpu-moe` trick, lighter RAM footprint |
| **Gemma 4 12B (Unified)** | Dense | 6.6 GB | 8.1–9.1 GB (est.) | ~3 GB (est.) | Fits fully on an 8 GB-class GPU |
| **Qwen3.5-9B** | Dense | 5.3 GB | 6.8–7.8 GB (est.) | ~3 GB (est.) | Largest dense model in the catalog |
| **Gemma 4 E4B** | Dense | 4.6 GB | 6.1–7.1 GB (est.) | ~3 GB (est.) | Smallest Gemma 4 text model |
| **Qwen3.5-4B** | Dense | 2.6 GB | 4.1–5.1 GB (est.) | ~3 GB (est.) | Runs on almost anything, including CPU-only |

MoE VRAM stays roughly flat across quant sizes because the experts live in
system RAM. Dense VRAM scales with quant size. On macOS each model also ships
4/6/8-bit MLX quants served from one unified-memory pool instead of a separate
VRAM/RAM split (see [`docs/MACOS.md`](docs/MACOS.md)).

On Apple Silicon the picker chooses by memory tier: qwen3.5-4b at 8 GB,
qwen3.5-9b at 16 GB, gemma-4-12b at 24 GB, qwen3.6-35b-a3b at 32 GB and up (the
tier table is in [`docs/MACOS.md`](docs/MACOS.md)). Fitting in memory is not the
same as driving OpenCode: measured on a 16 GB M4, qwen3.5-9b is the smallest
Qwen that completes the agentic smoke test (2B and 4B print code instead of
calling tools, even at 8-bit), while Gemma's edge models tool-call at 2B. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#local-measurements-on-a-16-gb-apple-m4)
for the per-model tok/s, context ceilings, and pass/fail.

Full quant list and exact file sizes: [`docs/MODELS.md`](docs/MODELS.md).
Independent benchmark scores per model (coding, reasoning, tool use):
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Measured throughput, 8 GB context
fit, and CUDA-vs-Vulkan speed for all six models on one machine:
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).
TurboQuant KV-cache compression and the experimental TQ3_1S weight format both
need a community llama.cpp fork, so read [`docs/TURBOQUANT.md`](docs/TURBOQUANT.md)
and [`docs/RESULTS.md`](docs/RESULTS.md) first.

## Good to know

- These are reasoning models. They emit `<think>` tokens in a separate
  `reasoning_content` field before the answer, so give them `max_tokens` of
  1024 or more or `content` can come back empty. OpenCode handles this
  natively.
- tok/s never transfers between machines. It depends on your GPU clocks,
  thermals, and RAM bandwidth. Re-measure with `tests/validate.py` and
  `tests/vram_logger.ps1`, or an `nvidia-smi -l` loop on Linux.
- The first OpenCode run downloads `ripgrep` once and needs internet.
- Linux with NVIDIA defaults to the official Vulkan prebuilt, because llama.cpp
  ships no prebuilt Linux CUDA binary. For a CUDA build you compiled yourself,
  pass `--backend cuda --bin-dir <build/bin>` (see
  [`docs/DEPLOY.md`](docs/DEPLOY.md#linux--nvidia-vulkan-vs-building-cuda-from-source)).
- `install.py` doesn't measure memory itself. Use `tests/benchmark.py` (a
  stdlib `llama-bench` wrapper that enforces VRAM/RSS caps) or sample
  `nvidia-smi` and the server RSS while the tests run. Idle allocation is not a
  trustworthy peak.

## Repo layout

```
install.py                    one-shot installer + validator (Windows, Linux, macOS; GUI + CLI)
installer/                    hardware detection, model catalog, download, server lifecycle, wizards
installer/rapidmlx_setup.py   macOS/Apple Silicon engine
opencode.json                 OpenCode -> local endpoint config (regenerated per install)
models/                       drop a GGUF here, or let the installer download it (llama.cpp engine)
tests/validate.py             the 4 functional tests (LLAMA_BASE_URL / LLAMA_MODEL env-driven)
tests/benchmark.py            llama-bench wrapper with VRAM/RSS caps (Linux/Windows, CUDA/Vulkan)
tests/vram_logger.ps1         Windows VRAM/RAM sampler
docs/MODELS.md                full model catalog + fit-estimate methodology
docs/BENCHMARKS.md            independent benchmark scores per model (coding, reasoning, tool use)
docs/PERFORMANCE.md           measured tok/s, 8 GB context fit, CUDA vs Vulkan (this machine)
docs/MACOS.md                 macOS/Apple Silicon (rapid-mlx); start here if you're on a Mac
docs/TURBOQUANT.md            experimental TurboQuant quant + community fork links
docs/DEPLOY.md                manual usage, GPU backend selection, MoE re-tuning
```

Generated on first run and git-ignored: `llama.cpp/` (the runtime),
`models/*.gguf` (the weights), `run.ps1` / `run.sh` (restart the last server
without the wizard), and `RESULTS.md` (your machine's pass/fail summary).

## Credits

- Models: [unsloth](https://huggingface.co/unsloth) GGUF and
  [mlx-community](https://huggingface.co/mlx-community) MLX quantizations of
  Qwen ([Qwen team, Alibaba](https://github.com/QwenLM)) and
  [Gemma](https://ai.google.dev/gemma) (Google) releases.
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) on Windows and
  Linux, [rapid-mlx](https://github.com/raullenchai/Rapid-MLX) on Apple Silicon.
- Agent frontend: [OpenCode](https://opencode.ai).
