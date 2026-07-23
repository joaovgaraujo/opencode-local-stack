# Model catalog

`install.py` picks from a small, hand-curated catalog in `installer/catalog.py`.
Every filename and byte size in it was looked up against the Hugging Face API
(`/api/models/{repo}/tree/main`). Nothing here is guessed. If you add a model,
do the same (`curl https://huggingface.co/api/models/<repo>/tree/main`) rather
than typing a filename from memory; GGUF repos rename/reshuffle quant files
often enough that a guessed name is a real way to 404 mid-install.

## The six models

| Model | Arch | Total / active params | Why it's here |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | MoE | 35B / 3B | The model this repo was originally built around, validated end-to-end, see [`RESULTS.md`](RESULTS.md). |
| **Gemma 4 26B-A4B** | MoE | 26B / 4B | Same `--cpu-moe` trick as Qwen3.6, smaller RAM footprint. |
| **Gemma 4 12B (Unified)** | Dense | 12B | Fits fully on an 8GB-class GPU at Q4_K_M. |
| **Gemma 4 E4B** | Dense | ~4B (elastic) | Smallest Gemma 4 text-capable size. |
| **Qwen3.5-4B** | Dense | 4B | Runs on almost anything, including CPU-only. |
| **Qwen3.5-9B** | Dense | 9B | Largest dense model in this catalog. |

Each model offers 3–4 GGUF quantizations (Q3/Q4/Q5/Q6/Q8-class, all Unsloth
Dynamic quants) for the llama.cpp engine, **and** 3 MLX quantizations
(4bit/6bit/8bit, from `mlx-community`) for the rapid-mlx engine (macOS/Apple
Silicon only, see [`MACOS.md`](MACOS.md)). `python install.py --list-models`
prints the full list for both, with exact file/repo names and sizes.

## Two engines, two serving strategies

`hwdetect.pick_engine()` picks between them automatically: `rapidmlx` on
Apple Silicon Macs, `llamacpp` everywhere else. See [`MACOS.md`](MACOS.md)
for the rapid-mlx side (unified memory, no VRAM/RAM split; validated
end-to-end on a 16 GB Apple M4, with measured footprints and context ceilings).
The rest of this doc covers the llama.cpp engine's two serving strategies:

- **MoE models** (Qwen3.6-35B-A3B, Gemma 4 26B-A4B) run with `--cpu-moe`: the
  expert tensors live in system RAM, attention + KV cache stay on the GPU.
  VRAM use stays roughly constant (~3.5–4.5 GB) **regardless of quant size**;
  only RAM needs to hold the bigger file. This is what makes a 35B model
  usable on an 8 GB card.
- **Dense models** (everything else) run fully on GPU (`--n-gpu-layers 999`).
  VRAM has to hold the whole quant file plus the KV cache, so a bigger quant
  directly costs more VRAM.

## How the fit estimate works

`catalog.fit_verdict()` classifies each (model, quant, profile) as **fits**,
**tight**, or **won't fit**, using:

- MoE: `VRAM need ≈ 3.4–4.5 GB` (by context size) · `RAM need ≈ quant size + 4 GB`
- Dense: `VRAM need ≈ quant size + 1.5–2.5 GB` (by context size) · `RAM need ≈ 3 GB`

The fit check keeps **1 GB of detected free VRAM** and **2 GB of free RAM**
in reserve. A selection that only meets the raw estimate is marked **tight**,
not **fits**, so the default never relies on shared-memory fallback or paging.
With `--profile auto` (the default), the installer chooses the largest context
that retains those reserves.

For the validated 8 GB-class setup, the recommendation is
**Qwen3.6-35B-A3B Q4_K_M** (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`) with all experts
in RAM via `--cpu-moe`, q8 KV, and 65,536 context when at least 5.5 GB VRAM is
free. It measured ~4.0 GB server VRAM and ~22.7 GB RAM. If current free VRAM
falls below the reserve threshold, auto mode selects the 32,768 profile
instead. This keeps only the intended MoE experts in RAM; dense layers and KV
remain GPU-resident.

These remain conservative heuristics, not guarantees. `docs/RESULTS.md` shows
what the Qwen3.6 primary profile actually measured on real hardware (~4.0 GB
VRAM, ~22.7 GB RAM); expect similar ballpark numbers for the other MoE model,
and re-measure with `tests/vram_logger.ps1` / a `nvidia-smi -l` loop on your
own machine before trusting a number for capacity planning.

## TurboQuant

One experimental TurboQuant quant is listed for Qwen3.6-35B-A3B (weights
verified at `mad-lab-ai/Qwen3.6-35B-A3B-tq-gguf`). It needs a community
llama.cpp fork with TurboQuant kernel support; the official releases don't
understand the format. See [`TURBOQUANT.md`](TURBOQUANT.md) before picking it.
No verified TurboQuant GGUF weights were found for the other models at the
time this catalog was written, so none are listed. Don't assume one exists
just because the technique applies in principle.

## Sources checked while building this catalog (2026-07)

GGUF (llama.cpp engine):
- [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
- [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)
- [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
- [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
- [mad-lab-ai/Qwen3.6-35B-A3B-tq-gguf](https://huggingface.co/mad-lab-ai/Qwen3.6-35B-A3B-tq-gguf)
- [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases/latest) (asset names for backend/OS selection)
- [Can't disable thinking in gemma4 (26b-a4b) · llama.cpp Discussion #21338](https://github.com/ggml-org/llama.cpp/discussions/21338). Gemma 4's jinja template has known thinking/tool-call interplay quirks on some builds; if `reasoning_content`/tool calls look wrong, update llama.cpp first.

MLX (rapid-mlx engine, macOS, see [`MACOS.md`](MACOS.md)):
- [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/gemma-4-12B-it-4bit](https://huggingface.co/mlx-community/gemma-4-12B-it-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/gemma-4-e4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/Qwen3.5-4B-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) (and `-6bit`/`-8bit`)
- [rapid-mlx PyPI project metadata](https://pypi.org/pypi/rapid-mlx/json). Used to resolve which of two identically-described GitHub repos is canonical (see `MACOS.md`)
- [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), canonical source per the PyPI metadata above
