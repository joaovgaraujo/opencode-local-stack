# Model catalog

`install.py` picks from a small, hand-curated catalog in `installer/catalog.py`.
Every filename and byte size in it was looked up against the Hugging Face API
(`/api/models/{repo}/tree/main`). Nothing here is guessed. If you add a model,
do the same (`curl https://huggingface.co/api/models/<repo>/tree/main`) rather
than typing a filename from memory; GGUF repos rename/reshuffle quant files
often enough that a guessed name is a real way to 404 mid-install.

## The models

| Model | Arch | Total / active params | Why it's here |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | MoE | 35B / 3B | The model this repo was originally built around, validated end-to-end, see [`RESULTS.md`](RESULTS.md). |
| **Gemma 4 26B-A4B** | MoE | 26B / 4B | Same `--cpu-moe` trick as Qwen3.6, smaller RAM footprint. |
| **Gemma 4 12B (Unified)** | Dense | 12B | Fits fully on an 8GB-class GPU at Q4_K_M. |
| **Gemma 4 E4B** | Dense | ~4B (elastic) | Smallest Gemma 4 text-capable size. |
| **Qwen3.5-4B** | Dense | 4B | Runs on almost anything, including CPU-only. |
| **Qwen3.5-9B** | Dense | 9B | Smallest Qwen that passed the agentic smoke test on a 16 GB Mac. |
| **Qwen3.8-27B** | Dense | 27B | Newest Qwen generation (2026-08); for 24 GB-class GPUs and 32 GB+ Macs. |
| **Nemotron 3.5 Lightning 30B-A3B** | Hybrid Mamba-2/MoE | ~33B / 3B | NVIDIA (2026-08). Few attention layers, so a small KV cache at long context. |
| **LFM2.5-24B-A2B** | MoE | 24B / 2.45B | Liquid AI (2026-10). Its card reports 48.09 on LiveClaw against 38.19 for Qwen3.6-35B-A3B. |
| **Ornith-1.0-9B** | Dense | 9B | deepreinforce-ai agentic-coding fine-tune of Qwen3.5-9B (2026-07), MIT. |
| **Nemotron 3.5 Bolt E4B** | Dense | ~4B (elastic) | NVIDIA post-train of Gemma 4 E4B (2026-10). GGUF only. |

The last five were added on 2026-11-07 and are not yet measured on this stack.
Nemotron 3.5 Bolt E4B has no plain 4/6/8-bit MLX build, so it is hidden on
macOS; Qwen3.8-27B and Nemotron Lightning have 4-bit and 8-bit MLX builds only.

Each model offers 3–5 GGUF quantizations (Q3/Q4/Q5/Q6/Q8-class, all Unsloth
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

- **MoE models** (Qwen3.6-35B-A3B, Gemma 4 26B-A4B, Nemotron 3.5 Lightning,
  LFM2.5-24B-A2B) run with `--cpu-moe`: the
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

The fit check runs against the machine's **total** VRAM and RAM, keeping
**1 GB of VRAM** and **2 GB of RAM** in reserve for the desktop. It
deliberately ignores how much is free at the moment: free memory depends on
what else is open, so ranking on it made the recommended model change between
runs on the same machine. A selection that only meets the raw estimate is
marked **tight**, not **fits**. With `--profile auto` (the default), the
installer chooses the largest context that keeps those reserves.

Free memory is checked separately, just before `llama-server` starts. If
other apps are holding the VRAM or RAM the estimate needs, the installer
prints a warning (the model may spill into shared GPU memory or swap) but
keeps your pick.

For the validated 8 GB-class setup, the recommendation is
**Qwen3.6-35B-A3B Q4_K_M** (`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`) with all experts
in RAM via `--cpu-moe`, q8 KV, and 65,536 context on any GPU with at least
5.5 GB total VRAM. It measured ~4.0 GB server VRAM and ~22.7 GB RAM. Below that,
auto mode selects the 32,768 profile instead. This keeps only the intended MoE experts in RAM; dense layers and KV
remain GPU-resident.

These remain conservative heuristics, not guarantees. `docs/RESULTS.md` shows
what the Qwen3.6 primary profile actually measured on real hardware (~4.0 GB
VRAM, ~22.7 GB RAM); expect similar ballpark numbers for the other MoE model,
and re-measure with `tests/vram_logger.ps1` / a `nvidia-smi -l` loop on your
own machine before trusting a number for capacity planning.

## Sources checked while building this catalog (2026-07)

GGUF (llama.cpp engine):
- [unsloth/Qwen3.6-35B-A3B-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF)
- [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF)
- [unsloth/gemma-4-12b-it-GGUF](https://huggingface.co/unsloth/gemma-4-12b-it-GGUF)
- [unsloth/gemma-4-E4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF)
- [unsloth/Qwen3.5-4B-GGUF](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF)
- [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
- Added 2026-11: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), [unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF), [unsloth/LFM2.5-24B-A2B-GGUF](https://huggingface.co/unsloth/LFM2.5-24B-A2B-GGUF), [unsloth/Ornith-1.0-9B-GGUF](https://huggingface.co/unsloth/Ornith-1.0-9B-GGUF), [unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF](https://huggingface.co/unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF)
- [ggml-org/llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases/latest) (asset names for backend/OS selection)
- [Can't disable thinking in gemma4 (26b-a4b) · llama.cpp Discussion #21338](https://github.com/ggml-org/llama.cpp/discussions/21338). Gemma 4's jinja template has known thinking/tool-call interplay quirks on some builds; if `reasoning_content`/tool calls look wrong, update llama.cpp first.

MLX (rapid-mlx engine, macOS, see [`MACOS.md`](MACOS.md)):
- [mlx-community/Qwen3.6-35B-A3B-4bit](https://huggingface.co/mlx-community/Qwen3.6-35B-A3B-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/gemma-4-26b-a4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-26b-a4b-it-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/gemma-4-12B-it-4bit](https://huggingface.co/mlx-community/gemma-4-12B-it-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/gemma-4-e4b-it-4bit](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/Qwen3.5-4B-4bit](https://huggingface.co/mlx-community/Qwen3.5-4B-4bit) (and `-6bit`/`-8bit`)
- [mlx-community/Qwen3.5-9B-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit) (and `-6bit`/`-8bit`)
- Added 2026-11: [mlx-community/Qwen3.8-27B-4bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit) (and `-8bit`), [mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit](https://huggingface.co/mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit) (and `-8bit`), [LiquidAI/LFM2.5-24B-A2B-MLX-4bit](https://huggingface.co/LiquidAI/LFM2.5-24B-A2B-MLX-4bit) (and `-6bit`/`-8bit`), [mlx-community/Ornith-1.0-9B-4bit](https://huggingface.co/mlx-community/Ornith-1.0-9B-4bit) (and `-6bit`/`-8bit`)
- [rapid-mlx PyPI project metadata](https://pypi.org/pypi/rapid-mlx/json). Used to resolve which of two identically-described GitHub repos is canonical (see `MACOS.md`)
- [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX), canonical source per the PyPI metadata above
