# opencode-local-stack

Local llama.cpp / rapid-mlx + OpenCode coding-agent stack (Windows, Linux, macOS)

Run a local LLM fully on your own machine, sized to whatever GPU/RAM (or
Apple Silicon unified memory) you actually have, with the coding agent
**OpenCode** as the frontend. Pick from six current small-to-mid models
(35B-A3B down to 4B) across a range of quantizations; a GUI installer
detects your hardware and only shows you the combinations that plausibly fit.

The trick behind the biggest model (Qwen3.6-35B-A3B, 35B total / 3B active):
MoE expert weights live in system RAM (`--cpu-moe`) while attention + the KV
cache stay on the GPU, so a 35B model fits in ~4 GB of VRAM. Smaller dense
models just run fully on GPU.

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
   (or reuses ones you already have) — on macOS, installs rapid-mlx instead,
   which downloads its own MLX weights on first run.
4. **Starts the server**, waits for it to become healthy, runs the validation
   suite (short completion, code-gen, tool-calling, 30k-token
   needle-in-haystack), and logs peak VRAM/RAM.
5. **Sets up OpenCode** — installs it, writes `opencode.json`, runs an
   agentic smoke test (writes + runs a Python file).
6. **Writes `RESULTS.md`** with a pass/fail summary for your machine.

```
python install.py --list-models              # see every model id / quant / size
python install.py --model qwen3.5-9b --profile primary --non-interactive
python install.py --cli                      # force the text wizard (no GUI)
python install.py --skip-tests               # just stand up the server
```

Idempotent — safe to re-run. Node.js (required for OpenCode) is never
installed silently; if it's missing, `install.py` prints the install command
for your OS and stops (or pass `--install-node` on Windows to let it
`winget install` for you).

---

## Pick a model

| Model | Arch | Size | Good for |
|---|---|---|---|
| **Qwen3.6-35B-A3B** | MoE 35B/3B active | 15.7–34.4 GB (quant-dependent) | Best quality; needs ~24 GB free RAM, only ~4 GB VRAM |
| **Gemma 4 26B-A4B** | MoE 26B/4B active | 12–25 GB | Same trick, lighter RAM footprint |
| **Gemma 4 12B (Unified)** | Dense | 6.6–11.8 GB | Fits fully on an 8 GB-class GPU |
| **Qwen3.5-9B** | Dense | 5.3–8.9 GB | Alibaba reports it beating much larger models on reasoning benchmarks |
| **Gemma 4 E4B** | Dense | 4.6–7.6 GB | Smallest Gemma 4 text model |
| **Qwen3.5-4B** | Dense | 2.6–4.2 GB | Runs on almost anything, including CPU-only |

Sizes above are GGUF (llama.cpp/Windows/Linux); on macOS each model also has
4bit/6bit/8bit MLX quants (smaller — e.g. Qwen3.5-4B is 2.8–4.8 GB in MLX)
served by rapid-mlx instead — see [`docs/MACOS.md`](docs/MACOS.md).

Full quant list, exact file sizes, and how the fit estimate is computed:
[`docs/MODELS.md`](docs/MODELS.md). An experimental TurboQuant quant is also
available for Qwen3.6-35B-A3B — see [`docs/TURBOQUANT.md`](docs/TURBOQUANT.md)
before reaching for it (it needs a community llama.cpp fork, not the stock
build the installer downloads).

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
- **Linux + NVIDIA defaults to a Vulkan build**, not CUDA — llama.cpp doesn't
  publish a prebuilt Linux CUDA binary. See
  [`docs/DEPLOY.md`](docs/DEPLOY.md#linux--nvidia-vulkan-vs-building-cuda-from-source)
  for building CUDA from source if you want the extra performance.

## Repo layout

```
install.py                    one-shot installer + validator (Windows, Linux, macOS; GUI + CLI)
installer/                    hardware detection, model catalog, download, server lifecycle, wizards
installer/rapidmlx_setup.py   macOS/Apple Silicon engine
opencode.json                 OpenCode -> local endpoint config (regenerated per install)
models/                       drop a GGUF here, or let the installer download it (llama.cpp engine)
tests/validate.py             the 4 functional tests (LLAMA_BASE_URL / LLAMA_MODEL env-driven)
tests/vram_logger.ps1         Windows VRAM/RAM sampler
docs/MODELS.md                full model catalog + fit-estimate methodology
docs/MACOS.md                 macOS/Apple Silicon (rapid-mlx) — start here if you're on a Mac
docs/TURBOQUANT.md            experimental TurboQuant quant + community fork links
docs/DEPLOY.md                manual usage, GPU backend selection, MoE re-tuning
docs/RESULTS.md               historical validated results (Qwen3.6, primary profile)
```

Generated on first run (git-ignored): `llama.cpp/` (the runtime), `models/*.gguf`
(the weights), `run.ps1` / `run.sh` (restart the last server without the
wizard), `RESULTS.md` (your machine's pass/fail + measured VRAM/RAM).

## Credits

- Models: [unsloth](https://huggingface.co/unsloth) GGUF and
  [mlx-community](https://huggingface.co/mlx-community) MLX quantizations of
  Qwen ([Qwen team, Alibaba](https://github.com/QwenLM)) and
  [Gemma](https://ai.google.dev/gemma) (Google) releases.
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) (Windows/Linux),
  [rapid-mlx](https://github.com/raullenchai/Rapid-MLX) (macOS/Apple Silicon)
- Agent frontend: [OpenCode](https://opencode.ai)
