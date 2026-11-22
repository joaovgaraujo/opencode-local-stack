# opencode-local-stack

Run a local LLM on your own machine with the [OpenCode](https://opencode.ai)
coding agent as the frontend. Works on Windows, Linux, and macOS/Apple Silicon.
The installer detects your hardware, downloads a model that fits, serves it
with llama.cpp, and tests the whole stack, including an OpenCode run that
writes and executes a file.

Models are matched to your machine's total VRAM and RAM. What happens to be
free when you run the installer doesn't count: a browser or the file cache
gives memory back when the model loads, so the same PC always gets the same
recommendation.

Qwen3.6-35B-A3B, the default on most PCs, is a mixture-of-experts model (35B
parameters, 3B active per token). llama.cpp's `--fit` puts as many expert
layers in VRAM as fit and keeps the rest in system RAM, so it runs on a 12 GB
card with 32 GB of RAM, and more VRAM makes it faster.

## Install (Windows)

```
git clone https://github.com/joaovgaraujo/opencode-local-stack.git
cd opencode-local-stack
.\setup.cmd
```

`setup.cmd` installs Python 3.12 with winget if it's missing, then asks four
questions, each with a default you accept with Enter:

1. Model: the catalog sorted by fit, or `H` for a GGUF model from a Hugging
   Face link.
2. Context: 32K to 256K tokens. The default comes from total VRAM: 64K below
   16 GB, 128K below 24 GB, 256K above.
3. Thinking: limited (default), off, or unlimited.
4. Whether PowerShell may run scripts for your user, if Windows blocks them
   (that block is what makes a plain `opencode` fail with "running scripts is
   disabled").

Then it downloads the newest llama.cpp build for your GPU driver, the model
and OpenCode 2.3.2, runs the tests, and installs the `opencode-local` and
`opencode` commands. Re-running it is safe; it reuses what is already downloaded and
offers to update an older llama.cpp.

## Install (Linux and macOS)

```
curl -fsSL https://github.com/joaovgaraujo/opencode-local-stack/releases/latest/download/opencode-local-installer.sh | bash
```

That downloads the latest release into `~/opencode-local` (set `OCL_DEST` to
change it) and runs `setup.sh`, the same wizard as on Windows. With git:
`git clone` the repo and run `./setup.sh`. Nothing uses sudo: llama.cpp,
OpenCode and the model go into the install folder, the `opencode-local` and
`opencode` commands into `~/.local/bin`. If `~/.local/bin` isn't on your PATH, the installer adds it to
`~/.profile` (Linux) or `~/.zprofile` (macOS); open a new terminal afterwards.

**Linux with an NVIDIA GPU** gets llama.cpp's prebuilt CUDA build, matched to
the CUDA version your driver supports. Those builds need glibc 2.38 or newer
(Ubuntu 24.04, Debian 13, Fedora 39 or later). On older systems, or when you
pick it in the wizard (`--build-llamacpp`), the installer compiles llama.cpp
with CUDA from source instead: 5 to 20 minutes, and it needs git, cmake, a C++
compiler and the CUDA toolkit (`nvcc`). AMD uses ROCm, other GPUs Vulkan.

**macOS on Apple Silicon** runs MLX models with rapid-mlx 1.0.5. It needs
Python 3.12 or 3.13: the installer uses one if the Mac has it (Homebrew or
python.org), and otherwise downloads `uv` into the install folder to get a
private Python 3.12, with no Homebrew or admin rights needed. Thinking off/limited/on and
the idle unload work the same way; Hugging Face GGUF links are llama.cpp-only
and not offered on a Mac.

## Use it

Open a terminal in your project folder and run:

```
opencode-local
```

It starts the model server in the background if it isn't running (about a
minute the first time), then opens OpenCode in that folder. The server keeps
running after you close OpenCode, so the next start is instant. After 10 idle
minutes it unloads the model, freeing the GPU memory and RAM; the next request
loads it again, which takes seconds while the file is still in Windows' file
cache.

```
opencode-local --think off                  this session without thinking
opencode-local --think limited --think-budget 2048
opencode-local --think on                   unlimited thinking
opencode-local --ctx 131072                 another context size
opencode-local --idle-minutes 30            unload after 30 idle minutes (0 = never)
opencode-local --think off --save           make it the default
opencode-local --status                     what is running
opencode-local -c                           continue the last OpenCode session
opencode-local --stop                       stop the model server
opencode-local run "fix the failing test"   arguments after the options go to OpenCode
```

`opencode-local` is a `.cmd` file, so PowerShell's script policy never blocks
it. Settings live in `settings.json` in this folder.

## Thinking

These models reason before answering. Unlimited thinking can take several
minutes on hard prompts, so the default caps it with llama.cpp's
`--reasoning-budget`. On six coding tasks (Qwen3.6-35B-A3B Q4_K_M):

| Thinking | Tasks passed | Tokens for all six |
|---|---|---|
| limited to 1,024 tokens | 5/6 | 7,704 |
| limited to 4,096 tokens (default) | 6/6 | 23,137 |
| unlimited | 6/6 | 27,023 |
| off | 6/6 | 2,578 |

On a 12 GB card each limited answer took 28 to 69 seconds. These were short,
self-contained tasks; thinking matters more on bugs that span several files.

## Speed on a 12 GB GPU

Measured with llama.cpp b12359 on an RTX 4090 limited to the ~10 GB a 12 GB
card has free, Ryzen 9 7950X3D, 64 GB RAM. Qwen3.6-35B-A3B Q4_K_M, 16K-token
prompt. Your tok/s will differ with RAM speed and CPU.

| Setup | Prefill | Generation |
|---|---|---|
| 64K context, all experts in RAM (`--cpu-moe`, the old default) | 3,726 tok/s | 51 tok/s |
| 64K context, `--fit` (now the default) | 4,437 tok/s | 73 tok/s |
| 256K context, all experts in RAM | 3,763 tok/s | 51 tok/s |
| 256K context, `--fit` | 4,209 tok/s | 63 tok/s |

Other defaults, from the same runs: prompt batches of 2048 (`-ub 2048`; about
1,400 tok/s prefill at the old 512), a q8_0 K and V cache (a mixed q8_0/q4_0
cache falls back to a slow FlashAttention path in current builds), and
`--load-mode none`, which reads the weights into RAM instead of memory-mapping
them. `--fit` keeps a 1.5 GB VRAM margin; it overshot its own plan by about
0.5 GB in testing, and going past the card's memory spills into slow shared
memory.

## Models from Hugging Face

In the wizard choose `H`, or pass a link:

```
.\setup.cmd --hf https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF
.\setup.cmd --hf https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF --hf-file Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
```

A repo link picks the Q4_K_M-class file; a link to one `.gguf` file picks
that file. The installer reads the file sizes and the GGUF architecture from
Hugging Face to decide how the model is served. The repo needs single-file
GGUFs (most `-GGUF` repos by bartowski, unsloth or mradermacher have them).
Repos from a link are not curated; the tests at the end of the install are
what tell you whether one works with OpenCode.

## Other options

```
python install.py --list-models              # every catalog model, quant, and size
python install.py --model qwen3.5-9b --think off --non-interactive
python install.py --model qwen3.6-35b-a3b --ctx 262144 --non-interactive
python install.py --model-path models/<file>.gguf --model qwen3.6-35b-a3b   # a GGUF you already have
python install.py --update-llamacpp          # replace llama.cpp with the newest build
python install.py --gui                      # the older Tkinter window
python install.py --skip-tests
```

OpenCode is downloaded from its GitHub release (2.x is not published on npm),
checked against the release's SHA256SUMS, into `opencode/` in the install
folder, so Node.js is not needed. If that download fails, the installer falls
back to `npm install -g opencode-ai@1.19.6`, which needs Node.js; `--install-node`
(passed by `setup.cmd` and `setup.sh`) lets it install Node.js for that: winget
on Windows, Homebrew or nodejs.org on macOS, the nodejs.org tarball on Linux.

## Pick a model

The six rows at the bottom were added in 2026-11 and have not been run on
this stack yet. The RAM and VRAM columns below are conservative estimates from the fit
heuristic in `installer/catalog.py`, not measurements. The exception is
Qwen3.6-35B-A3B, which was measured on real hardware (see
[`docs/RESULTS.md`](docs/RESULTS.md)). See [`docs/MODELS.md`](docs/MODELS.md)
for how the estimate is computed. Re-measure on your own machine with
`tests/benchmark.py` before capacity planning.

| Model | Arch | Default quant | Est. VRAM | Est. RAM | Notes |
|---|---|---:|---:|---:|---|
| **Qwen3.6-35B-A3B** | MoE 35B/3B active | 20.6 GB | 3.3–6.3 GB minimum (measured, 64K–256K ctx); spare VRAM speeds it up | 21–22 GB (measured RSS) | See [`docs/RESULTS.md`](docs/RESULTS.md) for measured tok/s |
| **Gemma 4 26B-A4B** | MoE 26B/4B active | 15.8 GB | 3.8–4.5 GB (est.) | 19.8 GB (est.) | Experts in RAM like Qwen3.6, lighter RAM footprint |
| **Gemma 4 12B (Unified)** | Dense | 6.6 GB | 8.1–9.1 GB (est.) | ~3 GB (est.) | Needs a 12 GB-class GPU; measured not to fit 8 GB even at 8k ctx (see [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)) |
| **Qwen3.5-9B** | Dense | 5.3 GB | 6.8–7.8 GB (est.) | ~3 GB (est.) | Largest dense model in the catalog |
| **Gemma 4 E4B** | Dense | 4.6 GB | 6.1–7.1 GB (est.) | ~3 GB (est.) | |
| **Qwen3.5-4B** | Dense | 2.6 GB | 4.1–5.1 GB (est.) | ~3 GB (est.) | |
| **Qwen3.8-27B** | Dense | 16.5 GB | 18.0–19.0 GB (est.) | ~3 GB (est.) | Newest Qwen (2026-08). Needs a 24 GB-class GPU |
| **Nemotron 3.5 Lightning 30B-A3B** | Hybrid Mamba/MoE 33B/3B active | 25.3 GB | 3.8–4.5 GB (est.) | 29.3 GB (est.) | NVIDIA; needs 32 GB+ RAM |
| **LFM2.5-24B-A2B** | MoE 24B/2.45B active | 14.9 GB | 3.8–4.5 GB (est.) | 18.9 GB (est.) | Liquid AI; text-only, 128K max context |
| **Ornith-1.0-35B** | MoE 35B/3B active | 22.1 GB | 3.8–4.5 GB (est.) | 26.1 GB (est.) | Coding-agent fine-tune of Qwen3.5-35B-A3B (MIT). Its card: SWE-bench Verified 75.6, Terminal-Bench 2.1 64.2 (Qwen3.6-35B-A3B: 73.4 / 52.5) |
| **Ornith-1.0-9B** | Dense | 5.7 GB | 7.2–8.2 GB (est.) | ~3 GB (est.) | Agentic-coding fine-tune of Qwen3.5-9B (MIT) |
| **Nemotron 3.5 Bolt E4B** | Dense | 5.1 GB | 6.6–7.6 GB (est.) | ~3 GB (est.) | NVIDIA post-train of Gemma 4 E4B; llama.cpp only (no MLX build) |

The MoE VRAM column is the minimum, with every expert layer in system RAM; it
stays about the same across quant sizes. `--fit` fills any VRAM above it with
expert layers. Dense VRAM scales with quant size. On macOS each model also ships
4/6/8-bit MLX quants served from one unified-memory pool instead of a separate
VRAM/RAM split (see [`docs/MACOS.md`](docs/MACOS.md)).

On Apple Silicon the picker chooses by memory tier: qwen3.5-4b at 8 GB,
qwen3.5-9b at 16 GB, gemma-4-12b at 24 GB, qwen3.6-35b-a3b at 32 GB and up (the
tier table is in [`docs/MACOS.md`](docs/MACOS.md)). Fitting in memory is not the
same as driving OpenCode. Measured on a 16 GB M4, qwen3.5-9b is the smallest
Qwen that passes the agentic smoke test (2B and 4B print code instead of calling
tools, even at 8-bit), while Gemma's edge models tool-call at 2B. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md#local-measurements-on-a-16-gb-apple-m4)
for per-model tok/s, context ceilings, and pass/fail.

Full quant list and exact file sizes: [`docs/MODELS.md`](docs/MODELS.md).
Independent benchmark scores per model (coding, reasoning, tool use):
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md). Measured throughput, 8 GB context
fit, and CUDA-vs-Vulkan speed for the original six models on one machine:
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

## Good to know

- These are reasoning models. Thinking arrives in a separate
  `reasoning_content` field before the answer, so a client calling the server
  directly needs `max_tokens` of 1024 or more or `content` can come back
  empty. OpenCode handles this.
- With an MoE model the CPU does part of the work, so the GPU is not at 100%
  while it answers. That is expected.
- tok/s never transfers between machines. It depends on your GPU, RAM speed
  and CPU. Re-measure with `tests/validate.py` and `tests/vram_logger.ps1`,
  or an `nvidia-smi -l` loop on Linux.
- The first OpenCode run downloads `ripgrep` once and needs internet.
- OpenCode reads its settings from `%USERPROFILE%\.config\opencode\opencode.json`
  (`~/.config/opencode` on Linux). The installer adds its provider to that file
  and keeps anything else you configured there.
- To use a llama.cpp you built yourself, pass `--bin-dir <folder with llama-server>`
  (and `--backend cuda|vulkan|rocm|cpu` to say which). `--backend vulkan` also
  works on NVIDIA if the CUDA build gives you trouble.

## Repo layout

```
setup.cmd, setup.ps1          Windows one-step setup (installs Python if needed, runs the wizard)
install.py                    installer + validator (Windows, Linux, macOS)
start.py                      the opencode-local start command
installer/                    hardware detection, catalog, Hugging Face links, settings, download, server, wizard
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
docs/DEPLOY.md                manual usage, GPU backend selection, MoE re-tuning
```

Generated on first run and git-ignored: `llama.cpp/` (the runtime),
`models/*.gguf` (the weights), `settings.json` (model, context, thinking),
`run.ps1` / `run.sh` (start only the server, same as
`opencode-local --server-only`), and `RESULTS.md` (your machine's pass/fail
summary).

## Credits

- Models: [unsloth](https://huggingface.co/unsloth) GGUF and
  [mlx-community](https://huggingface.co/mlx-community) MLX quantizations of
  Qwen ([Qwen team, Alibaba](https://github.com/QwenLM)) and
  [Gemma](https://ai.google.dev/gemma) (Google) releases.
- Inference: [llama.cpp](https://github.com/ggml-org/llama.cpp) on Windows and
  Linux, [rapid-mlx](https://github.com/raullenchai/Rapid-MLX) on Apple Silicon.
- Agent frontend: [OpenCode](https://opencode.ai).
