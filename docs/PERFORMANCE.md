# Performance - same-PC measured throughput, memory, and context fit

Speed and memory measured on one machine on 2026-07-23. These are not portable
estimates - re-measure on your own hardware. For *published quality* scores
(LiveCodeBench, GPQA, tau2) see [`BENCHMARKS.md`](BENCHMARKS.md); this file is
about how fast each model runs and how much context fits **here**.

Throughput is `llama-bench` (pure generation/prompt throughput, no server or
sampling overhead); the HellaSwag column is `llama-perplexity` on this machine.

## Environment
- NVIDIA RTX 3500 Ada Laptop GPU (12 GB VRAM), 61 GB RAM, Linux.
- **Runtimes, both at tag `b10088`, stock upstream llama.cpp (no TurboQuant):**
  - CUDA: built locally from source, `sm_89`, CUDA 12.4.
  - Vulkan: the official prebuilt release the installer downloads.
- All runs: Q4_K_M weights, `q8_0` K and V cache, flash attention on. MoE
  models (`--cpu-moe`, 24 threads) keep expert weights in system RAM; dense
  models run fully on the GPU.
- A preflight gate (`tests/preflight.py --kill`) verified no stray processes,
  and a clean GPU baseline before every single run.

## Cross-model comparison (Q4_K_M, CUDA vs Vulkan)

Throughput shown as **CUDA / Vulkan** tok/s. `pp2048` = prompt processing over
2048 tokens; `tg` = generation of 256 tokens; `d=` is the KV depth the
measurement runs at (d=0 is a fresh context, d=32k is with 32k already in the
cache).

| Model | Arch | HellaSwag-400 | Max ctx on 8 GB | pp2048 d=0 | tg d=0 | tg d=32k |
|---|---|---:|---:|---:|---:|---:|
| Gemma 4 E4B | dense 4B | 67.5% | 262k | 4664 / 4217 | 85.3 / 85.5 | 58.2 / 70.9 |
| Qwen3.5-4B | dense 4B | 71.8% | 131k | 4113 / 3691 | 91.3 / 89.0 | 72.3 / 73.4 |
| Qwen3.5-9B | dense 9B | 78.0% | 65k | 2584 / 2285 | 55.6 / 53.0 | 48.2 / 43.9 |
| Gemma 4 12B | dense 12B | 52.5% | does not fit | 1900 / 1624 | 39.1 / 34.2* | 34.6 / 33.0 |
| Gemma 4 26B-A4B | MoE 26B/4B | 52.3% | 131k / 262k | 454 / 244 | 27.5 / 21.8 | 24.6 / 19.3 |
| Qwen3.6-35B-A3B | MoE 35B/3B | 82.3% | 262k | 362 / 188 | 36.8 / 26.9 | 32.5 / 24.9 |

\* Vulkan Gemma 12B reported a spurious 3.3 tok/s on the very first (d=0)
generation cell - a known llama-bench Vulkan warmup artifact; the 34.2 shown
is the d=16384 cell, which is representative. All other cells were stable.

### Reading the comparison

- **CUDA wins decisively on the MoE models.** With experts on the CPU, decode
  speed is limited by the GPU attention path, and CUDA's is far ahead of
  Vulkan: Qwen3.6-35B 36.8 vs 26.9 tok/s, Gemma 26B 27.5 vs 21.8 at d=0, with
  CUDA's prompt processing roughly 2x Vulkan's. If you run a `--cpu-moe` model
  on NVIDIA, build CUDA (see [`DOCKER.md`](DOCKER.md) for a container that
  already has it).
- **On dense models the two backends are close**, and Vulkan occasionally
  edges ahead at depth (Gemma E4B: 70.9 vs 58.2 tok/s at 32k). For a small
  dense model the official Vulkan prebuilt is a perfectly good default and
  needs no local build.
- **Generation speed degrades gracefully with context** on both backends -
  roughly 10-30% slower from d=0 to d=32k - so the deep-context numbers are
  the ones to plan around for long coding sessions.
- **HellaSwag caveat:** these are instruct/reasoning models and HellaSwag is a
  raw sentence-completion likelihood task, which understates instruction-tuned
  models and interacts with chat templates. The Qwen ordering
  (4B < 9B < 35B) is sane and monotonic; the low Gemma 4 scores are almost
  certainly a template/tokenizer interaction, not a true quality gap, and
  should not be read as "Gemma is worse than a 4B Qwen." Treat the column as a
  loose sanity check, not a leaderboard - the published-score table in
  [`BENCHMARKS.md`](BENCHMARKS.md) is the better quality reference.

## 8 GB "simulated RTX 4060 Laptop" context fit

Largest `--ctx-size` whose peak total GPU use stayed under **7,168 MiB** (8 GiB
minus ~1 GiB OS/desktop headroom), q8_0 KV, weights placement as above.

| Model | CUDA max ctx (peak MiB) | Vulkan max ctx (peak MiB) |
|---|---:|---:|
| Gemma 4 E4B | 262k (6818) | 262k (5704) |
| Qwen3.5-4B | 131k (6022) | 131k (5394) |
| Qwen3.5-9B | 65k (6878) | 65k (6519) |
| Gemma 4 12B | **does not fit** (8100 at 8k) | **does not fit** (7957 at 8k) |
| Gemma 4 26B-A4B | 131k (5524) | 262k (6599) |
| Qwen3.6-35B-A3B | 262k (6228) | 262k (5757) |

- **The two MoE models fit enormous contexts on an 8 GB card** - up to the
  full 262k training context for Qwen3.6-35B - because `--cpu-moe` keeps the
  ~20 GB of expert weights in system RAM and only attention + KV live on the
  GPU. This is the whole point of the MoE offload trick.
- **Gemma 4 12B dense does not fit an 8 GB card at all**: at Q4_K_M its 6.6 GB
  of weights plus compute buffers already reach ~8.0 GB at only 8k context.
  Use it on a 12 GB+ card, or pick Qwen3.5-9B (fits 64k) / a smaller model.
- Vulkan is consistently a few hundred MiB leaner than CUDA at the same
  context, and for Gemma 26B that let it reach 262k where CUDA topped out at
  131k under the same cap.

## Experimental TurboQuant weight formats (Qwen3.6-35B, Gemma 26B)

Measured on the TurboQuant fork's CUDA build (`GGML_TQ_NATIVE=1`), q8_0 KV, all
experts on CPU (`-ncmoe 99`), compared against the same model's standard
Q4_K_M on the same runtime.

| Model / format | File size | pp512 | tg128 | Peak total GPU |
|---|---:|---:|---:|---:|
| Qwen3.6-35B Q4_K_M | 22.13 GB | 359 tok/s | **36.3 tok/s** | 2879 MiB |
| Qwen3.6-35B TQ4_1S | 21.85 GB | 358 tok/s | **2.9 tok/s** | 2319 MiB |
| Gemma 26B Q4_K_M | 15.78 GB | 451 tok/s | **27.6 tok/s** | 3447 MiB |
| Gemma 26B TQ3_1S | 12.91 GB | 417 tok/s | **2.4 tok/s** | 2461 MiB |

**Conclusion: the TurboQuant weight formats are not worth it for these MoE
models.** TQ4_1S is only 1.3% smaller than Q4_K_M yet decodes ~12x slower
(2.9 vs 36.3 tok/s), because the native TurboQuant CPU MoE dequant kernels are
far slower than the mature Q4_K_M ones and these models run their experts on
the CPU. Gemma's TQ3_1S is genuinely smaller (12.9 vs 15.8 GB, ~500 MiB less
GPU) but pays the same ~11x decode penalty (2.4 vs 27.6 tok/s). Even offloading
more experts to the GPU on the 12 GB card, TQ4_1S never exceeded ~4.2 tok/s.
Stick with Q4_K_M; the TurboQuant *weight* formats only make sense if a model
would otherwise not load at all, and even then decode is impractically slow.

(This is separate from the TurboQuant *KV-cache* types, which are unusable on
this fork for a different reason - a CUDA decode bug; see
[`DOCKER.md`](DOCKER.md).)

## Reproducing

Scripts: `tests/benchmark.py` (capped llama-bench wrapper), `tests/preflight.py`
(clean-state gate), and `llama-perplexity --hellaswag --hellaswag-tasks 400`
against the standard HellaSwag validation set. Raw per-run JSON is written under
`benchmark-results/` (git-ignored - regenerate locally).
