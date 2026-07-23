> **Historical reference.** This run predates the multi-model catalog and
> cross-platform `install.py`; it's kept as the one real, measured data point
> behind the VRAM/RAM fit estimates in [`MODELS.md`](MODELS.md) for the
> Qwen3.6-35B-A3B / Q4_K_M / primary-profile case specifically. `install.py`
> now writes a fresh `RESULTS.md` at the repo root for whatever
> model/quant/profile you actually installed; that one reflects your machine.

# RESULTS: Qwen3.6-35B-A3B llama.cpp stack (8 GB profile)

Measured on a 24 GB desktop GPU as a stand-in for an 8 GB-class laptop target.
The config was **deliberately constrained to an 8 GB budget**. The spare VRAM
was left unused on purpose. See "What transfers" in `DEPLOY.md`.

## Test setup (constrained to the 8 GB target)

| | |
|---|---|
| GPU | 24 GB desktop GPU, same GPU architecture as the 8 GB target, so identical CUDA kernels run on both |
| RAM | ample system RAM (experts run in RAM via `--cpu-moe`) |
| OS | Windows x64 |

## Stack

| | |
|---|---|
| Binary | llama.cpp **b9928**, prebuilt **CUDA 12.4** Windows x64 (fat binary, PTX fallback) |
| Model | `unsloth/Qwen3.6-35B-A3B-GGUF` → `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| Model size | 22,134,528,992 bytes (~20.6 GiB), verified against HF; single file (not sharded) |
| Model meta | 34.66B params (A3B MoE), vocab 248320, n_ctx_train 262144, ftype Q4_K_M |

## Final flags (primary profile, the target 8 GB config)

```
llama-server
  -m Qwen3.6-35B-A3B-UD-Q4_K_M.gguf
  --alias qwen3.6-35b-a3b
  --n-gpu-layers 999          # all layers to GPU; --cpu-moe forces expert tensors to CPU/RAM
  --cpu-moe                   # ALL MoE experts in system RAM
  --ctx-size 65536
  --flash-attn on
  -ctk q8_0  -ctv q8_0        # quantized KV cache
  --jinja                     # chat template + tool calling
  --host 127.0.0.1 --port 8080
```

## Performance (RE-MEASURE ON YOUR OWN GPU: tok/s does NOT transfer)

| Metric | Value (on the test GPU) |
|---|---|
| Gen tok/s @ fresh context | **~47.6 tok/s** |
| Gen tok/s @ ~30k context | **~44.4 tok/s** |
| Prompt eval (30k prompt) | ~586 tok/s (~74s to ingest 30.5k tokens) |

MoE experts run on CPU, so prompt ingestion is CPU-bound; a different machine
(CPU, RAM bandwidth, GPU clocks, thermals) will differ; treat these as
baselines only.

## VRAM / RAM fit (this DOES transfer: same 8 GB-class footprint)

Windows/WDDM does not expose per-process VRAM (`nvidia-smi` shows `[N/A]`), so the
llama-server footprint below = (whole-card VRAM with server) − (whole-card baseline
with server stopped), sampled once per second through the 30k needle test.

| Profile | Peak VRAM (llama-server) | Peak RAM (working set) | Verdict vs 6.5 GB |
|---|---|---|---|
| **Primary, 65536 ctx, q8_0 KV** | **~4.0 GB** | ~22.7 GB | ✅ PASS (~2.5 GB headroom) |
| Conservative, 32768 ctx, q8_0 KV | ~3.3 GB | ~22 GB | ✅ PASS (~3.2 GB headroom) |

- KV cache (65536 @ q8_0) accounts for ~1.4 GB of the primary footprint; the rest is
  dense/attention weights + output head + compute buffer (constant, ctx-independent).
- Halving ctx to 32768 saves ~0.7 GB VRAM. It does **not** meaningfully reduce RAM:
  the ~20 GB of experts dominate RAM regardless of ctx.
- On a real 8 GB laptop, the desktop compositor eats ~1–1.5 GB, so expect ~5.5 GB total
  card usage at the primary profile, still comfortably inside 8 GB.

## Pass / fail table

| Test | Result | Detail |
|---|---|---|
| Short completion | ✅ PASS | Coherent one-sentence Rayleigh-scattering answer |
| Code gen + `ast.parse` | ✅ PASS | Valid `fib(n)`, parsed, 1 func, docstring present |
| Tool calling (`--jinja`) | ✅ PASS | Well-formed `get_weather(location="Paris")` tool call |
| Long context: 30k needle @ 60% | ✅ PASS | Retrieved `PURPLE-WOMBAT-4291` exactly (prompt 30,507 tok) |
| VRAM fit ≤ 6.5 GB | ✅ PASS | Peak ~4.0 GB (primary), ~3.3 GB (conservative) |
| OpenCode agentic (write+run+read) | ✅ PASS | Wrote `calc.py` = `print(2+3)`, ran it, read back `5` |

**Overall: PASS.** The full target profile fits the 8 GB budget with ~2.5 GB to spare.

## Gotcha found during testing (important)

**Qwen3.6 is a reasoning model.** It emits `<think>` tokens (returned in a separate
`reasoning_content` field) *before* the final `content`. With small `max_tokens`, the
whole budget is consumed by reasoning and `content` comes back **empty** (finish_reason
`length`). This is NOT a config or KV failure; it bit the first test run. Give generous
`max_tokens` (≥1024, ideally 2048+) so reasoning + answer both fit. OpenCode handles this
natively (it has 32768 output budget configured).
