# macOS / Apple Silicon (rapid-mlx)

`install.py` supports macOS on Apple Silicon (M1 and later) through
[**rapid-mlx**](https://github.com/raullenchai/Rapid-MLX), an OpenAI-compatible
inference server built on Apple's MLX framework. It's picked automatically -
`hwdetect.pick_engine()` returns `'rapidmlx'` whenever it detects
`platform.system() == "Darwin"` and `platform.machine() == "arm64"`.

```
python install.py                              # GUI/CLI wizard, mlx catalog only
python install.py --model qwen3.5-9b --non-interactive   # unattended, default quant
python install.py --list-models                # both engines' catalogs
```

## Real-hardware validation

The rapid-mlx path is validated end-to-end on Apple Silicon with sufficient
unified memory using Python 3.14, Rapid-MLX 0.10.15, and
`mlx-community/Qwen3.5-4B-4bit`. The OpenAI-compatible endpoint passed short
completion, Python code generation, tool calling, and the 30,509-token needle
test; OpenCode also passed its write-and-run agent test. The same four endpoint
tests pass with Rapid-MLX's native `--kv-cache-turboquant k8v4` mode enabled.
Per-machine `RESULTS.md`, `server.log`, and `server.err` remain the source of
truth for other chips and larger models.

## Why rapid-mlx and not llama.cpp's Metal build

llama.cpp does publish a prebuilt macOS build (`llama-*-bin-macos-arm64.tar.gz`,
Metal-accelerated), and it would work here too. This installer uses rapid-mlx
instead because that's what was asked for: it's built specifically around
MLX, which several independent sources report as 2-4x faster than llama.cpp's
Metal backend on the same Apple Silicon hardware for supported models. If you
want the llama.cpp/GGUF path on a Mac instead, the `quants` (GGUF) list in
`installer/catalog.py` already has the file/repo data - `download.py` just
doesn't have a `("cuda"|"vulkan"|..., "Darwin")` entry in `ASSET_PATTERNS`
yet.

## Resolving a naming collision before trusting anything

While researching this, two different GitHub orgs were found making
identical claims to be "Rapid-MLX": `raullenchai/Rapid-MLX` and
`bitandmortar/rapid-mlx`. Rather than guess, the `rapid-mlx` PyPI package's
own metadata (`https://pypi.org/pypi/rapid-mlx/json` - Homepage/Repository/
Documentation) was checked directly: all three point at
`raullenchai/Rapid-MLX`. That's the one this installer trusts and installs
in a project-local `.rapidmlx-venv` (`rapid-mlx==0.10.15` from the normal
PyPI channel). This works with Homebrew Python, which correctly rejects
global `pip install` calls under PEP 668. It does **not** run the project's
own `curl | bash` one-liner installer, for the same reason
[`TURBOQUANT.md`](TURBOQUANT.md) gives for not auto-running third-party
prebuilt binaries: piping a remote script into a shell, or trusting an
unreviewed second implementation of the same project, isn't something this
installer will do silently on your behalf. If you install rapid-mlx some
other way first, `install.py` just uses whatever's already on PATH.

## How it differs from the llama.cpp path

|  | llama.cpp (Windows/Linux) | rapid-mlx (macOS) |
|---|---|---|
| Weight format | GGUF | MLX (safetensors) |
| Who downloads weights | `install.py` (from the exact `unsloth/*` file) | rapid-mlx itself, on first `serve <repo-id>` |
| Memory model | separate VRAM (GPU) + RAM (CPU/experts) | one unified pool - see below |
| MoE handling | `--cpu-moe` (experts to system RAM) | rapid-mlx manages placement itself; not configured by this installer |
| Context/profile | `--ctx-size`, primary/conservative/auto | no server context flag; OpenCode uses the catalog context and K8V4 reduces long-context KV pressure |
| KV compression | q8_0 by default | native Rapid-MLX TurboQuant K8V4 by default; `--mlx-turboquant none` disables it |
| Text-only serving | n/a | The catalog models are launched with `--no-mllm`; this prevents Rapid-MLX from misclassifying a text checkpoint as vision-only and requiring `mlx-vlm`. |
| Restart script | `run.ps1` / `run.sh` | `run.sh` only (no `run.ps1` - there's no Windows rapid-mlx target) |

## Unified memory and the fit estimate

Apple Silicon has no discrete VRAM; the GPU and CPU share one physical pool.
`catalog.mlx_fit_verdict()` ranks a model against total capacity, not against
the current free RAM.

The ceiling is not total RAM. rapid-mlx admits a request only while
`active weights + projected KV` stays under
`gpu_memory_utilization (default 0.90) x Metal recommendedMaxWorkingSetSize`.
On a 16 GB machine that working set measured 11.4 GB, so
`MLX_METAL_CAP_FRACTION = 0.71` models the cap as `0.71 x total RAM`. A request
that crosses it returns HTTP 503 `"would exceed gpu_memory_utilization cap"`,
not a hang. You can raise the working set with `sudo sysctl
iogpu.wired_limit_mb=<N>` (at the risk of OS instability if you push it too
far; `install.py` never does this for you), or pass a higher
`--gpu-memory-utilization`.

The memory a model needs is not flat across architectures, and the gap is
large. `estimate_mlx_requirements()` returns
`size x 1.05 + 1.0 GB + size x kv_factor`, where `kv_factor` is 1.07 for Gemma
and 0.0 for Qwen. Two runs on the same 16 GB M4 fixed those numbers, both with
TurboQuant K8V4 and `--pflash off`:

- `qwen3.5-9b-4bit` loads at 5.2 GB and served its full 262,144-token context
  window; memory was never the limit, only prefill speed.
- `gemma-4-12b-4bit` loads at 6.8 GB, but a real prompt 503s above ~1,200
  tokens. Its KV plus prefill working set is far heavier, so 4.6 GB of headroom
  bought almost no usable context.

A model is `fits` when `need <= cap`, `tight` within 15% above the cap, `no`
beyond that. The result is a per-tier auto-pick (default 4bit quant):

| Unified memory | Auto-pick | Also fits |
|---|---|---|
| 8 GB | `qwen3.5-4b` | (only the 4B) |
| 16 GB | `qwen3.5-9b` | qwen3.5-4b, gemma-4-e4b |
| 24 GB | `gemma-4-12b` | qwen3.5-9b, gemma-4-e4b |
| 32 GB | `qwen3.6-35b-a3b` | gemma-4-12b |
| 48 GB+ | `qwen3.6-35b-a3b` | gemma-4-26b-a4b |

Ranking uses total memory, not free RAM, on purpose. macOS parks most of its
RAM in inactive and cached pages, so a free-RAM gate made the pick flip between
runs: a 16 GB Mac that runs `qwen3.5-9b` would drop to `qwen3.5-4b` because a
browser held memory at that moment. Whether there's room right now is a
separate question, answered at serve time by the preflight.

Before starting the server, `install.py` runs a memory preflight
(`hwdetect.macos_available_ram_gb()`, a fresh `vm_stat` read of free, inactive,
speculative, and purgeable pages). It aborts if there isn't room to load the
weights, and warns if there's room to load but not enough KV headroom. Free RAM
on macOS is an approximation, so this catches obvious trouble early rather than
guaranteeing a fit. Measure your own machine after install.

### Measured performance (16 GB Apple Silicon M4, rapid-mlx 0.10.15)

Both models at 4bit MLX, TurboQuant K8V4, `--pflash off`. tok/s does not
transfer between machines; re-measure your own.

| Model | Footprint | Decode | Prefill | Max usable context |
|---|---:|---:|---:|---:|
| `qwen3.5-9b-4bit` | 5.2 GB | 20.5 tok/s | ~198 tok/s | ~262,000 tok (full window; prefill speed is the limit) |
| `gemma-4-12b-4bit` | 6.8 GB | 13.2 tok/s | ~83 tok/s | ~1,200 tok with 256-token output; ~690 with 1024 |

On 16 GB, `gemma-4-12b`'s usable context is smaller than a typical OpenCode
system prompt, which is why it 503'd the agentic smoke test. The catalog rates
it `no` at 16 GB and `fits` from 24 GB up. `qwen3.5-9b` holds real context on
16 GB, so it's the pick there.

Two startup problems the installer now handles automatically:

- **`rapid-mlx` version.** The `--kv-cache-turboquant <mode>` value form and
  the `--pflash` flag are `0.10.x`-era; an older Homebrew/pipx `rapid-mlx`
  (e.g. `0.6.x`, where `--kv-cache-turboquant` is a bare boolean) crashes with
  `unrecognized arguments: k8v4`. The installer now version-checks any
  `rapid-mlx` on `PATH` and falls back to the pinned `.rapidmlx-venv` when
  it's too old, instead of trusting whatever `PATH` happens to resolve.
- **PFlash + multimodal misclassification.** rapid-mlx auto-enables PFlash
  (`--pflash always`) for verified Qwen3.5/3.6 aliases, then refuses to run it
  once the model is (mis)classified as multimodal - and that check ignores
  `--no-mllm`, so `serve` hard-exits with `"--pflash is not supported for
  multimodal models"` before binding. The installer now passes `--pflash off`
  on this path (the catalog is text-only, so nothing is lost), which keeps
  startup reliable across every model.

## Model catalog

Same six models as the llama.cpp catalog, each with 4bit/6bit/8bit MLX
quants from `mlx-community` on Hugging Face - see
[`MODELS.md`](MODELS.md#sources-checked-while-building-this-catalog-2026-07)
for the exact repo ids and verified sizes. `--list-models` prints both
catalogs together.

## What you need

- macOS on Apple Silicon (M1 or later). Intel Macs fall back to the
  llama.cpp path, which isn't wired up for Darwin yet (see above) - running
  `install.py` on an Intel Mac will currently stop with a clear error rather
  than silently doing the wrong thing.
- Python 3.10+ (`rapid-mlx` requires it; macOS ships an older Python 3, so
  install a newer one first if needed - `brew install python@3.12` or
  python.org).
- Node.js for OpenCode - not installed silently on macOS either;
  `install.py` prints `brew install node` (or nvm) and stops if it's
  missing, same as Linux.
