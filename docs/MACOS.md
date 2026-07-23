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

Apple Silicon has no discrete VRAM - the GPU and CPU share the same physical
RAM. `catalog.mlx_fit_verdict()` models this as a single pool:

- `need_ram ≈ quant size + 2.5 GB` (KV cache/activations - smaller than the
  llama.cpp estimate since there's no separate GPU/CPU split to account for)
- **fits**: `need_ram ≤ total_RAM × 0.75` (macOS's approximate default GPU
  "wired memory" ceiling) and free RAM covers most of it too
- **tight**: within 85% of that ceiling
- **no**: otherwise

The 75% figure is a commonly-cited default for how much of total unified
memory macOS will let the GPU actually allocate; you can raise it yourself
with `sudo sysctl iogpu.wired_limit_mb=<N>` if you want to push closer to
100% (at real risk of OS instability if you go too far - this isn't
something `install.py` does for you). Free RAM on macOS (`hw.ram_free_gb`)
is a `vm_stat`-based approximation (free + inactive pages), not an exact
"available" figure the way Windows/Linux report it - treat it as a rough
signal, same spirit as everywhere else in this repo: these are heuristics
for sorting the picker, not a guarantee. Measure your own machine after
install.

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
