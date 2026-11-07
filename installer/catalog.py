"""Model catalog: every model/quant/size figure here was looked up against the
Hugging Face API (exact GGUF/MLX filenames + byte sizes) rather than guessed —
see docs/MODELS.md for the sources. Keep it that way when adding entries: if
you can't verify a filename/size, don't add it.

Two engines, chosen per platform by hwdetect.pick_engine:

  llamacpp - GGUF weights served by llama-server (Windows/Linux). Each
             model's `quants` list feeds this path. Two serving strategies
             within it, chosen per model by `arch`:
               moe   - served with --cpu-moe: expert tensors live in system
                       RAM, attention + KV cache stay on the GPU. VRAM use is
                       ~constant and small regardless of quant size; RAM must
                       hold the whole quant file.
               dense - served fully on GPU (--n-gpu-layers 999). VRAM must
                       hold the quant file plus KV cache/compute buffer; RAM
                       needs are just OS overhead.
  rapidmlx - MLX-format weights served by the rapid-mlx CLI (Apple Silicon
             Mac only). Each model's `mlx` list feeds this path. Apple
             Silicon uses unified memory - there's no separate VRAM/RAM
             split, so this engine has its own single-pool fit estimate
             (see mlx_fit_verdict). arch/cpu-moe don't apply here: rapid-mlx
             manages GPU/CPU placement itself.

All GB figures are decimal (1 GB = 1e9 bytes), matching Hugging Face listings.
"""

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"

# Fit-heuristic constants. These are conservative estimates, not measurements -
# tests/validate.py + tests/vram_logger.ps1|sh give you real numbers for your
# machine after install. See docs/MODELS.md "How the fit estimate works".
MOE_VRAM_BASE_GB = {262144: 5.8, 131072: 5.0, 65536: 4.5, 32768: 3.8, 16384: 3.4}
# 262144: Qwen3.6-35B-A3B Q4_K_M measured +5.4 GB with --cpu-moe and q8_0 KV on CUDA
# (2026-11-07); 131072 is interpolated. Dense models are only measured to 32768.
MOE_RAM_OVERHEAD_GB = 4.0          # on top of the quant file size
DENSE_VRAM_OVERHEAD_GB = {32768: 2.5, 16384: 1.5}   # on top of the quant file size
DENSE_RAM_OVERHEAD_GB = 3.0         # OS + mmap bookkeeping only; weights live on GPU
VRAM_RESERVE_GB = 1.0                # keep display/desktop and transient buffers off the edge
RAM_RESERVE_GB = 2.0                 # avoid paging model data under ordinary desktop load

# rapid-mlx / unified memory: everything (weights + KV cache + prefill working
# set + activations) shares one pool. rapid-mlx admits a request only while
# (active weights + projected KV) stays under gpu_memory_utilization (0.90) x
# Metal recommendedMaxWorkingSetSize; on a 16 GB machine that ceiling is
# ~11.4 GB, so MLX_METAL_CAP_FRACTION models it as 0.71 of total RAM. Cross it
# and the server returns HTTP 503 "would exceed gpu_memory_utilization cap".
#
# The runtime memory a model needs is NOT flat across architectures, and the gap
# is large. Two runs on the same 16 GB M4, both with the tested KV
# compression and --pflash off:
#   qwen3.5-9b-4bit: loads at ~5.2 GB, served its full 262,144-token context
#                    window (memory was never the limit, only prefill speed).
#   gemma-4-12b-4bit: loads at ~6.8 GB, but a real prompt 503s above ~1,200
#                    tokens. Its KV plus prefill working set is far heavier, so
#                    ~4.6 GB of headroom bought almost no usable context.
# So the estimate is size + base overhead + a per-family KV term. Qwen (GQA,
# GQA) carries almost no KV overhead; Gemma dense carries a lot.
# This puts gemma-4-12b at "no" on 16 GB (measured: unusable for agents there)
# and "fits" from 24 GB up, while qwen3.5-9b "fits" from 16 GB and qwen3.5-4b is
# the only model that fits 8 GB. The live preflight still gates actual startup.
MLX_METAL_CAP_FRACTION = 0.71   # 11.4 GB working set on a 16 GB Mac
MLX_WEIGHT_FACTOR = 1.05        # resident weights run slightly above file size
MLX_BASE_OVERHEAD_GB = 1.0      # activations + a minimum prefill buffer
MLX_KV_FACTOR = {"gemma": 1.07, "qwen": 0.0}  # KV+prefill per GB of weights
MLX_KV_FACTOR_DEFAULT = 0.4     # unknown family: assume moderate KV weight

CTX_PROFILES = {
    "moe":   {"primary": 65536, "conservative": 32768},
    "dense": {"primary": 32768, "conservative": 16384},
}

MODELS = [
    {
        "id": "qwen3.6-35b-a3b",
        "display_name": "Qwen3.6-35B-A3B",
        "arch": "moe",
        "total_params_b": 35,
        "active_params_b": 3,
        "reasoning": True,
        "notes": "MoE, 256 experts. Same model validated in docs/RESULTS.md.",
        "quants": [
            {"label": "Q3_K_XL (compact)", "file": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
             "repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "size_gb": 15.69},
            {"label": "Q4_K_M (recommended)", "file": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
             "repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "size_gb": 20.61, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "Qwen3.6-35B-A3B-UD-Q5_K_M.gguf",
             "repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "size_gb": 24.64},
            {"label": "Q8_0 (max quality)", "file": "Qwen3.6-35B-A3B-Q8_0.gguf",
             "repo": "unsloth/Qwen3.6-35B-A3B-GGUF", "size_gb": 34.37},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/Qwen3.6-35B-A3B-4bit",
             "size_gb": 19.0, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/Qwen3.6-35B-A3B-6bit",
             "size_gb": 27.07},
            {"label": "8bit (max quality)", "repo": "mlx-community/Qwen3.6-35B-A3B-8bit",
             "size_gb": 35.13},
        ],
    },
    {
        "id": "gemma-4-26b-a4b",
        "display_name": "Gemma 4 26B-A4B",
        "arch": "moe",
        "total_params_b": 26,
        "active_params_b": 4,
        "reasoning": True,
        "notes": "MoE. Gemma 4's jinja chat template has known thinking/tool-call "
                 "quirks on some llama.cpp builds - see docs/MODELS.md.",
        "quants": [
            {"label": "Q3_K_XL (compact)", "file": "gemma-4-26B-A4B-it-UD-Q3_K_XL.gguf",
             "repo": "unsloth/gemma-4-26B-A4B-it-GGUF", "size_gb": 12.02},
            {"label": "Q4_K_M (recommended)", "file": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
             "repo": "unsloth/gemma-4-26B-A4B-it-GGUF", "size_gb": 15.78, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "gemma-4-26B-A4B-it-UD-Q5_K_M.gguf",
             "repo": "unsloth/gemma-4-26B-A4B-it-GGUF", "size_gb": 19.7},
            {"label": "Q8_0 (max quality)", "file": "gemma-4-26B-A4B-it-Q8_0.gguf",
             "repo": "unsloth/gemma-4-26B-A4B-it-GGUF", "size_gb": 25.02},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/gemma-4-26b-a4b-it-4bit",
             "size_gb": 14.29, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/gemma-4-26b-a4b-it-6bit",
             "size_gb": 20.16},
            {"label": "8bit (max quality)", "repo": "mlx-community/gemma-4-26b-a4b-it-8bit",
             "size_gb": 26.03},
        ],
    },
    {
        "id": "gemma-4-12b",
        "display_name": "Gemma 4 12B (Unified)",
        "arch": "dense",
        "total_params_b": 12,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense, fully on GPU. Omni (text/image/audio) weights; this stack "
                 "only exercises the text + tool-calling path.",
        "quants": [
            {"label": "Q4_K_M (recommended)", "file": "gemma-4-12b-it-Q4_K_M.gguf",
             "repo": "unsloth/gemma-4-12b-it-GGUF", "size_gb": 6.63, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "gemma-4-12b-it-Q5_K_M.gguf",
             "repo": "unsloth/gemma-4-12b-it-GGUF", "size_gb": 7.84},
            {"label": "Q6_K (near-lossless)", "file": "gemma-4-12b-it-Q6_K.gguf",
             "repo": "unsloth/gemma-4-12b-it-GGUF", "size_gb": 9.11},
            {"label": "Q8_0 (max quality)", "file": "gemma-4-12b-it-Q8_0.gguf",
             "repo": "unsloth/gemma-4-12b-it-GGUF", "size_gb": 11.8},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/gemma-4-12B-it-4bit",
             "size_gb": 6.28, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/gemma-4-12B-it-6bit",
             "size_gb": 9.06},
            {"label": "8bit (max quality)", "repo": "mlx-community/gemma-4-12B-it-8bit",
             "size_gb": 11.84},
        ],
    },
    {
        "id": "gemma-4-e4b",
        "display_name": "Gemma 4 E4B",
        "arch": "dense",
        "total_params_b": 4,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense elastic-4B model, fully on GPU. Smallest Gemma 4 text-capable size.",
        "quants": [
            {"label": "Q4_K_M (recommended)", "file": "gemma-4-E4B-it-Q4_K_M.gguf",
             "repo": "unsloth/gemma-4-E4B-it-GGUF", "size_gb": 4.64, "default": True},
            {"label": "Q6_K (near-lossless)", "file": "gemma-4-E4B-it-Q6_K.gguf",
             "repo": "unsloth/gemma-4-E4B-it-GGUF", "size_gb": 6.59},
            {"label": "Q8_0 (max quality)", "file": "gemma-4-E4B-it-Q8_0.gguf",
             "repo": "unsloth/gemma-4-E4B-it-GGUF", "size_gb": 7.63},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/gemma-4-e4b-it-4bit",
             "size_gb": 4.79, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/gemma-4-e4b-it-6bit",
             "size_gb": 6.53},
            {"label": "8bit (max quality)", "repo": "mlx-community/gemma-4-e4b-it-8bit",
             "size_gb": 8.27},
        ],
    },
    {
        "id": "qwen3.5-4b",
        "display_name": "Qwen3.5-4B",
        "arch": "dense",
        "total_params_b": 4,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense, fully on GPU. Smallest model in this catalog - runs on almost anything.",
        "quants": [
            {"label": "Q4_K_M (recommended)", "file": "Qwen3.5-4B-Q4_K_M.gguf",
             "repo": "unsloth/Qwen3.5-4B-GGUF", "size_gb": 2.55, "default": True},
            {"label": "Q6_K (near-lossless)", "file": "Qwen3.5-4B-Q6_K.gguf",
             "repo": "unsloth/Qwen3.5-4B-GGUF", "size_gb": 3.28},
            {"label": "Q8_0 (max quality)", "file": "Qwen3.5-4B-Q8_0.gguf",
             "repo": "unsloth/Qwen3.5-4B-GGUF", "size_gb": 4.17},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/Qwen3.5-4B-4bit",
             "size_gb": 2.83, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/Qwen3.5-4B-6bit",
             "size_gb": 3.8},
            {"label": "8bit (max quality)", "repo": "mlx-community/Qwen3.5-4B-8bit",
             "size_gb": 4.78},
        ],
    },
    {
        "id": "qwen3.5-9b",
        "display_name": "Qwen3.5-9B",
        "arch": "dense",
        "total_params_b": 9,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense, fully on GPU. Largest dense model in this catalog.",
        "quants": [
            {"label": "Q4_K_M (recommended)", "file": "Qwen3.5-9B-Q4_K_M.gguf",
             "repo": "unsloth/Qwen3.5-9B-GGUF", "size_gb": 5.29, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "Qwen3.5-9B-Q5_K_M.gguf",
             "repo": "unsloth/Qwen3.5-9B-GGUF", "size_gb": 6.13},
            {"label": "Q8_0 (max quality)", "file": "Qwen3.5-9B-Q8_0.gguf",
             "repo": "unsloth/Qwen3.5-9B-GGUF", "size_gb": 8.87},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/Qwen3.5-9B-4bit",
             "size_gb": 5.54, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/Qwen3.5-9B-6bit",
             "size_gb": 7.63},
            {"label": "8bit (max quality)", "repo": "mlx-community/Qwen3.5-9B-8bit",
             "size_gb": 9.71},
        ],
    },
    # ---- Added 2026-11 (newer releases). Filenames and sizes checked against
    # the HF tree API on 2026-11-07. Not measured on this stack yet: their fit
    # figures are heuristic only, like every row except Qwen3.6-35B-A3B.
    {
        "id": "qwen3.8-27b",
        "display_name": "Qwen3.8-27B",
        "arch": "dense",
        "family": "qwen",
        "total_params_b": 27,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense, fully on GPU. Newest Qwen (2026-08); needs a 24 GB-class GPU "
                 "at Q4_K_M. Text + tool-calling path only (vision encoder unused).",
        "quants": [
            {"label": "Q3_K_XL (compact)", "file": "Qwen3.8-27B-UD-Q3_K_XL.gguf",
             "repo": "unsloth/Qwen3.8-27B-GGUF", "size_gb": 13.15},
            {"label": "Q4_K_M (recommended)", "file": "Qwen3.8-27B-UD-Q4_K_M.gguf",
             "repo": "unsloth/Qwen3.8-27B-GGUF", "size_gb": 16.46, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "Qwen3.8-27B-UD-Q5_K_M.gguf",
             "repo": "unsloth/Qwen3.8-27B-GGUF", "size_gb": 19.77},
            {"label": "Q6_K (near-lossless)", "file": "Qwen3.8-27B-UD-Q6_K.gguf",
             "repo": "unsloth/Qwen3.8-27B-GGUF", "size_gb": 21.98},
            {"label": "Q8_0 (max quality)", "file": "Qwen3.8-27B-Q8_0.gguf",
             "repo": "unsloth/Qwen3.8-27B-GGUF", "size_gb": 29.05},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/Qwen3.8-27B-4bit",
             "size_gb": 16.05, "default": True},
            {"label": "8bit (max quality)", "repo": "mlx-community/Qwen3.8-27B-8bit",
             "size_gb": 29.5},
        ],
    },
    {
        "id": "nemotron-3.5-lightning-30b-a3b",
        "display_name": "Nemotron 3.5 Lightning 30B-A3B",
        "arch": "moe",
        "total_params_b": 33,
        "active_params_b": 3,
        "reasoning": True,
        "notes": "NVIDIA hybrid Mamba-2 + MoE (2026-08), served with --cpu-moe. "
                 "Few attention layers, so the KV cache stays small at long context. "
                 "NVIDIA Open Model License.",
        "quants": [
            {"label": "Q3_K_XL (compact)",
             "file": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q3_K_XL.gguf",
             "repo": "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "size_gb": 21.24},
            {"label": "Q4_K_M (recommended)",
             "file": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q4_K_M.gguf",
             "repo": "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "size_gb": 25.27,
             "default": True},
            {"label": "Q5_K_M (higher quality)",
             "file": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-UD-Q5_K_M.gguf",
             "repo": "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "size_gb": 30.22},
            {"label": "Q8_0 (max quality)",
             "file": "NVIDIA-Nemotron-3.5-Lightning-30B-A3B-Q8_0.gguf",
             "repo": "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", "size_gb": 35.0},
        ],
        "mlx": [
            {"label": "4bit (recommended)",
             "repo": "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit",
             "size_gb": 17.78, "default": True},
            {"label": "8bit (max quality)",
             "repo": "mlx-community/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-8bit",
             "size_gb": 33.56},
        ],
    },
    {
        "id": "lfm2.5-24b-a2b",
        "display_name": "LFM2.5-24B-A2B",
        "arch": "moe",
        "total_params_b": 24,
        "active_params_b": 2,
        "reasoning": True,
        "notes": "Liquid AI MoE (2026-10), 2.45B active, served with --cpu-moe. "
                 "Text-only, built for on-device agents. 128K max context. "
                 "LFM Open License.",
        "quants": [
            {"label": "Q3_K_XL (compact)", "file": "LFM2.5-24B-A2B-UD-Q3_K_XL.gguf",
             "repo": "unsloth/LFM2.5-24B-A2B-GGUF", "size_gb": 10.95},
            {"label": "Q4_K_XL (recommended)", "file": "LFM2.5-24B-A2B-UD-Q4_K_XL.gguf",
             "repo": "unsloth/LFM2.5-24B-A2B-GGUF", "size_gb": 14.88, "default": True},
            {"label": "Q6_K_XL (near-lossless)", "file": "LFM2.5-24B-A2B-UD-Q6_K_XL.gguf",
             "repo": "unsloth/LFM2.5-24B-A2B-GGUF", "size_gb": 21.86},
            {"label": "Q8_0 (max quality)", "file": "LFM2.5-24B-A2B-Q8_0.gguf",
             "repo": "unsloth/LFM2.5-24B-A2B-GGUF", "size_gb": 25.49},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "LiquidAI/LFM2.5-24B-A2B-MLX-4bit",
             "size_gb": 13.57, "default": True},
            {"label": "6bit (higher quality)", "repo": "LiquidAI/LFM2.5-24B-A2B-MLX-6bit",
             "size_gb": 19.49},
            {"label": "8bit (max quality)", "repo": "LiquidAI/LFM2.5-24B-A2B-MLX-8bit",
             "size_gb": 25.48},
        ],
    },
    {
        "id": "ornith-1.0-35b",
        "display_name": "Ornith-1.0-35B",
        "arch": "moe",
        "family": "qwen",
        "total_params_b": 35,
        "active_params_b": 3,
        "reasoning": True,
        "notes": "MoE, served with --cpu-moe. Agentic-coding fine-tune of Qwen3.5-35B-A3B by "
                 "deepreinforce-ai (2026-07), MIT. Its card reports SWE-bench Verified 75.6 and "
                 "Terminal-Bench 2.1 64.2 vs 73.4 / 52.5 for Qwen3.6-35B-A3B.",
        "quants": [
            {"label": "Q3_K_XL (compact)", "file": "Ornith-1.0-35B-UD-Q3_K_XL.gguf",
             "repo": "unsloth/Ornith-1.0-35B-GGUF", "size_gb": 16.8},
            {"label": "Q4_K_M (recommended)", "file": "Ornith-1.0-35B-UD-Q4_K_M.gguf",
             "repo": "unsloth/Ornith-1.0-35B-GGUF", "size_gb": 22.13, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "Ornith-1.0-35B-UD-Q5_K_M.gguf",
             "repo": "unsloth/Ornith-1.0-35B-GGUF", "size_gb": 26.46},
            {"label": "Q8_0 (max quality)", "file": "Ornith-1.0-35B-Q8_0.gguf",
             "repo": "unsloth/Ornith-1.0-35B-GGUF", "size_gb": 36.9},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/Ornith-1.0-35B-4bit",
             "size_gb": 20.4, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/Ornith-1.0-35B-6bit",
             "size_gb": 29.06},
            {"label": "8bit (max quality)", "repo": "mlx-community/Ornith-1.0-35B-8bit",
             "size_gb": 37.72},
        ],
    },
    {
        "id": "ornith-1.0-9b",
        "display_name": "Ornith-1.0-9B",
        "arch": "dense",
        "family": "qwen",
        "total_params_b": 9,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense, fully on GPU. Agentic-coding fine-tune of Qwen3.5-9B by "
                 "deepreinforce-ai (2026-07), MIT licensed. Same footprint as Qwen3.5-9B.",
        "quants": [
            {"label": "Q4_K_M (recommended)", "file": "Ornith-1.0-9B-Q4_K_M.gguf",
             "repo": "unsloth/Ornith-1.0-9B-GGUF", "size_gb": 5.7, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "Ornith-1.0-9B-Q5_K_M.gguf",
             "repo": "unsloth/Ornith-1.0-9B-GGUF", "size_gb": 6.54},
            {"label": "Q6_K (near-lossless)", "file": "Ornith-1.0-9B-Q6_K.gguf",
             "repo": "unsloth/Ornith-1.0-9B-GGUF", "size_gb": 7.48},
            {"label": "Q8_0 (max quality)", "file": "Ornith-1.0-9B-Q8_0.gguf",
             "repo": "unsloth/Ornith-1.0-9B-GGUF", "size_gb": 9.53},
        ],
        "mlx": [
            {"label": "4bit (recommended)", "repo": "mlx-community/Ornith-1.0-9B-4bit",
             "size_gb": 5.95, "default": True},
            {"label": "6bit (higher quality)", "repo": "mlx-community/Ornith-1.0-9B-6bit",
             "size_gb": 8.19},
            {"label": "8bit (max quality)", "repo": "mlx-community/Ornith-1.0-9B-8bit",
             "size_gb": 10.43},
        ],
    },
    {
        "id": "nemotron-3.5-bolt-e4b",
        "display_name": "Nemotron 3.5 Bolt E4B",
        "arch": "dense",
        "family": "gemma",
        "total_params_b": 4,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense, fully on GPU. NVIDIA post-train of Gemma 4 E4B (2026-10). "
                 "llama.cpp only: no plain 4/6/8-bit MLX build exists yet.",
        "quants": [
            {"label": "Q3_K_XL (compact)", "file": "NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q3_K_XL.gguf",
             "repo": "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF", "size_gb": 4.22},
            {"label": "Q4_K_XL (recommended)", "file": "NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q4_K_XL.gguf",
             "repo": "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF", "size_gb": 5.06,
             "default": True},
            {"label": "Q6_K_XL (near-lossless)", "file": "NVIDIA-Nemotron-3.5-Bolt-E4B-UD-Q6_K_XL.gguf",
             "repo": "unsloth/NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF", "size_gb": 7.26},
        ],
        "mlx": [],
    },
]


def get_model(model_id):
    for m in MODELS:
        if m["id"] == model_id:
            return m
    raise KeyError(f"Unknown model id: {model_id}")


def get_quant(model, quant_file):
    for q in model["quants"]:
        if q["file"] == quant_file:
            return q
    raise KeyError(f"Unknown quant file {quant_file!r} for model {model['id']}")


def default_quant(model):
    for q in model["quants"]:
        if q.get("default"):
            return q
    return model["quants"][0]


def download_url(quant):
    return HF_RESOLVE.format(repo=quant["repo"], file=quant["file"])


def get_mlx_quant(model, repo):
    for q in model.get("mlx", []):
        if q["repo"] == repo:
            return q
    raise KeyError(f"Unknown MLX repo {repo!r} for model {model['id']}")


def default_mlx_quant(model):
    if not model.get("mlx"):
        raise KeyError(f"{model['id']} has no MLX build; it runs on the llama.cpp "
                       f"engine (Windows/Linux) only")
    for q in model["mlx"]:
        if q.get("default"):
            return q
    return model["mlx"][0]


def ctx_sizes(model):
    """Context per profile. install.py --ctx sets model["ctx_override"], which
    replaces both profiles with one fixed size."""
    override = model.get("ctx_override")
    if override:
        return {"primary": override, "conservative": override}
    return CTX_PROFILES[model["arch"]]


def _nearest_up(table, ctx):
    """Value for the smallest tabulated context >= ctx (largest if beyond)."""
    keys = sorted(table)
    return table[next((k for k in keys if k >= ctx), keys[-1])]


def estimate_requirements(model, quant, profile="primary"):
    """Return (vram_gb, ram_gb) conservative estimates for this model/quant/profile.
    See the module docstring - these are heuristics, not measurements."""
    ctx = ctx_sizes(model)[profile]
    if model["arch"] == "moe":
        vram = _nearest_up(MOE_VRAM_BASE_GB, ctx)
        ram = quant["size_gb"] + MOE_RAM_OVERHEAD_GB
    else:
        vram = quant["size_gb"] + _nearest_up(DENSE_VRAM_OVERHEAD_GB, ctx)
        ram = DENSE_RAM_OVERHEAD_GB
    return round(vram, 1), round(ram, 1)


def fit_verdict(model, quant, profile, vram_gb, ram_gb, disk_free_gb):
    """Classify fit against the machine's TOTAL VRAM and RAM.

    Callers pass hw.vram_total_gb / hw.ram_total_gb, not the free figures.
    Free memory is a snapshot of whatever else happened to be open when the
    installer ran (a browser, a game, another model server), so ranking on it
    made the recommended model change from run to run on the same machine. The
    reserves below stand in for the OS/desktop baseline instead. Whether there
    is room *right now* is checked separately at serve time by install.py's
    memory preflight, which warns but never changes the pick."""
    need_vram, need_ram = estimate_requirements(model, quant, profile)
    if disk_free_gb is not None and disk_free_gb < quant["size_gb"] + 2:
        return "no"
    if vram_gb is None:
        # No GPU detected - only viable at all for small dense models, and only
        # via slow CPU inference. Treat as 'tight' rather than an outright 'no'.
        return "tight" if ram_gb is not None and ram_gb >= need_ram + quant["size_gb"] else "no"
    if (vram_gb >= need_vram + VRAM_RESERVE_GB and ram_gb is not None and
            ram_gb >= need_ram + RAM_RESERVE_GB):
        return "fits"
    if vram_gb >= need_vram and ram_gb is not None and ram_gb >= need_ram:
        return "tight"
    return "no"


def hw_fit_verdict(model, quant, profile, hw):
    """fit_verdict against the detected hardware's total capacity."""
    return fit_verdict(model, quant, profile, hw.vram_total_gb, hw.ram_total_gb,
                       hw.disk_free_gb)


def recommended_profile(model, quant, hw):
    """Choose the largest context that fits the machine's total memory with
    the desktop reserve kept. Based on capacity, not free memory, so the same
    machine always gets the same recommendation (see fit_verdict)."""
    for profile in ("primary", "conservative"):
        if hw_fit_verdict(model, quant, profile, hw) == "fits":
            return profile
    return "conservative"


def all_variants():
    """Flatten to (model, quant, profile) tuples for scoring against hardware."""
    out = []
    for model in MODELS:
        for quant in model["quants"]:
            for profile in ("primary", "conservative"):
                out.append((model, quant, profile))
    return out


def _mlx_kv_factor(model):
    """KV + prefill working-set weight, per GB of model weights, by family.
    Gemma dense attention is heavy (measured); Qwen GQA is not.
    See the MLX_KV_FACTOR comment block for the two measured anchor runs.
    A model's explicit "family" key wins over its id prefix (fine-tunes such as
    Ornith keep their base model's attention layout under a different name)."""
    family = model.get("family")
    if family is not None:
        return MLX_KV_FACTOR.get(family, MLX_KV_FACTOR_DEFAULT)
    mid = model["id"]
    for family, factor in MLX_KV_FACTOR.items():
        if mid.startswith(family):
            return factor
    return MLX_KV_FACTOR_DEFAULT


def estimate_mlx_requirements(model, quant):
    """Return the peak unified-memory estimate in GB for this model/MLX-quant:
    resident weights + a base activation/prefill buffer + a per-family KV term.
    rapid-mlx manages its own context/KV sizing, so this is one figure, not the
    primary/conservative split the GGUF path uses. The KV term is what separates
    a model that holds real context on a given Mac from one that 503s on the
    first long prompt (see MLX_KV_FACTOR)."""
    size = quant["size_gb"]
    return round(size * MLX_WEIGHT_FACTOR + MLX_BASE_OVERHEAD_GB
                 + size * _mlx_kv_factor(model), 1)


def mlx_fit_verdict(model, quant, ram_total_gb, ram_free_gb, disk_free_gb):
    """Classify an MLX (rapid-mlx / Apple Silicon) model+quant as 'fits',
    'tight', or 'no', against total unified-memory capacity.

    Deliberately does NOT gate on ram_free_gb. macOS free memory is a volatile
    vm_stat figure (the OS parks lots of RAM in inactive/cached pages), so
    gating the picker on it made the recommended model flip between runs
    depending on what else happened to be open - e.g. a 16 GB Mac that runs
    qwen3.5-9b fine would silently drop to qwen3.5-4b just because a browser was
    holding memory the moment you ran the installer. Capacity is the fixed
    hardware fact to rank on; whether there's room *right now* is a separate
    question answered at serve time by install.py's memory preflight
    (hwdetect.macos_available_ram_gb). ram_free_gb stays in the signature for
    symmetry with the llama.cpp fit_verdict and its callers."""
    need_ram = estimate_mlx_requirements(model, quant)
    if disk_free_gb is not None and disk_free_gb < quant["size_gb"] + 2:
        return "no"
    if ram_total_gb is None:
        return "no"
    cap = ram_total_gb * MLX_METAL_CAP_FRACTION
    if cap >= need_ram:
        return "fits"
    if cap >= need_ram * 0.85:
        return "tight"
    return "no"


def all_mlx_variants():
    """Flatten to (model, quant) tuples for scoring against hardware - no
    profile dimension (see estimate_mlx_requirements)."""
    out = []
    for model in MODELS:
        for quant in model.get("mlx", []):
            out.append((model, quant))
    return out
