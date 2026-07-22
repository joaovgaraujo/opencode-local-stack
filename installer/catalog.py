"""Model catalog: every model/quant/size figure here was looked up against the
Hugging Face API (exact GGUF filenames + byte sizes) rather than guessed — see
docs/MODELS.md for the sources. Keep it that way when adding entries: if you
can't verify a filename/size, don't add it.

Two serving strategies, chosen per model by `arch`:

  moe   - MoE model served with --cpu-moe: expert tensors live in system RAM,
          attention + KV cache stay on the GPU. VRAM use is ~constant and
          small regardless of quant size; RAM must hold the whole quant file.
  dense - Dense model served fully on GPU (--n-gpu-layers 999). VRAM must
          hold the quant file plus KV cache/compute buffer; RAM needs are
          just OS overhead.

All GB figures are decimal (1 GB = 1e9 bytes), matching Hugging Face listings.
"""

HF_RESOLVE = "https://huggingface.co/{repo}/resolve/main/{file}"

# Fit-heuristic constants. These are conservative estimates, not measurements -
# tests/validate.py + tests/vram_logger.ps1|sh give you real numbers for your
# machine after install. See docs/MODELS.md "How the fit estimate works".
MOE_VRAM_BASE_GB = {65536: 4.5, 32768: 3.8, 16384: 3.4}
MOE_RAM_OVERHEAD_GB = 4.0          # on top of the quant file size
DENSE_VRAM_OVERHEAD_GB = {32768: 2.5, 16384: 1.5}   # on top of the quant file size
DENSE_RAM_OVERHEAD_GB = 3.0         # OS + mmap bookkeeping only; weights live on GPU

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
            {"label": "TurboQuant TQ3_1S (experimental - needs a custom llama.cpp fork)",
             "file": "qwen3.6-35b-a3b-instruct-TQ3_1S.gguf",
             "repo": "mad-lab-ai/Qwen3.6-35B-A3B-tq-gguf", "size_gb": 16.37,
             "experimental": True, "engine": "turboquant"},
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
    },
    {
        "id": "qwen3.5-9b",
        "display_name": "Qwen3.5-9B",
        "arch": "dense",
        "total_params_b": 9,
        "active_params_b": None,
        "reasoning": True,
        "notes": "Dense, fully on GPU. Alibaba reports this beating much larger models "
                 "on reasoning benchmarks.",
        "quants": [
            {"label": "Q4_K_M (recommended)", "file": "Qwen3.5-9B-Q4_K_M.gguf",
             "repo": "unsloth/Qwen3.5-9B-GGUF", "size_gb": 5.29, "default": True},
            {"label": "Q5_K_M (higher quality)", "file": "Qwen3.5-9B-Q5_K_M.gguf",
             "repo": "unsloth/Qwen3.5-9B-GGUF", "size_gb": 6.13},
            {"label": "Q8_0 (max quality)", "file": "Qwen3.5-9B-Q8_0.gguf",
             "repo": "unsloth/Qwen3.5-9B-GGUF", "size_gb": 8.87},
        ],
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


def ctx_sizes(model):
    return CTX_PROFILES[model["arch"]]


def estimate_requirements(model, quant, profile="primary"):
    """Return (vram_gb, ram_gb) conservative estimates for this model/quant/profile.
    See the module docstring - these are heuristics, not measurements."""
    ctx = ctx_sizes(model)[profile]
    if model["arch"] == "moe":
        vram = MOE_VRAM_BASE_GB.get(ctx, 4.5)
        ram = quant["size_gb"] + MOE_RAM_OVERHEAD_GB
    else:
        vram = quant["size_gb"] + DENSE_VRAM_OVERHEAD_GB.get(ctx, 2.5)
        ram = DENSE_RAM_OVERHEAD_GB
    return round(vram, 1), round(ram, 1)


def fit_verdict(model, quant, profile, vram_gb, ram_gb, disk_free_gb):
    """Classify whether (vram_gb, ram_gb) free resources can run this
    model/quant/profile. Returns one of: 'fits', 'tight', 'no'."""
    need_vram, need_ram = estimate_requirements(model, quant, profile)
    if disk_free_gb is not None and disk_free_gb < quant["size_gb"] + 2:
        return "no"
    if vram_gb is None:
        # No GPU detected - only viable at all for small dense models, and only
        # via slow CPU inference. Treat as 'tight' rather than an outright 'no'
        # so the option still surfaces with a clear warning.
        return "tight" if ram_gb is not None and ram_gb >= need_ram + quant["size_gb"] else "no"
    if vram_gb >= need_vram and ram_gb >= need_ram:
        return "fits"
    if vram_gb >= need_vram * 0.75 and ram_gb >= need_ram * 0.85:
        return "tight"
    return "no"


def all_variants():
    """Flatten to (model, quant, profile) tuples for scoring against hardware."""
    out = []
    for model in MODELS:
        for quant in model["quants"]:
            for profile in ("primary", "conservative"):
                out.append((model, quant, profile))
    return out
