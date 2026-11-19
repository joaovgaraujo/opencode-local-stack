"""Text wizard: the default way to install (setup.cmd runs it).

Four short questions - model, context, thinking, and (Windows) letting
PowerShell run `opencode` - each with a default you get by pressing Enter.
Every fit verdict is against TOTAL memory: what else happens to be open (a
browser, file cache) is freed by the OS when the model loads."""
from . import catalog, hf, settings

VERDICT_LABEL = {"fits": "fits", "tight": "tight fit", "no": "too big for this PC"}


def print_hardware(hw):
    print("=== Your computer ===")
    if hw.gpu_vendor and hw.gpu_vendor != "apple":
        vram = f"{hw.vram_total_gb:.0f} GB VRAM" if hw.vram_total_gb else "VRAM unknown"
        print(f"  GPU: {hw.gpu_name or hw.gpu_vendor} ({vram})")
    else:
        print("  GPU: none detected (CPU only - slow)")
    if hw.ram_total_gb:
        print(f"  RAM: {hw.ram_total_gb:.0f} GB")
    if hw.disk_free_gb:
        print(f"  Disk free: {hw.disk_free_gb:.0f} GB")
    print("  Models are matched to your total memory, not to what is free right now.\n")


def ask(prompt, default, valid=None):
    while True:
        ans = input(f"{prompt} [{default}]: ").strip() or str(default)
        if valid is None or ans.lower() in valid:
            return ans
        print("  please type one of: " + ", ".join(sorted(valid)))


def confirm(prompt, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{prompt} {suffix} ").strip().lower()
    return default if not ans else ans.startswith("y")


def _verdict(model, quant, ctx, hw):
    return catalog.hw_fit_verdict(catalog.with_ctx(model, ctx), quant, "primary", hw)


def _choose_catalog(hw):
    rows = []
    for model in catalog.MODELS:
        if not model.get("quants"):
            continue
        quant = catalog.default_quant(model)
        ctx = catalog.default_ctx(model, hw)
        rows.append((model, quant, ctx, _verdict(model, quant, ctx, hw)))
    order = {"fits": 0, "tight": 1, "no": 2}
    rows.sort(key=lambda r: (order[r[3]], -r[0]["total_params_b"]))
    rows = [r for r in rows if r[3] != "no"][:8] or rows[:3]
    print("=== Step 1: model ===")
    for i, (m, q, _ctx, v) in enumerate(rows, 1):
        tag = "  <- recommended" if i == 1 else ""
        print(f"  [{i}] {m['display_name']:28} {q['size_gb']:5.1f} GB  {VERDICT_LABEL[v]}{tag}")
    print("  [U] uncensored / abliterated, or any model from a Hugging Face link")
    valid = {str(i) for i in range(1, len(rows) + 1)} | {"u"}
    ans = ask("Choose", 1, valid).lower()
    if ans == "u":
        return _choose_hf(hw)
    m, q, _ctx, _v = rows[int(ans) - 1]
    return m, q


def _choose_hf(hw):
    print("\n  Hugging Face models:")
    for i, (repo, note) in enumerate(hf.SUGGESTED, 1):
        print(f"  [{i}] {note}")
    n = len(hf.SUGGESTED) + 1
    print(f"  [{n}] paste a link (https://huggingface.co/OWNER/NAME)")
    ans = ask("Choose", 1, {str(i) for i in range(1, n + 1)})
    while True:
        link = hf.SUGGESTED[int(ans) - 1][0] if int(ans) < n else input("  Link: ").strip()
        try:
            model, quant, ggufs = hf.resolve(link)
            break
        except (ValueError, OSError) as e:
            print(f"  {e}")
            ans = str(n)
    ctx = catalog.default_ctx(model, hw)
    files = [g for g in ggufs if g["shards"] == 1 and not g["turboquant"]]
    print(f"\n  {model['display_name']} ({model['arch'].upper()}) - pick a file. Q4_K_M is the "
          f"usual best size/quality trade-off; larger = better quality, slower.")
    default_i = 1
    for i, g in enumerate(files, 1):
        _m, q = hf.build_offline(model, g)
        v = _verdict(model, q, ctx, hw)
        if g["file"] == quant["file"]:
            default_i = i
        mark = "  <- default" if g["file"] == quant["file"] else ""
        print(f"  [{i:2}] {hf.quant_tag(g['file']):12} {g['size_gb']:5.1f} GB  "
              f"{VERDICT_LABEL[v]}{mark}")
    pick = int(ask("Choose", default_i, {str(i) for i in range(1, len(files) + 1)}))
    return hf.build_offline(model, files[pick - 1])


def _choose_ctx(model, quant, hw):
    default = catalog.default_ctx(model, hw)
    choices = catalog.CTX_CHOICES[model["arch"]]
    print("\n=== Step 2: context (how much code/conversation the model sees at once) ===")
    for i, ctx in enumerate(choices, 1):
        need_vram, need_ram = catalog.estimate_requirements(catalog.with_ctx(model, ctx), quant)
        v = _verdict(model, quant, ctx, hw)
        mark = "  <- recommended" if ctx == default else ""
        print(f"  [{i}] {ctx // 1024:4}K tokens   needs ~{need_vram:.1f} GB VRAM + "
              f"~{need_ram:.0f} GB RAM   {VERDICT_LABEL[v]}{mark}")
    if model["arch"] == "moe":
        print("  Smaller context leaves more VRAM for the model itself, so answers come faster.")
    ans = ask("Choose", choices.index(default) + 1, {str(i) for i in range(1, len(choices) + 1)})
    return choices[int(ans) - 1]


def _choose_thinking():
    print("\n=== Step 3: thinking (the model reasons before it answers) ===")
    print("  [1] limited - thinks up to ~4,000 tokens (about a minute at most)  <- recommended")
    print("  [2] off     - answers right away; fastest, a bit weaker on hard bugs")
    print("  [3] on      - no limit; best on hard problems, can take several minutes")
    print("  Change any time: opencode-local --think off | limited | on")
    ans = ask("Choose", 1, {"1", "2", "3"})
    return {"1": "limited", "2": "off", "3": "on"}[ans]


def run_wizard(hw, update_info=None, scripts_blocked=False):
    """Return a dict of choices for install.py. update_info = (installed_build,
    latest_build) when an older llama.cpp is already present."""
    print_hardware(hw)
    model, quant = _choose_catalog(hw)
    ctx = _choose_ctx(model, quant, hw)
    thinking = _choose_thinking()
    out = {"model": model, "quant": quant, "ctx": ctx, "thinking": thinking,
           "think_budget": settings.DEFAULTS["think_budget"], "update_llamacpp": False,
           "allow_ps_scripts": False}
    print()
    if update_info:
        out["update_llamacpp"] = confirm(f"Update llama.cpp from build {update_info[0]} to "
                                         f"{update_info[1]} (faster, recommended)?")
    if scripts_blocked:
        print("Windows is set to block PowerShell scripts, which stops a plain `opencode` "
              "from starting.")
        out["allow_ps_scripts"] = confirm("Allow scripts for your Windows user (RemoteSigned, "
                                          "no admin needed)?")
    print(f"\nSummary: {model['display_name']} ({quant['label']}, {quant['size_gb']:.1f} GB), "
          f"{ctx // 1024}K context, thinking {thinking}.")
    if not confirm("Install now?"):
        raise SystemExit("Cancelled.")
    return out


# --- macOS / Apple Silicon (rapid-mlx) picker ------------------------------

def _scored_mlx_options(hw):
    options = []
    for model, quant in catalog.all_mlx_variants():
        verdict = catalog.mlx_fit_verdict(model, quant, hw.ram_total_gb, hw.ram_free_gb,
                                           hw.disk_free_gb)
        need_ram = catalog.estimate_mlx_requirements(model, quant)
        options.append({"model": model, "quant": quant, "verdict": verdict, "need_ram": need_ram})
    order = {"fits": 0, "tight": 1, "no": 2}
    options.sort(key=lambda o: (order[o["verdict"]], not o["quant"].get("default"),
                                -o["model"]["total_params_b"]))
    return options


def choose_mlx_model_quant(hw):
    """macOS/Apple Silicon picker - unified memory, no context profiles
    (see catalog.estimate_mlx_requirements)."""
    print_hardware(hw)
    options = _scored_mlx_options(hw)
    print("=== Available models for rapid-mlx (sorted by fit on this machine) ===")
    for i, o in enumerate(options, 1):
        m, q = o["model"], o["quant"]
        print(f"  [{i:2}] {m['display_name']:26} {q['label']:32} "
              f"~{o['need_ram']:.1f}GB unified memory  -> {VERDICT_LABEL[o['verdict']]}")
    print()
    ans = ask(f"Pick a number [1-{len(options)}]", 1,
              {str(i) for i in range(1, len(options) + 1)})
    picked = options[int(ans) - 1]
    return picked["model"], picked["quant"]
