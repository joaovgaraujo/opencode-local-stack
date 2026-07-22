"""Text-based installer wizard - used when --cli is passed, or automatically
when Tkinter isn't available (e.g. headless Linux without python3-tk)."""
from . import catalog

VERDICT_LABEL = {"fits": "fits comfortably", "tight": "tight fit - may need the conservative profile",
                  "no": "does not fit on this machine"}


def print_hardware(hw):
    print("=== Detected hardware ===")
    for line in hw.summary_lines():
        print(f"  {line}")
    print()


def _scored_options(hw):
    options = []
    for model, quant, profile in catalog.all_variants():
        if quant.get("experimental"):
            continue  # experimental (TurboQuant) entries are opt-in only, see docs/TURBOQUANT.md
        if not quant.get("default") and profile != "primary":
            continue  # keep the list short: only show conservative profile alongside the default quant
        verdict = catalog.fit_verdict(model, quant, profile, hw.vram_free_gb or hw.vram_total_gb,
                                       hw.ram_free_gb, hw.disk_free_gb)
        need_vram, need_ram = catalog.estimate_requirements(model, quant, profile)
        options.append({
            "model": model, "quant": quant, "profile": profile, "verdict": verdict,
            "need_vram": need_vram, "need_ram": need_ram,
        })
    order = {"fits": 0, "tight": 1, "no": 2}
    options.sort(key=lambda o: (order[o["verdict"]], o["model"]["total_params_b"]))
    return options


def choose_model_quant(hw):
    print_hardware(hw)
    options = _scored_options(hw)
    print("=== Available models (sorted by fit on this machine) ===")
    for i, o in enumerate(options, 1):
        m, q = o["model"], o["quant"]
        label = q["label"] + (f"  [{o['profile']}]" if o["profile"] != "primary" else "")
        print(f"  [{i:2}] {m['display_name']:26} {label:42} "
              f"~{o['need_vram']:.1f}GB VRAM / ~{o['need_ram']:.1f}GB RAM  "
              f"-> {VERDICT_LABEL[o['verdict']]}")
    print()
    print("  [t]  show experimental TurboQuant variants (needs a custom llama.cpp fork)")
    while True:
        choice = input(f"Pick a number [1-{len(options)}] (default 1): ").strip()
        if not choice:
            choice = "1"
        if choice.lower() == "t":
            _print_experimental(hw)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            picked = options[int(choice) - 1]
            return picked["model"], picked["quant"], picked["profile"]
        print("  invalid choice, try again")


def _scored_mlx_options(hw):
    options = []
    for model, quant in catalog.all_mlx_variants():
        verdict = catalog.mlx_fit_verdict(model, quant, hw.ram_total_gb, hw.ram_free_gb,
                                           hw.disk_free_gb)
        need_ram = catalog.estimate_mlx_requirements(model, quant)
        options.append({"model": model, "quant": quant, "verdict": verdict, "need_ram": need_ram})
    order = {"fits": 0, "tight": 1, "no": 2}
    options.sort(key=lambda o: (order[o["verdict"]], o["model"]["total_params_b"]))
    return options


def choose_mlx_model_quant(hw):
    """macOS/Apple Silicon picker - unified memory, no primary/conservative
    profile split (see catalog.estimate_mlx_requirements)."""
    print_hardware(hw)
    options = _scored_mlx_options(hw)
    print("=== Available models for rapid-mlx (sorted by fit on this machine) ===")
    for i, o in enumerate(options, 1):
        m, q = o["model"], o["quant"]
        print(f"  [{i:2}] {m['display_name']:26} {q['label']:32} "
              f"~{o['need_ram']:.1f}GB unified memory  -> {VERDICT_LABEL[o['verdict']]}")
    print()
    while True:
        choice = input(f"Pick a number [1-{len(options)}] (default 1): ").strip()
        if not choice:
            choice = "1"
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            picked = options[int(choice) - 1]
            return picked["model"], picked["quant"]
        print("  invalid choice, try again")


def _print_experimental(hw):
    print("\n=== Experimental (TurboQuant) ===")
    print("  Requires a community llama.cpp fork with TQ kernel support - the")
    print("  official llama.cpp releases do NOT understand these quant types.")
    print("  This installer will download the weights but will NOT auto-download")
    print("  or execute a third-party prebuilt binary. See docs/TURBOQUANT.md.\n")
    idx = []
    for model in catalog.MODELS:
        for quant in model["quants"]:
            if quant.get("experimental"):
                idx.append((model, quant))
                print(f"  [{len(idx)}] {model['display_name']:26} {quant['label']}")
    if not idx:
        print("  (none in the catalog yet)")
    print()


def confirm(prompt, default=True):
    suffix = "[Y/n]" if default else "[y/N]"
    ans = input(f"{prompt} {suffix} ").strip().lower()
    if not ans:
        return default
    return ans.startswith("y")
