"""settings.json: the one place the installer and the `opencode-local` start
command read the server setup from.

install.py writes it; `opencode-local --think off --save` and friends update
it. server_args() turns it into llama-server arguments, so the installer, the
start command and run.ps1/run.sh can never drift apart.

Defaults, measured on Qwen3.6-35B-A3B Q4_K_M (llama.cpp b12359):
  --fit on        llama.cpp places as many MoE expert layers in VRAM as fit and
                  keeps the rest in RAM. On a simulated 12 GB card at 64K
                  context: 73 tok/s generation vs 51 with every expert in RAM
                  (--cpu-moe); at 262K: 63 vs 51.
  -ub/-b 2048     larger prompt batches: prefill 1,400 -> 3,700+ tok/s with
                  experts in RAM, 4,400 with --fit on the same 12 GB budget.
  -ctk/-ctv q8_0  matched K/V types. Mixed q8_0/q4_0 has no FlashAttention
                  kernel in current builds and falls back to a slow f16 path.
  --load-mode none  read weights into RAM instead of memory-mapping them;
                  faster when experts run on the CPU.
  thinking        "limited" caps thinking at think_budget tokens (default
                  4096): 6/6 on our coding test vs 6/6 unlimited, at about a
                  sixth of the tokens. "off" skips it; "on" is unlimited.
"""
import json
import os

FILE_NAME = "settings.json"

THINK_MODES = ("limited", "off", "on")
DEFAULTS = {
    "port": 8080,
    "thinking": "limited",
    "think_budget": 4096,
    "kv_type": "q8_0",
    "batch": 2048,
    "fit_margin_mib": 1536,
    "extra_args": [],
}

KV_K_FLAGS = ("-ctk", "--cache-type-k")
KV_V_FLAGS = ("-ctv", "--cache-type-v")
LOAD_FLAGS = ("-lm", "--load-mode", "--mmap", "--no-mmap")
PLACEMENT_FLAGS = ("-ngl", "--n-gpu-layers", "--gpu-layers", "--cpu-moe", "-cmoe", "-ncmoe",
                   "--n-cpu-moe", "-ot", "--override-tensor", "-fit", "--fit")
BATCH_FLAGS = ("-ub", "--ubatch-size", "-b", "--batch-size")
REASONING_FLAGS = ("-rea", "--reasoning", "--reasoning-budget")
PARALLEL_FLAGS = ("-np", "--parallel")
# OpenCode's system prompt and tool definitions take ~10K tokens.
MIN_CTX = 32768


def path(root):
    return os.path.join(root, FILE_NAME)


def load(root):
    try:
        with open(path(root), encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return dict(DEFAULTS, **data)


def save(root, data):
    with open(path(root), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def user_sets(extra_args, flags):
    """True when extra_args already pass one of these flags; the matching
    default is then left out (no duplicate-flag warnings, user wins)."""
    return any(a in flags or a.split("=", 1)[0] in flags for a in (extra_args or []))


def resolve_path(root, p):
    return p if os.path.isabs(p) else os.path.join(root, p)


def thinking_args(mode, budget):
    if mode == "off":
        return ["--reasoning", "off"]
    if mode == "limited":
        return ["--reasoning", "on", "--reasoning-budget", str(int(budget))]
    return ["--reasoning", "on"]


def server_args(s, root):
    """llama-server arguments for settings dict s."""
    extra = list(s.get("extra_args") or [])
    backend = s["backend"]
    gpu = backend != "cpu"
    moe = s.get("arch") == "moe"
    args = ["-m", resolve_path(root, s["model_path"]), "--alias", s["model_id"],
            "--ctx-size", str(s["ctx"]), "--flash-attn", "on"]
    if not user_sets(extra, KV_K_FLAGS):
        args += ["-ctk", s["kv_type"]]
    if not user_sets(extra, KV_V_FLAGS):
        args += ["-ctv", s["kv_type"]]
    args += ["--jinja", "--host", "127.0.0.1", "--port", str(s["port"])]
    if not user_sets(extra, PARALLEL_FLAGS):
        # One slot. llama.cpp's default (4 slots sharing one KV cache) breaks
        # when OpenCode sends its title/summary requests alongside the main
        # one: "failed to find N available cells in kv cache", and OpenCode
        # retries without end. With one slot the extra requests just queue.
        args += ["--parallel", "1"]
    if not user_sets(extra, PLACEMENT_FLAGS):
        if gpu:
            args += ["--fit", "on", "--fit-target", str(s["fit_margin_mib"])]
        else:
            args += ["--n-gpu-layers", "0"]
    if gpu and not user_sets(extra, BATCH_FLAGS):
        args += ["-ub", str(s["batch"]), "-b", str(s["batch"])]
    if moe and not user_sets(extra, LOAD_FLAGS):
        args += ["--load-mode", "none"]
    if s.get("reasoning", True) and not user_sets(extra, REASONING_FLAGS):
        args += thinking_args(s["thinking"], s["think_budget"])
    return args + extra


def describe(s):
    think = {"off": "thinking off",
             "limited": f"thinking limited to {s['think_budget']} tokens",
             "on": "thinking unlimited"}[s["thinking"]]
    return (f"{s['display_name']} | context {s['ctx']:,} | {think} | "
            f"http://127.0.0.1:{s['port']}/v1")
