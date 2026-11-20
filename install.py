#!/usr/bin/env python3
"""install.py - one-shot cross-platform installer + validator for the local
LLM + OpenCode coding-agent stack (Windows, Linux, and macOS/Apple Silicon).

    setup.cmd                         # Windows: double-click, or run in PowerShell
    python install.py                 # text wizard: model, context, thinking
    python install.py --hf https://huggingface.co/OWNER/NAME     # any GGUF repo
    python install.py --model qwen3.6-35b-a3b --ctx 65536 --think limited \\
                       --non-interactive
    python install.py --list-models   # print the catalog and exit
    python install.py --gui           # the older Tkinter window

Afterwards, in any project folder:  opencode-local   (see start.py)

Two engines, picked automatically by hwdetect.pick_engine - see
docs/MODELS.md and docs/MACOS.md:
  llamacpp - Windows/Linux: llama-server serving GGUF weights.
  rapidmlx - macOS/Apple Silicon: rapid-mlx serving MLX weights.

What it does (idempotent - safe to re-run):
  1. Detects OS, GPU, total VRAM and RAM, free disk. Models are matched to
     TOTAL memory: browsers and file cache give memory back when the model
     loads, so what happens to be free right now doesn't change the pick.
  2. Asks for a model (catalog or Hugging Face link), context and thinking
     mode, or takes them from flags.
  3. Downloads (or reuses / updates) the newest llama.cpp build for this GPU
     and driver, and the GGUF.
  4. Writes settings.json and starts the server: llama.cpp --fit puts as many
     MoE expert layers in VRAM as fit, the rest in RAM.
  5. Runs tests/validate.py (unless --skip-tests).
  6. Installs OpenCode, points it at the server, installs the
     `opencode-local` start command, runs an agentic smoke test.
  7. Writes RESULTS.md.

Node (required for OpenCode) is never installed silently. Pass --install-node
(setup.cmd does) to opt into a first-party installer: winget on Windows,
Homebrew on macOS, the official nodejs.org tarball into ./node on Linux.
"""
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from installer import (catalog, download, hf, hwdetect, opencode_setup, rapidmlx_setup,
                       server, settings)

ROOT = os.path.dirname(os.path.abspath(__file__))


def die(msg):
    print(f"[STOP] {msg}", file=sys.stderr)
    sys.exit(1)


def gguf_is_valid(path):
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False


def verify_download_size(quant, path, log, fatal=True):
    """Best-effort size check against the HF API.

    Return ``True`` when the known size matches, ``False`` for a known
    mismatch, and ``None`` when the API cannot be reached. Existing valid
    GGUFs remain usable offline, while known partial files can be resumed.
    """
    try:
        url = f"https://huggingface.co/api/models/{quant['repo']}/tree/main?recursive=true"
        req = urllib.request.Request(url, headers={"User-Agent": "opencode-local-installer"})
        with urllib.request.urlopen(req, timeout=15) as r:
            entries = json.load(r)
        expected = next((e["size"] for e in entries if e.get("path") == quant["file"]), None)
        actual = os.path.getsize(path)
        if expected and actual != expected:
            message = (f"GGUF size mismatch for {quant['file']} (have {actual}, expect "
                       f"{expected}).")
            if fatal:
                die(message + " Re-run install.py to resume the download.")
            log(f"  {message} Resuming the partial download.")
            return False
        if expected:
            log(f"  model verified: size matches Hugging Face ({actual / 1e9:.1f} GB)")
            return True
        return None
    except (OSError, ValueError, urllib.error.URLError) as e:
        log(f"  size check skipped ({e})")
        return None


def installed_build(bin_dir):
    """llama.cpp build number of the llama-server in bin_dir, or None."""
    exe = server.find_server_binary(bin_dir) if os.path.isdir(bin_dir) else None
    if not exe:
        return None
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30,
                           encoding="utf-8", errors="replace")
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"build[: ]+(\d+)", (r.stdout or "") + (r.stderr or ""))
    return int(m.group(1)) if m else None


def latest_build(backend, cuda_driver=None):
    """(build_number, assets) of the newest llama.cpp release, or (None, None)."""
    try:
        assets = download.resolve_llamacpp_assets(backend, platform.system(), cuda_driver)
    except (OSError, RuntimeError, urllib.error.URLError, ValueError):
        return None, None
    m = re.fullmatch(r"b(\d+)", assets.get("tag") or "")
    return (int(m.group(1)) if m else None), assets


def ensure_llamacpp(bin_dir, backend, log, progress, cuda_driver=None, update=False):
    existing = server.find_server_binary(bin_dir) if os.path.isdir(bin_dir) else None
    if existing and update:
        have = installed_build(bin_dir)
        newest, _assets = latest_build(backend, cuda_driver)
        if have and newest and have >= newest:
            log(f"llama.cpp build {have} is already the newest")
            update = False
    if existing and not update:
        build = installed_build(bin_dir)
        log(f"Using existing llama.cpp{f' build {build}' if build else ''}: {existing}")
        return os.path.dirname(existing)

    log(f"Downloading the newest llama.cpp (backend={backend}, os={platform.system()})")
    try:
        assets = download.resolve_llamacpp_assets(backend, platform.system(), cuda_driver)
    except (OSError, RuntimeError, urllib.error.URLError) as e:
        if existing:
            log(f"  could not check for a newer llama.cpp ({e}); keeping the installed one")
            return os.path.dirname(existing)
        die(f"Could not resolve a prebuilt llama.cpp runtime: {e}")
    # Extract into a fresh folder, then swap, so a failed download never
    # leaves a half-replaced install behind.
    target = bin_dir + ".new" if existing else bin_dir
    if os.path.isdir(target):
        shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target, exist_ok=True)

    def dl(asset, label):
        dest = os.path.join(ROOT, asset["name"])

        def cb(done, total):
            progress(done, total, label)

        download.download_with_retries(asset["browser_download_url"], dest, progress_cb=cb)
        return dest

    bin_archive = dl(assets["binary"], "llama.cpp binary")
    download.extract_archive(bin_archive, target)
    os.remove(bin_archive)
    if assets["cudart"]:
        cudart_archive = dl(assets["cudart"], "CUDA runtime")
        download.extract_archive(cudart_archive, target)
        os.remove(cudart_archive)
    if not server.find_server_binary(target):
        die(f"llama-server not found after extracting the {assets['tag']} release")
    if existing:
        old = bin_dir + ".old"
        shutil.rmtree(old, ignore_errors=True)
        os.replace(bin_dir, old)
        os.replace(target, bin_dir)
        shutil.rmtree(old, ignore_errors=True)
    found = server.find_server_binary(bin_dir)
    log(f"Installed llama.cpp {assets['tag']} ({assets['binary']['name']}) -> "
        f"{os.path.dirname(found)}")
    return os.path.dirname(found)


def ensure_model(model, quant, model_path, log, progress):
    if model_path and gguf_is_valid(model_path):
        log(f"Using existing model file: {model_path}")
        return model_path
    if not model_path:
        model_dir = os.path.join(ROOT, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, *quant["file"].split("/"))
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path) and gguf_is_valid(model_path):
        size_ok = verify_download_size(quant, model_path, log, fatal=False)
        if size_ok is not False:
            log(f"Using existing model file: {model_path}")
            return model_path
        part_path = model_path + ".part"
        if os.path.exists(part_path):
            if os.path.getsize(model_path) > os.path.getsize(part_path):
                os.replace(model_path, part_path)
            else:
                os.remove(model_path)
        else:
            os.replace(model_path, part_path)

    log(f"Downloading {quant['file']} ({quant['size_gb']:.1f} GB) from {quant['repo']}")
    url = catalog.download_url(quant)

    def cb(done, total):
        progress(done, total, "model download")

    download.download_with_retries(url, model_path, progress_cb=cb)
    if not gguf_is_valid(model_path):
        die(f"Downloaded file at {model_path} is not a valid GGUF (bad magic).")
    verify_download_size(quant, model_path, log)
    return model_path


def _rel(path):
    """Path relative to the repo when inside it, so the folder can be moved."""
    path = os.path.abspath(path)
    try:
        rel = os.path.relpath(path, ROOT)
    except ValueError:  # other drive on Windows
        return path
    return path if rel.startswith("..") else rel


def run_pipeline_llamacpp(model, quant, profile, hw, skip_tests=False, port=8080, bin_dir=None,
                          model_path=None, install_node=False, stop_when_done=False,
                          backend=None, extra_server_args=None, thinking=None,
                          think_budget=None, update_llamacpp=False, allow_ps_scripts=None,
                          log=print, progress=lambda *a: None):
    """The GGUF/llama.cpp install pipeline (Windows/Linux), shared by the
    wizard, flags and GUI front-ends. See run_pipeline_mlx for macOS."""
    report = {}
    extra_server_args = list(extra_server_args or [])

    backend = backend or hwdetect.pick_llamacpp_backend(hw)
    if platform.system() == "Linux" and backend == "cuda":
        if not bin_dir or not os.path.isdir(bin_dir) or not server.find_server_binary(bin_dir):
            die("Linux CUDA requires a source-built llama-server because official llama.cpp "
                "releases do not provide a Linux CUDA binary. Pass both --backend cuda and "
                "--bin-dir /path/to/build/bin; see docs/DEPLOY.md.")
    bin_dir = bin_dir or os.path.join(ROOT, "llama.cpp")
    report["Backend"] = backend
    report["Hardware"] = "; ".join(hw.summary_lines())

    log("=== 1/6 llama.cpp ===")
    server.kill_existing_server(ROOT)
    resolved_bin_dir = ensure_llamacpp(bin_dir, backend, log, progress,
                                       cuda_driver=getattr(hw, "cuda_driver", None),
                                       update=update_llamacpp)

    log("=== 2/6 Model weights ===")
    resolved_model_path = ensure_model(model, quant, model_path, log, progress)

    log("=== 3/6 Starting llama-server ===")
    ctx = catalog.ctx_sizes(model)[profile]
    _memory_plan(model, quant, profile, hw, log)
    old = settings.load(ROOT) or {}
    s = dict(settings.DEFAULTS)
    s.update({
        "engine": "llamacpp",
        "model_id": model["id"],
        "display_name": model["display_name"],
        "arch": model["arch"],
        "reasoning": bool(model.get("reasoning", True)),
        "model_path": _rel(resolved_model_path),
        "bin_dir": _rel(resolved_bin_dir),
        "backend": backend,
        "port": port,
        "ctx": ctx,
        "thinking": thinking or old.get("thinking") or settings.DEFAULTS["thinking"],
        "think_budget": think_budget or old.get("think_budget") or
        settings.DEFAULTS["think_budget"],
        "extra_args": extra_server_args,
    })
    settings.save(ROOT, s)
    args = settings.server_args(s, ROOT)
    log(f"  llama-server {' '.join(args)}")
    proc = server.start_server(os.path.join(resolved_bin_dir, server.server_exe_name()), args,
                               os.path.join(ROOT, "server.log"), os.path.join(ROOT, "server.err"),
                               root=ROOT)
    with open(os.path.join(ROOT, ".server-state.json"), "w", encoding="utf-8") as f:
        json.dump({"pid": proc.pid, "args": args}, f)
    log("  waiting for /health (loading the model takes about a minute) ...")
    if not server.wait_for_health(port, timeout_s=900, process=proc):
        die("Server did not become healthy (or exited during startup) - see server.err")
    alias = server.get_served_alias(port)
    if alias != model["id"]:
        die(f"/v1/models reports '{alias}', expected '{model['id']}'")
    log(f"  healthy, alias={alias}, pid={proc.pid}")
    report["Model"] = f"{model['display_name']} / {quant['label']} ({resolved_model_path})"
    report["Server"] = settings.describe(s)

    log("=== 4/6 Writing config + start command ===")
    opencode_json = os.path.join(ROOT, "opencode.json")
    server.write_opencode_json(opencode_json, model, port, ctx)
    server.write_run_scripts(ROOT, sys.executable)
    log("  settings.json, opencode.json, run.ps1, run.sh written")

    log("=== 5/6 Validation ===")
    test_results = {}
    if not skip_tests:
        test_results = _run_validate(port, model["id"], log)
    else:
        log("  skipped (--skip-tests)")

    log("=== 6/6 OpenCode ===")
    oc_result = _setup_opencode(model["id"], port, install_node, log,
                                allow_ps_scripts=allow_ps_scripts, skip_smoke=skip_tests)
    test_results.update(oc_result)

    _write_results_md(report, test_results, log)

    if stop_when_done:
        log("Stopping server (--stop-when-done)")
        server.kill_existing_server(ROOT)
    else:
        log(f"Server left running on http://127.0.0.1:{port}/v1 (pid {proc.pid})")
    log("")
    log("Done. To code: open a terminal in your project folder and run   opencode-local")
    log("  thinking off for one session:  opencode-local --think off")
    log("  stop the model server:         opencode-local --stop")
    return {"report": report, "tests": test_results}


def _memory_plan(model, quant, profile, hw, log):
    """Say what the setup needs against TOTAL memory. Free memory is not
    checked: browsers and the file cache give memory back when the model
    loads, and llama.cpp --fit sizes the GPU part to what the GPU has."""
    need_vram, need_ram = catalog.estimate_requirements(model, quant, profile)
    have = []
    if hw.vram_total_gb:
        have.append(f"{hw.vram_total_gb:.0f} GB VRAM")
    if hw.ram_total_gb:
        have.append(f"{hw.ram_total_gb:.0f} GB RAM")
    log(f"  memory plan: needs ~{need_vram:.1f} GB VRAM minimum and ~{need_ram:.0f} GB RAM; "
        f"this PC has {' / '.join(have) or 'unknown memory'} in total.")
    if model["arch"] == "moe" and hw.vram_total_gb:
        log("  spare VRAM above the minimum holds expert layers (llama.cpp --fit): "
            "more VRAM = faster")
    if catalog.hw_fit_verdict(model, quant, profile, hw) == "no":
        log("  WARNING: this is more than the PC's total memory; it may fail to load. "
            "Choose a smaller context or quant.")


def _mlx_memory_preflight(model, quant, log):
    """Verify there's enough free unified memory to actually start the server,
    measured *now* (after killing any prior server) rather than from detect()'s
    startup snapshot.

    rapid-mlx admits a request only while active weights + projected KV stay
    under ~0.90 * the Metal working set, so a machine that's low on free memory
    doesn't hang - it loads and then 503s the first real request. Catching that
    here turns a confusing mid-validation failure into an upfront, actionable
    message. This warns (and, when clearly too low, aborts) but never silently
    proceeds into a run that can't succeed."""
    free = hwdetect.macos_available_ram_gb()
    weights = quant["size_gb"]
    need = catalog.estimate_mlx_requirements(model, quant)
    if free is None:
        log(f"  [preflight] couldn't read free memory; need ~{need:.1f} GB "
            f"(weights {weights:.1f} GB + KV/activations headroom). Continuing.")
        return
    log(f"  [preflight] free unified memory ~{free:.1f} GB; this model wants "
        f"~{need:.1f} GB (weights {weights:.1f} GB + KV/activations headroom).")
    # Hard floor: without room for the weights plus a minimal working margin the
    # server cannot even load without heavy paging, and every request will 503.
    hard_floor = weights + 1.5
    if free < hard_floor:
        die(f"Not enough free memory to start: ~{free:.1f} GB free, but "
            f"{quant['repo']} needs at least ~{hard_floor:.1f} GB just to load "
            f"its weights. Close other apps (a browser or another model server "
            f"is the usual culprit), or re-run with a smaller model/quant "
            f"(`python install.py --list-models`). Free memory is measured live, "
            f"so freeing some and re-running is enough - the installer is idempotent.")
    if free < need:
        log(f"  [preflight] WARNING: only ~{free:.1f} GB free vs ~{need:.1f} GB "
            f"wanted. Small requests will work, but long prompts or large "
            f"max_tokens may come back as HTTP 503 (KV cache would exceed the "
            f"Metal cap). Close some apps or pick a smaller quant for headroom.")


def run_pipeline_mlx(model, quant, hw, skip_tests=False, port=8000, install_node=False,
                      stop_when_done=False, log=print,
                      progress=lambda *a: None):
    """The rapid-mlx install pipeline (macOS/Apple Silicon only). Unlike
    llama.cpp, rapid-mlx manages its own weight download (see
    installer/rapidmlx_setup.py), so there's no ensure_llamacpp/ensure_model
    equivalent here - just "make sure the CLI is installed" then "serve"."""
    report = {"Backend": "rapidmlx", "Hardware": "; ".join(hw.summary_lines())}

    log("=== 1/5 rapid-mlx CLI ===")
    exe = rapidmlx_setup.ensure_rapidmlx(ROOT, log)
    if not exe:
        die("Could not install/locate rapid-mlx. Check that `python3 -m venv` works, "
            "then re-run the installer; it creates a project-local .rapidmlx-venv.")

    log("=== 2/5 Starting rapid-mlx server ===")
    rapidmlx_setup.kill_existing(ROOT)
    _mlx_memory_preflight(model, quant, log)
    args = rapidmlx_setup.build_serve_args(quant["repo"], port)
    log(f"  rapid-mlx {' '.join(args)}  (first run downloads the model from Hugging Face)")
    proc = rapidmlx_setup.start_rapidmlx(exe, args, os.path.join(ROOT, "server.log"),
                                          os.path.join(ROOT, "server.err"))
    rapidmlx_setup.write_pid(ROOT, proc.pid)
    log("  waiting for the server to come up (this includes the model download - can take a while) ...")
    if not server.wait_for_health(port, timeout_s=1800, path="/v1/models", process=proc):
        die("Server did not become healthy (or exited during startup) - see server.log / server.err")
    try:
        alias = server.get_served_alias(port)
    except Exception as e:
        die(f"Server is up but /v1/models didn't return a usable response: {e}")
    if alias != quant["repo"]:
        log(f"  [WARN] /v1/models reports '{alias}', expected '{quant['repo']}' - rapid-mlx's "
            f"exact model-id echoing wasn't verified against real hardware for this installer "
            f"(see installer/rapidmlx_setup.py); continuing with the reported id.")
        alias = alias or quant["repo"]
    log(f"  healthy, alias={alias}, pid={proc.pid}")
    # rapid-mlx's CLI surface (as documented) has no --ctx-size equivalent - reuse this
    # model's GGUF "primary" context as a reasonable opencode.json default; adjust by hand
    # if rapid-mlx's actual max context for this model differs.
    ctx = catalog.ctx_sizes(model)["primary"]
    report["Model"] = f"{model['display_name']} / {quant['label']} ({quant['repo']})"
    report["Server"] = f"port {port}, ctx {ctx} (assumed)"

    log("=== 3/5 Writing config + run script ===")
    opencode_json = os.path.join(ROOT, "opencode.json")
    server.write_opencode_json(opencode_json, model, port, ctx, provider_key="rapidmlx",
                                provider_label="rapid-mlx (local)", served_model_id=alias)
    rapidmlx_setup.write_run_script(ROOT, quant["repo"], port)
    log("  opencode.json, run.sh written")

    log("=== 4/5 Validation ===")
    test_results = {}
    if not skip_tests:
        test_results = _run_validate(port, alias, log)
    else:
        log("  skipped (--skip-tests)")

    log("=== 5/5 OpenCode ===")
    oc_result = _setup_opencode(alias, port, install_node, log, provider_key="rapidmlx")
    test_results.update(oc_result)

    _write_results_md(report, test_results, log)

    if stop_when_done:
        log("Stopping server (--stop-when-done)")
        rapidmlx_setup.kill_existing(ROOT)
    else:
        log(f"Server left running on http://127.0.0.1:{port}/v1 (pid {proc.pid})")

    return {"report": report, "tests": test_results}


def _run_validate(port, model_id, log):
    import subprocess
    env = dict(os.environ)
    env["LLAMA_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["LLAMA_MODEL"] = model_id
    python_exe = sys.executable
    # encoding/errors explicit: on Windows, text=True alone decodes with the
    # console's active code page (e.g. cp1252), which crashes on non-ASCII
    # bytes llama.cpp/validate.py may print (seen in testing: a stray 0x90).
    r = subprocess.run([python_exe, os.path.join(ROOT, "tests", "validate.py")],
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                        env=env, timeout=1200)
    log(r.stdout)
    if r.stderr:
        log(r.stderr)
    results = {}
    for line in r.stdout.splitlines():
        if line.startswith("[PASS]") or line.startswith("[FAIL]"):
            name = line.split()[1].rstrip(":")
            results[name] = "PASS" if line.startswith("[PASS]") else "FAIL"
    return results


def _setup_opencode(model_id, port, install_node, log, provider_key="llamacpp",
                    allow_ps_scripts=None, skip_smoke=False):
    results = {}
    node = opencode_setup.find_node()
    if not node:
        # Node may already be installed in a terminal opened before its PATH change
        opencode_setup.refresh_windows_path()
        node = opencode_setup.find_node()
    if not node:
        if install_node:
            node = opencode_setup.try_install_node(log)
        if not node:
            log(f"Node not found - skipping OpenCode. Install it with "
                f"{opencode_setup.node_install_hint()}")
            return results
    log("Installing OpenCode (npm) ...")
    opencode_setup.npm_install_opencode(log)
    dest = opencode_setup.install_config(os.path.join(ROOT, "opencode.json"))
    log(f"  OpenCode config -> {dest}")
    if not opencode_setup.ensure_opencode_on_path(log):
        results["opencode-agentic"] = "FAIL"
        return results
    if provider_key == "llamacpp":
        opencode_setup.install_launcher(ROOT, sys.executable, log)
    if opencode_setup.powershell_scripts_blocked():
        if allow_ps_scripts:
            opencode_setup.allow_powershell_scripts(log)
        else:
            log("  NOTE: PowerShell blocks scripts on this PC, so a plain `opencode` fails; "
                "`opencode-local` and `opencode.cmd` work. To allow it: "
                "Set-ExecutionPolicy -Scope CurrentUser RemoteSigned")
    if skip_smoke:
        return results
    scratch = os.path.join(ROOT, "opencode-scratch")
    alias = f"{provider_key}/{model_id}"
    log("  warm-up run (first run downloads ripgrep once - needs internet)")
    opencode_setup.warm_up(scratch, alias)
    log("  agentic smoke test (write + run a python file)")
    ok = opencode_setup.agentic_smoke_test(scratch, alias)
    results["opencode-agentic"] = "PASS" if ok else "FAIL"
    log(f"  opencode-agentic: {'PASS' if ok else 'FAIL (see opencode-scratch/)'}")
    return results


def _write_results_md(report, test_results, log):
    lines = ["# RESULTS - install.py run on this machine", ""]
    lines.append("## Environment")
    for k, v in report.items():
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    lines.append("## Pass / fail")
    lines.append("| Test | Result |")
    lines.append("|---|---|")
    for k, v in test_results.items():
        lines.append(f"| {k} | {v} |")
    path = os.path.join(ROOT, "RESULTS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gui", action="store_true", help="use the Tkinter window instead of the "
                                                      "text wizard")
    p.add_argument("--cli", action="store_true", help="text wizard (the default)")
    p.add_argument("--hf", metavar="LINK", help="any GGUF model from Hugging Face: a "
                   "https://huggingface.co/OWNER/NAME link (a file link picks that file) "
                   "or OWNER/NAME")
    p.add_argument("--hf-file", metavar="FILE", help="with --hf: which .gguf file "
                   "(default: the Q4_K_M-class one)")
    p.add_argument("--think", choices=settings.THINK_MODES, default=None,
                   help="thinking: limited (default, capped by --think-budget), off, on "
                        "(unlimited). Change later with opencode-local --think ...")
    p.add_argument("--think-budget", type=int, default=None, metavar="TOKENS",
                   help="token cap for --think limited (default 4096)")
    p.add_argument("--update-llamacpp", action="store_true",
                   help="replace an existing llama.cpp folder with the newest build "
                        "(skipped when it is already the newest)")
    p.add_argument("--allow-ps-scripts", action="store_true",
                   help="Windows: let PowerShell run local scripts for this user "
                        "(RemoteSigned) so a plain `opencode` works")
    p.add_argument("--log", metavar="FILE", help="also write the output to FILE")
    p.add_argument("--list-models", action="store_true", help="print the catalog and exit")
    p.add_argument("--model", help="catalog model id, e.g. qwen3.6-35b-a3b (see --list-models), "
                                    "or 'auto' for the one that best fits this PC")
    p.add_argument("--quant", help="quant GGUF filename (llama.cpp) or mlx-community repo id "
                                    "(rapid-mlx/macOS) - see --list-models")
    p.add_argument("--profile", choices=["auto", "primary", "conservative"], default="auto",
                    help="llama.cpp context profile; auto picks the largest context that fits total VRAM with a 1 GB reserve")
    p.add_argument("--ctx", type=int, default=None, metavar="TOKENS",
                    help="context size, e.g. 65536 or 262144 (default: by total VRAM - "
                         "64K below 16 GB, 128K below 24 GB, 256K above, for MoE models). "
                         "Also sets the OpenCode context limit")
    p.add_argument("--port", type=int, default=None,
                    help="default 8080 for llama.cpp, 8000 for rapid-mlx")
    p.add_argument("--bin-dir", default=None, help="llama.cpp only: reuse an existing llama.cpp folder")
    p.add_argument("--backend", choices=["auto", "cuda", "vulkan", "rocm", "cpu"],
                    default="auto", help="llama.cpp backend; auto uses hardware/OS defaults. "
                                         "Use cuda with a custom Linux CUDA --bin-dir")
    p.add_argument("--extra-server-arg", action="append", default=[], metavar="ARG",
                    help="llama.cpp only: append and persist one server argument; repeat as needed. "
                         "Use --extra-server-arg=--flag for values beginning with a dash")
    p.add_argument("--model-path", default=None,
                    help="llama.cpp only: reuse an existing GGUF instead of downloading")
    p.add_argument("--skip-tests", action="store_true")
    p.add_argument("--stop-when-done", action="store_true")
    p.add_argument("--install-node", action="store_true",
                   help="auto-install Node if missing: winget (Windows), Homebrew (macOS), "
                        "official nodejs.org tarball into ./node (Linux, no sudo)")
    p.add_argument("--non-interactive", action="store_true",
                    help="never prompt; needs --model or --hf")
    return p.parse_args()


def print_catalog():
    for model in catalog.MODELS:
        print(f"\n{model['id']}  ({model['display_name']}, {model['arch']})")
        print("  llama.cpp (GGUF, Windows/Linux):")
        for q in model["quants"]:
            tag = " [default]" if q.get("default") else ""
            print(f"    {q['file']:45} {q['size_gb']:6.1f} GB  {q['label']}{tag}")
        print("  rapid-mlx (MLX, macOS/Apple Silicon):")
        for q in model.get("mlx", []):
            tag = " [default]" if q.get("default") else ""
            print(f"    {q['repo']:45} {q['size_gb']:6.1f} GB  {q['label']}{tag}")


def main():
    args = parse_args()
    if args.log:
        _tee_output(args.log)
    if args.list_models:
        print_catalog()
        return

    hw = hwdetect.detect(ROOT)
    engine = hwdetect.pick_engine(hw)
    port = args.port if args.port is not None else (8000 if engine == "rapidmlx" else 8080)
    args.allow_ps_scripts = args.allow_ps_scripts or None
    args.update_llamacpp = bool(args.update_llamacpp)

    model = quant = None
    if args.hf:
        if engine == "rapidmlx":
            die("--hf installs GGUF models for llama.cpp (Windows/Linux); on a Mac pick a "
                "catalog model.")
        try:
            model, quant, _ggufs = hf.resolve(args.hf, args.hf_file)
        except (ValueError, OSError) as e:
            die(str(e))
        print(f"Hugging Face model: {model['display_name']} -> {quant['file']} "
              f"({quant['size_gb']:.1f} GB, {model['arch']})")
    elif args.model:
        if args.model == "auto":
            if engine == "rapidmlx":
                die("--model auto is for llama.cpp (Windows/Linux); on a Mac pick a model.")
            model, _q = catalog.recommended_model(hw)
            print(f"Best fit for this PC: {model['display_name']}")
        else:
            model = catalog.get_model(args.model)
        if engine == "rapidmlx":
            quant = (catalog.get_mlx_quant(model, args.quant) if args.quant
                     else catalog.default_mlx_quant(model))
        else:
            quant = catalog.get_quant(model, args.quant) if args.quant else catalog.default_quant(model)

    if model is None:
        if args.non_interactive:
            die("--non-interactive requires --model or --hf")
        if args.gui:
            from installer import tk_setup
            gui_ready, detail = tk_setup.ensure_tkinter(print)
            if not gui_ready:
                die(detail)
            from installer import gui

            def pipeline_for_gui(m, q, prof, hw_, skip_tests, install_node, log, progress):
                args.install_node = install_node
                if args.ctx:
                    m = catalog.with_ctx(m, args.ctx)
                _dispatch(m, q, prof, hw_, engine, port, args, skip_tests, log, progress)

            gui.run_gui(hw, pipeline_for_gui)
            return
        from installer import cli
        if engine == "rapidmlx":
            model, quant = cli.choose_mlx_model_quant(hw)
        else:
            bin_dir = args.bin_dir or os.path.join(ROOT, "llama.cpp")
            update_info = None
            have = installed_build(bin_dir)
            if have and not args.bin_dir:
                newest, _assets = latest_build(args.backend if args.backend != "auto" else
                                               hwdetect.pick_llamacpp_backend(hw),
                                               getattr(hw, "cuda_driver", None))
                if newest and newest > have:
                    update_info = (have, newest)
            choice = cli.run_wizard(hw, update_info=update_info,
                                    scripts_blocked=opencode_setup.powershell_scripts_blocked())
            model, quant = choice["model"], choice["quant"]
            args.ctx = args.ctx or choice["ctx"]
            args.think = args.think or choice["thinking"]
            args.think_budget = args.think_budget or choice["think_budget"]
            args.update_llamacpp = args.update_llamacpp or choice["update_llamacpp"]
            args.allow_ps_scripts = choice["allow_ps_scripts"]

    profile = None
    if engine != "rapidmlx":
        ctx = args.ctx or catalog.default_ctx(model, hw)
        model = catalog.with_ctx(model, ctx)
        profile = "primary"
        print(f"Context: {ctx:,} tokens.")
        if catalog.hw_fit_verdict(model, quant, profile, hw) == "no":
            print("  [WARN] estimated to need more than this PC's total memory at this "
                  "context; llama-server may fail to load. Try a smaller --ctx or quant.")
    _dispatch(model, quant, profile, hw, engine, port, args, args.skip_tests, print,
              console_progress())


def console_progress():
    """Download progress on one console line, at most once a second."""
    state = {"t": 0.0}

    def cb(done, total, label):
        now = time.time()
        finished = bool(total) and done >= total
        if now - state["t"] < 1 and not finished:
            return
        state["t"] = now
        if total:
            line = f"  {label}: {done / 1e9:5.1f} / {total / 1e9:.1f} GB ({done * 100 // total}%)"
        else:
            line = f"  {label}: {done / 1e9:5.1f} GB"
        print("\r" + line + "   ", end="\n" if finished else "", flush=True)

    return cb


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stream.flush()

    def isatty(self):
        return self.stream.isatty()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _tee_output(path):
    """--log: copy everything printed to a file, keeping the console a real
    terminal (prompts and the progress line still work)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"\n===== install.py {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(sys.argv[1:])}\n")
    sys.stdout = _Tee(sys.stdout, path)
    sys.stderr = _Tee(sys.stderr, path)


def _dispatch(model, quant, profile, hw, engine, port, args, skip_tests, log, progress):
    """Route to the right pipeline. engine comes from hwdetect.pick_engine(hw)
    (computed once in main()) rather than re-derived from `profile is None`,
    so a caller can't accidentally send an mlx-shaped quant down the
    llama.cpp path by passing a profile."""
    if engine == "rapidmlx":
        if args.backend != "auto" or args.extra_server_arg:
            die("--backend and --extra-server-arg apply only to llama.cpp (Windows/Linux), "
                "not the macOS Rapid-MLX engine.")
        run_pipeline_mlx(model, quant, hw, skip_tests=skip_tests, port=port,
                          install_node=args.install_node, stop_when_done=args.stop_when_done,
                          log=log, progress=progress)
    else:
        selected_backend = (None if args.backend == "auto" else args.backend)
        run_pipeline_llamacpp(model, quant, profile, hw, skip_tests=skip_tests, port=port,
                              bin_dir=args.bin_dir, model_path=args.model_path,
                              install_node=args.install_node, stop_when_done=args.stop_when_done,
                              backend=selected_backend, extra_server_args=args.extra_server_arg,
                              thinking=getattr(args, "think", None),
                              think_budget=getattr(args, "think_budget", None),
                              update_llamacpp=getattr(args, "update_llamacpp", False),
                              allow_ps_scripts=getattr(args, "allow_ps_scripts", None),
                              log=log, progress=progress)


if __name__ == "__main__":
    main()
