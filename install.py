#!/usr/bin/env python3
"""install.py - one-shot cross-platform installer + validator for the local
LLM + OpenCode coding-agent stack (Windows, Linux, and macOS/Apple Silicon).

    python install.py                 # GUI wizard (falls back to text if no Tkinter)
    python install.py --cli           # force the text wizard
    python install.py --model qwen3.6-35b-a3b --quant Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \\
                       --profile primary --non-interactive
    python install.py --list-models   # print the catalog (both engines) and exit

Two engines, picked automatically by hwdetect.pick_engine - see
docs/MODELS.md and docs/MACOS.md:
  llamacpp - Windows/Linux: llama-server serving GGUF weights.
  rapidmlx - macOS/Apple Silicon: rapid-mlx serving MLX weights, validated
             on Apple Silicon with native K8V4 TurboQuant KV cache.

What it does (idempotent - safe to re-run):
  1. Detects OS, GPU/unified-memory, VRAM, RAM, free disk.
  2. Lets you pick a model + quantization (GUI/CLI, filtered by what plausibly
     fits) or takes it from --model/--quant/--profile.
  3. llamacpp: downloads (or reuses) a matching prebuilt llama.cpp release and
     the GGUF. rapidmlx: installs the rapid-mlx CLI (pip); it downloads its
     own weights on first `serve`.
  4. Starts the server, waits for it to become healthy, checks the /v1/models
     alias.
  5. Runs tests/validate.py (unless --skip-tests).
  6. Installs OpenCode, writes opencode.json, runs an agentic smoke test.
  7. Writes RESULTS.md.

Node (required for OpenCode) is never installed silently. Pass --install-node
(the GUI checks this by default) to opt into a first-party installer: winget on
Windows, Homebrew on macOS, the official nodejs.org tarball into a project-local
./node on Linux (no sudo) - see installer/opencode_setup.py.
"""
import argparse
import json
import os
import platform
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from installer import catalog, download, hwdetect, opencode_setup, rapidmlx_setup, server

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
        url = f"https://huggingface.co/api/models/{quant['repo']}/tree/main"
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


def ensure_llamacpp(bin_dir, backend, log, progress):
    existing = server.find_server_binary(bin_dir) if os.path.isdir(bin_dir) else None
    if existing:
        log(f"Using existing llama.cpp binary: {existing}")
        return os.path.dirname(existing)

    log(f"Downloading prebuilt llama.cpp release (backend={backend}, os={platform.system()})")
    try:
        assets = download.resolve_llamacpp_assets(backend, platform.system())
    except (OSError, RuntimeError, urllib.error.URLError) as e:
        die(f"Could not resolve a prebuilt llama.cpp runtime: {e}")
    os.makedirs(bin_dir, exist_ok=True)

    def dl(asset, label):
        dest = os.path.join(ROOT, asset["name"])

        def cb(done, total):
            progress(done, total, label)

        download.download_with_retries(asset["browser_download_url"], dest, progress_cb=cb)
        return dest

    bin_archive = dl(assets["binary"], "llama.cpp binary")
    download.extract_archive(bin_archive, bin_dir)
    os.remove(bin_archive)
    if assets["cudart"]:
        cudart_archive = dl(assets["cudart"], "CUDA runtime")
        download.extract_archive(cudart_archive, bin_dir)
        os.remove(cudart_archive)

    found = server.find_server_binary(bin_dir)
    if not found:
        die(f"llama-server not found after extracting the {assets['tag']} release into {bin_dir}")
    log(f"Installed llama.cpp {assets['tag']} -> {os.path.dirname(found)}")
    return os.path.dirname(found)


def ensure_model(model, quant, model_path, log, progress):
    if model_path and gguf_is_valid(model_path):
        log(f"Using existing model file: {model_path}")
        return model_path
    if not model_path:
        model_dir = os.path.join(ROOT, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, quant["file"])
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


def run_pipeline_llamacpp(model, quant, profile, hw, skip_tests=False, port=8080, bin_dir=None,
                          model_path=None, install_node=False, stop_when_done=False,
                          backend=None, extra_server_args=None,
                          log=print, progress=lambda *a: None):
    """The GGUF/llama.cpp install pipeline (Windows/Linux), shared by the CLI
    and GUI front-ends. See run_pipeline_mlx for the macOS/Apple Silicon
    counterpart."""
    report = {}
    extra_server_args = list(extra_server_args or [])
    is_turboquant = quant.get("engine") == "turboquant"
    if is_turboquant:
        if backend is None:
            die("TurboQuant uses a custom runtime whose backend cannot be inferred. "
                "Pass its real backend explicitly, for example --backend cuda, together "
                "with --bin-dir; see docs/TURBOQUANT.md.")
        if not bin_dir:
            die("TurboQuant weights require a reviewed TQ-enabled llama.cpp build. "
                "Pass --bin-dir /path/to/that/build and the build's real backend with "
                "--backend. Stock llama.cpp cannot load TQ3_1S; see docs/TURBOQUANT.md.")
        if not os.path.isdir(bin_dir) or not server.find_server_binary(bin_dir):
            die(f"TurboQuant requires --bin-dir to contain a llama-server binary; none "
                f"was found under {bin_dir!r}. Build or install a reviewed TQ-enabled "
                "fork there first. This installer never downloads a third-party runtime; "
                "see docs/TURBOQUANT.md.")
        log("TurboQuant weights selected. Using the explicitly supplied third-party "
            "runtime; this installer never downloads one automatically.")

    backend = backend or hwdetect.pick_llamacpp_backend(hw)
    if platform.system() == "Linux" and backend == "cuda":
        if not bin_dir or not os.path.isdir(bin_dir) or not server.find_server_binary(bin_dir):
            die("Linux CUDA requires a source-built llama-server because official llama.cpp "
                "releases do not provide a Linux CUDA binary. Pass both --backend cuda and "
                "--bin-dir /path/to/build/bin; see docs/DEPLOY.md.")
    bin_dir = bin_dir or os.path.join(ROOT, "llama.cpp")
    report["Backend"] = backend
    report["Hardware"] = "; ".join(hw.summary_lines())

    log("=== 1/6 llama.cpp binary ===")
    resolved_bin_dir = ensure_llamacpp(bin_dir, backend, log, progress)

    log("=== 2/6 Model weights ===")
    resolved_model_path = ensure_model(model, quant, model_path, log, progress)

    log("=== 3/6 Starting llama-server ===")
    server.kill_existing_server(ROOT)
    ctx = catalog.ctx_sizes(model)[profile]
    args = server.build_server_args(model, quant, resolved_model_path, port, profile,
                                    backend=backend, extra_args=extra_server_args)
    log(f"  llama-server {' '.join(args)}")
    proc = server.start_server(os.path.join(resolved_bin_dir, server.server_exe_name()), args,
                               os.path.join(ROOT, "server.log"), os.path.join(ROOT, "server.err"),
                               root=ROOT)
    log("  waiting for /health ...")
    if not server.wait_for_health(port, process=proc):
        die("Server did not become healthy (or exited during startup) - see server.log / server.err")
    alias = server.get_served_alias(port)
    if alias != model["id"]:
        die(f"/v1/models reports '{alias}', expected '{model['id']}'")
    log(f"  healthy, alias={alias}, pid={proc.pid}")
    report["Model"] = f"{model['display_name']} / {quant['label']} ({resolved_model_path})"
    report["Server"] = f"port {port}, ctx {ctx}, profile {profile}"

    log("=== 4/6 Writing config + run scripts ===")
    opencode_json = os.path.join(ROOT, "opencode.json")
    server.write_opencode_json(opencode_json, model, port, ctx)
    server.write_run_scripts(ROOT, resolved_bin_dir, resolved_model_path, model["id"],
                              model["arch"], port, ctx, "q8_0", backend,
                              extra_args=extra_server_args)
    log("  opencode.json, run.ps1, run.sh written")

    log("=== 5/6 Validation ===")
    test_results = {}
    if not skip_tests:
        test_results = _run_validate(port, model["id"], log)
    else:
        log("  skipped (--skip-tests)")

    log("=== 6/6 OpenCode ===")
    oc_result = _setup_opencode(model["id"], port, install_node, log)
    test_results.update(oc_result)

    _write_results_md(report, test_results, log)

    if stop_when_done:
        log("Stopping server (--stop-when-done)")
        server.kill_existing_server(ROOT)
    else:
        log(f"Server left running on http://127.0.0.1:{port}/v1 (pid {proc.pid})")

    return {"report": report, "tests": test_results}


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
                      stop_when_done=False, mlx_turboquant="k8v4", log=print,
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
    args = rapidmlx_setup.build_serve_args(quant["repo"], port, mlx_turboquant)
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
    report["Server"] = (f"port {port}, ctx {ctx} (assumed), "
                        f"Rapid-MLX TurboQuant={mlx_turboquant}")

    log("=== 3/5 Writing config + run script ===")
    opencode_json = os.path.join(ROOT, "opencode.json")
    server.write_opencode_json(opencode_json, model, port, ctx, provider_key="rapidmlx",
                                provider_label="rapid-mlx (local)", served_model_id=alias)
    rapidmlx_setup.write_run_script(ROOT, quant["repo"], port, mlx_turboquant)
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


def _setup_opencode(model_id, port, install_node, log, provider_key="llamacpp"):
    results = {}
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
    log(f"  opencode.json -> {dest}")
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
    p.add_argument("--gui", action="store_true", help="force the GUI wizard")
    p.add_argument("--cli", action="store_true", help="force the text wizard")
    p.add_argument("--list-models", action="store_true", help="print the catalog and exit")
    p.add_argument("--model", help="catalog model id, e.g. qwen3.6-35b-a3b (see --list-models)")
    p.add_argument("--quant", help="quant GGUF filename (llama.cpp) or mlx-community repo id "
                                    "(rapid-mlx/macOS) - see --list-models")
    p.add_argument("--profile", choices=["auto", "primary", "conservative"], default="auto",
                    help="llama.cpp context profile; auto keeps a 1 GB free-VRAM reserve")
    p.add_argument("--mlx-turboquant", choices=["k8v4", "v4", "none"], default="k8v4",
                    help="macOS only: Rapid-MLX KV cache mode (default: tested k8v4)")
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
                    help="require --model/--quant, never prompt")
    return p.parse_args()


def print_catalog():
    for model in catalog.MODELS:
        print(f"\n{model['id']}  ({model['display_name']}, {model['arch']})")
        print("  llama.cpp (GGUF, Windows/Linux):")
        for q in model["quants"]:
            tag = " [default]" if q.get("default") else ""
            tag += " [experimental]" if q.get("experimental") else ""
            print(f"    {q['file']:45} {q['size_gb']:6.1f} GB  {q['label']}{tag}")
        print("  rapid-mlx (MLX, macOS/Apple Silicon):")
        for q in model.get("mlx", []):
            tag = " [default]" if q.get("default") else ""
            print(f"    {q['repo']:45} {q['size_gb']:6.1f} GB  {q['label']}{tag}")


def main():
    args = parse_args()
    if args.list_models:
        print_catalog()
        return

    hw = hwdetect.detect(ROOT)
    engine = hwdetect.pick_engine(hw)
    port = args.port if args.port is not None else (8000 if engine == "rapidmlx" else 8080)

    model = quant = None
    if args.model:
        model = catalog.get_model(args.model)
        if engine == "rapidmlx":
            quant = (catalog.get_mlx_quant(model, args.quant) if args.quant
                     else catalog.default_mlx_quant(model))
        else:
            quant = catalog.get_quant(model, args.quant) if args.quant else catalog.default_quant(model)

    if model is None:
        if args.non_interactive:
            die("--non-interactive requires --model (and optionally --quant)")
        use_gui = args.gui or (not args.cli)
        if use_gui:
            from installer import tk_setup
            gui_ready, detail = tk_setup.ensure_tkinter(print)
            if not gui_ready:
                if args.gui:
                    die(detail)
                print(f"{detail}\nFalling back to the text wizard.")
                use_gui = False
        if use_gui:
            from installer import gui

            def pipeline_for_gui(m, q, prof, hw_, skip_tests, install_node, log, progress):
                args.install_node = install_node
                _dispatch(m, q, prof, hw_, engine, port, args, skip_tests, log, progress)

            gui.run_gui(hw, pipeline_for_gui)
            return
        else:
            from installer import cli
            if engine == "rapidmlx":
                model, quant = cli.choose_mlx_model_quant(hw)
            else:
                model, quant, args.profile = cli.choose_model_quant(hw)

    if engine != "rapidmlx" and args.profile == "auto":
        args.profile = catalog.recommended_profile(model, quant, hw)
        print(f"Auto-selected {args.profile} context profile "
              f"({catalog.ctx_sizes(model)[args.profile]:,} tokens with VRAM reserve).")
    profile = None if engine == "rapidmlx" else args.profile
    _dispatch(model, quant, profile, hw, engine, port, args, args.skip_tests, print,
              lambda *a: None)


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
                          mlx_turboquant=args.mlx_turboquant, log=log, progress=progress)
    else:
        selected_backend = (None if args.backend == "auto" else args.backend)
        run_pipeline_llamacpp(model, quant, profile, hw, skip_tests=skip_tests, port=port,
                              bin_dir=args.bin_dir, model_path=args.model_path,
                              install_node=args.install_node, stop_when_done=args.stop_when_done,
                              backend=selected_backend, extra_server_args=args.extra_server_arg,
                              log=log, progress=progress)


if __name__ == "__main__":
    main()
