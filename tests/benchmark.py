#!/usr/bin/env python3
"""Run a capped llama-bench comparison and write machine-readable results.

The defaults model an 8 GiB GPU with 1 GiB total-card headroom and a 32 GiB
host with roughly 7 GiB left for the OS. No third-party Python packages are
required; NVIDIA memory sampling uses nvidia-smi when available.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MIB = 1024 * 1024
DEFAULT_PROCESS_VRAM_MIB = 6656
DEFAULT_TOTAL_VRAM_MIB = 7168
DEFAULT_RSS_MIB = 25600


def run_text(args, timeout=10):
    try:
        return subprocess.run(args, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=timeout).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def gpu_total_mib():
    text = run_text(["nvidia-smi", "--query-gpu=memory.used",
                     "--format=csv,noheader,nounits"])
    try:
        return int(text.splitlines()[0].strip())
    except (IndexError, ValueError):
        return None


def gpu_process_mib(pid):
    text = run_text(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                     "--format=csv,noheader,nounits"])
    for row in text.splitlines():
        fields = [field.strip() for field in row.split(",")]
        if len(fields) == 2 and fields[0] == str(pid):
            try:
                return int(fields[1])
            except ValueError:
                pass
    return None

def process_rss_mib(pid):
    if platform.system() == "Linux":
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        except OSError:
            return None
    if platform.system() == "Windows":
        text = run_text([
            "powershell", "-NoProfile", "-Command",
            f"(Get-Process -Id {pid} -ErrorAction SilentlyContinue).WorkingSet64",
        ])
        try:
            return int(text) / MIB
        except ValueError:
            return None
    return None


def available_ram_mib():
    if platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
        except OSError:
            return None
    return None


def parse_env(items):
    values = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--env must be KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        if not key:
            raise ValueError("--env key cannot be empty")
        values[key] = value
    return values


def build_command(args):
    command = [
        str(Path(args.bench_bin).resolve()),
        "-m", str(Path(args.model_path).resolve()),
        "-p", str(args.prompt_tokens),
        "-n", str(args.generation_tokens),
        "-r", str(args.repetitions),
        "-o", "json",
        "-ngl", "999",
        "-ncmoe", str(args.n_cpu_moe),
        "-ctk", args.cache_type_k,
        "-ctv", args.cache_type_v,
        "-fa", "on",
        "-t", str(args.threads),
    ]
    if args.device:
        command += ["-dev", args.device]
    command += args.extra_arg
    return command


def sample_resources(pid, baseline_total, peaks):
    total = gpu_total_mib()
    process = gpu_process_mib(pid)
    source = "per-process"
    if process is None and total is not None and baseline_total is not None:
        process = max(0, total - baseline_total)
        source = "total-delta"
    rss = process_rss_mib(pid)
    available = available_ram_mib()
    if process is not None:
        peaks["process_vram_mib"] = max(peaks["process_vram_mib"], process)
    if total is not None:
        peaks["total_vram_mib"] = max(peaks["total_vram_mib"], total)
    if rss is not None:
        peaks["rss_mib"] = max(peaks["rss_mib"], rss)
    if available is not None:
        peaks["minimum_available_ram_mib"] = min(
            peaks["minimum_available_ram_mib"], available)
    peaks["vram_source"] = source
    return process, total, rss

def execute_benchmark(args):
    env_overrides = parse_env(args.env)
    if "TQ4" in args.quant.upper() and env_overrides.get("GGML_TQ_NATIVE") != "1":
        raise ValueError("TQ4 requires --env GGML_TQ_NATIVE=1; default q8 conversion "
                         "invalidates an 8 GiB VRAM comparison")
    if "TQ3" in args.quant.upper() and args.n_cpu_moe > 0:
        print("[WARN] TQ3 CPU MoE kernels are known to be slow in the selected fork.",
              file=sys.stderr)

    command = build_command(args)
    environment = dict(os.environ)
    environment.update(env_overrides)
    baseline_total = gpu_total_mib()
    peaks = {
        "process_vram_mib": 0,
        "total_vram_mib": baseline_total or 0,
        "rss_mib": 0,
        "minimum_available_ram_mib": float("inf"),
        "vram_source": "unavailable",
    }
    breach = None
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="localcode-bench-") as tmp:
        stdout_path = Path(tmp) / "stdout.json"
        stderr_path = Path(tmp) / "stderr.log"
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(command, stdout=stdout_file, stderr=stderr_file,
                                       env=environment)
            while process.poll() is None:
                process_vram, total_vram, rss = sample_resources(
                    process.pid, baseline_total, peaks)
                if process_vram is not None and process_vram > args.max_process_vram_mib:
                    breach = (f"process VRAM {process_vram} MiB exceeded "
                              f"{args.max_process_vram_mib} MiB")
                elif total_vram is not None and total_vram > args.max_total_vram_mib:
                    breach = (f"total GPU use {total_vram} MiB exceeded "
                              f"{args.max_total_vram_mib} MiB")
                elif rss is not None and rss > args.max_rss_mib:
                    breach = f"RSS {rss:.1f} MiB exceeded {args.max_rss_mib} MiB"
                if breach:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    break
                time.sleep(args.sample_interval)
            returncode = process.returncode
        stdout = stdout_path.read_text(encoding="utf-8", errors="replace")
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")

    benchmark_rows = []
    parse_error = None
    if not breach and returncode == 0:
        try:
            benchmark_rows = json.loads(stdout)
        except json.JSONDecodeError as error:
            parse_error = str(error)
    minimum_available = peaks["minimum_available_ram_mib"]
    if minimum_available == float("inf"):
        minimum_available = None
    return {
        "status": "FAIL_CAP" if breach else ("PASS" if returncode == 0 and not parse_error else "FAIL"),
        "abort_reason": breach,
        "parse_error": parse_error,
        "backend": args.backend,
        "model": args.model,
        "quant": args.quant,
        "model_path": str(Path(args.model_path).resolve()),
        "command": command,
        "environment": env_overrides,
        "elapsed_s": round(time.time() - started, 2),
        "caps_mib": {"process_vram": args.max_process_vram_mib,
                     "total_vram": args.max_total_vram_mib, "rss": args.max_rss_mib},
        "resources": {**peaks, "minimum_available_ram_mib": minimum_available,
                      "baseline_total_vram_mib": baseline_total},
        "benchmarks": benchmark_rows,
        "returncode": returncode,
        "stderr_tail": stderr[-4000:],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-bin", required=True,
                        help="backend-specific llama-bench executable")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model", required=True, help="stable model identifier")
    parser.add_argument("--quant", required=True, help="quant label, e.g. Q4_K_M")
    parser.add_argument("--backend", required=True, choices=["cuda", "vulkan", "rocm", "cpu"])
    parser.add_argument("--device", help="llama.cpp device, e.g. Vulkan1")
    parser.add_argument("--n-cpu-moe", type=int, default=99,
                        help="MoE layers forced to CPU; 99 means all for current catalog models")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 8) // 2))
    parser.add_argument("--prompt-tokens", type=int, default=512)
    parser.add_argument("--generation-tokens", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--extra-arg", action="append", default=[], metavar="ARG")
    parser.add_argument("--max-process-vram-mib", type=int, default=DEFAULT_PROCESS_VRAM_MIB)
    parser.add_argument("--max-total-vram-mib", type=int, default=DEFAULT_TOTAL_VRAM_MIB)
    parser.add_argument("--max-rss-mib", type=int, default=DEFAULT_RSS_MIB)
    parser.add_argument("--sample-interval", type=float, default=0.5)
    parser.add_argument("--output", help="JSON output path; default is benchmark-results/<timestamp>.json")
    args = parser.parse_args()
    if not Path(args.bench_bin).is_file():
        parser.error(f"llama-bench not found: {args.bench_bin}")
    if not Path(args.model_path).is_file():
        parser.error(f"model not found: {args.model_path}")
    if args.prompt_tokens < 0 or args.prompt_tokens > 262144:
        parser.error("--prompt-tokens must be between 0 and 262144")
    if args.generation_tokens < 0:
        parser.error("--generation-tokens cannot be negative")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    return args


def main():
    args = parse_args()
    try:
        result = execute_benchmark(args)
    except ValueError as error:
        print(f"[STOP] {error}", file=sys.stderr)
        return 2
    output = args.output
    if not output:
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        output = f"benchmark-results/{args.model}-{args.quant}-{args.backend}-{stamp}.json"
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"status={result['status']} output={output_path}")
    resources = result["resources"]
    print(f"peak_process_vram={resources['process_vram_mib']} MiB "
          f"peak_total_vram={resources['total_vram_mib']} MiB "
          f"peak_rss={resources['rss_mib']:.1f} MiB")
    for row in result["benchmarks"]:
        label = f"pp{row['n_prompt']}" if row.get("n_prompt") else f"tg{row['n_gen']}"
        print(f"{label}: {row['avg_ts']:.2f} ± {row['stddev_ts']:.2f} tok/s")
    if result["abort_reason"]:
        print(result["abort_reason"], file=sys.stderr)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
