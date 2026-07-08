# run.ps1 — Qwen3.6-35B-A3B (Q4_K_M) for 8GB-class NVIDIA GPUs (8 GB-class GPU target)
# Strategy: all MoE experts in system RAM (--cpu-moe), attention + KV cache on GPU.
# Portable: paths below are relative to THIS script's folder. To relocate, either keep the
# layout (llama.cpp\  models\  next to this script) or override the two params.
param(
    # Folder containing llama-server.exe (+ CUDA DLLs)
    [string]$BinDir    = "$PSScriptRoot\llama.cpp",
    # Full path to the GGUF
    [string]$ModelPath = "$PSScriptRoot\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    # KV offload: "" = --cpu-moe (all experts on CPU, safest/lowest VRAM).
    # On the real 8 GB target, binary-search this upward: -NCpuMoe 28, 32, 36... until VRAM ~6.5GB.
    [int]$NCpuMoe      = -1,
    [int]$Port         = 8080,
    [int]$CtxSize      = 65536,
    [string]$KvType    = "q8_0"       # drop to q4_0 only if you must fit more ctx in 8GB
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $ModelPath)) { throw "Model not found: $ModelPath" }
if (-not (Test-Path "$BinDir\llama-server.exe")) { throw "llama-server.exe not found in: $BinDir" }

# --cpu-moe (all experts on CPU) vs --n-cpu-moe N (first N layers' experts on CPU, rest on GPU)
$moeArgs = if ($NCpuMoe -ge 0) { @("--n-cpu-moe", "$NCpuMoe") } else { @("--cpu-moe") }

$serverArgs = @(
    "-m", $ModelPath,
    "--alias", "qwen3.6-35b-a3b",
    "--n-gpu-layers", "999",          # all layers to GPU; --cpu-moe overrides expert tensors to CPU
    "--ctx-size", "$CtxSize",
    "--flash-attn", "on",
    "-ctk", $KvType,
    "-ctv", $KvType,
    "--jinja",
    "--host", "127.0.0.1",
    "--port", "$Port"
) + $moeArgs

Write-Host "==> llama-server $($serverArgs -join ' ')"
& "$BinDir\llama-server.exe" @serverArgs
