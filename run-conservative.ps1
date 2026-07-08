# run-conservative.ps1 — LOW-MEMORY profile for 8 GB-class GPU 8GB
# Same model/binary as run.ps1, but tuned for maximum safety margin:
#   - ctx-size 32768 (half) -> ~half the KV-cache VRAM *and* less RAM pressure
#   - --cpu-moe (ALL experts in RAM, nothing extra on GPU)
#   - q8_0 KV kept for quality; flip -KvType q4_0 to shave KV VRAM further
# Use this if the full 65536/q8_0 profile ever bumps the 8GB card, or on a machine
# with a busy desktop/other GPU apps eating into the 8GB. Trade-off: 32k context window.
param(
    [string]$BinDir    = "$PSScriptRoot\llama.cpp",
    [string]$ModelPath = "$PSScriptRoot\models\Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    [int]$Port         = 8080,
    [int]$CtxSize      = 32768,       # conservative: half of the 65536 target
    [string]$KvType    = "q8_0"       # set to "q4_0" for an even smaller KV cache
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path $ModelPath)) { throw "Model not found: $ModelPath" }
if (-not (Test-Path "$BinDir\llama-server.exe")) { throw "llama-server.exe not found in: $BinDir" }

$serverArgs = @(
    "-m", $ModelPath,
    "--alias", "qwen3.6-35b-a3b",
    "--n-gpu-layers", "999",
    "--cpu-moe",                      # all MoE experts on CPU/RAM (lowest VRAM)
    "--ctx-size", "$CtxSize",
    "--flash-attn", "on",
    "-ctk", $KvType,
    "-ctv", $KvType,
    "--jinja",
    "--host", "127.0.0.1",
    "--port", "$Port"
)
Write-Host "==> [CONSERVATIVE] llama-server $($serverArgs -join ' ')"
& "$BinDir\llama-server.exe" @serverArgs
