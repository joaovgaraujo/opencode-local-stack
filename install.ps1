<#
  install.ps1 - one-shot installer + validator for the Qwen3.6-35B-A3B local stack.
  Target: 8 GB-class laptop GPU (8 GB), sufficient RAM, Windows. Frontend: OpenCode only.

  Run from inside the repo folder:
      powershell -ExecutionPolicy Bypass -File .\install.ps1

  What it does (idempotent - safe to re-run):
    1. Pre-flight: NVIDIA GPU, VRAM, compute cap, free RAM, free disk.
    2. Locates the llama.cpp CUDA binary (bundled .\llama.cpp, else downloads the prebuilt release).
    3. Locates the GGUF (.\models, -ModelPath, else downloads ~20GB from Hugging Face, resumable+verified).
    4. Starts llama-server with the chosen profile; waits for /health; checks the /v1/models alias.
    5. Runs the validation suite (short/code/tool/30k-needle) and logs peak VRAM + RAM.
    6. Installs OpenCode (npm) and writes opencode.json; runs a headless agentic smoke test.
    7. Writes RESULTS.md and prints a pass/fail summary.

  System packages are NOT installed silently. If Node (required) or Python (only for the
  test suite) is missing, it prints the winget command and stops - unless you pass
  -InstallNode / -InstallPython to opt in.
#>
[CmdletBinding()]
param(
    [ValidateSet("primary","conservative")]
    [string]$Profile   = "primary",           # primary=65536 ctx, conservative=32768 ctx
    [string]$ModelPath = "",                   # use an existing GGUF instead of downloading
    [string]$BinDir    = "",                   # use an existing llama.cpp folder instead of downloading
    [int]$Port         = 8080,
    [switch]$InstallNode,                      # opt in to 'winget install OpenJS.NodeJS.LTS'
    [switch]$InstallPython,                    # opt in to 'winget install Python.Python.3.12'
    [switch]$SkipTests,                        # skip validation + opencode smoke test
    [switch]$StopWhenDone                      # stop the server after tests (default: leave running)
)
# NOTE: 'Continue', not 'Stop'. This script drives many native commands (llama-server,
# curl, npm, python) that legitimately write to stderr; under 'Stop' those stderr writes
# get promoted to terminating errors. Correctness is enforced by the explicit Die() guards
# and $LASTEXITCODE / Test-Path checks below instead.
$ErrorActionPreference = "Continue"
$Root      = $PSScriptRoot
if (-not $BinDir) { $BinDir = Join-Path $Root "llama.cpp" }
$ModelDir  = Join-Path $Root "models"
$GgufName  = "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
$HfUrl     = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/$GgufName"
$HfApi     = "https://huggingface.co/api/models/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main?recursive=true"
$Alias     = "qwen3.6-35b-a3b"
$CtxSize   = if ($Profile -eq "conservative") { 32768 } else { 65536 }
$VramBudgetMiB = 6656                          # 6.5 GB target for the 8 GB target

function Say  ($m) { Write-Host "==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "  [OK]   $m" -ForegroundColor Green }
function Warn ($m) { Write-Host "  [WARN] $m" -ForegroundColor Yellow }
function Die  ($m) { Write-Host "  [STOP] $m" -ForegroundColor Red; exit 1 }
$script:Report = [ordered]@{}

# ------------------------------------------------- 0. Self-contained assets
# So a LONE install.ps1 can bootstrap everything: if opencode.json / validate.py
# are not sitting next to this script, write them from the embedded copies below.
# opencode.json is (re)generated to match the chosen -Port / -Profile every run.
Say "Materializing bundled assets (config + tests) if missing"
$OpencodeJson = Join-Path $Root "opencode.json"
$ocCfg = @"
{
  "`$schema": "https://opencode.ai/config.json",
  "model": "llamacpp/$Alias",
  "provider": {
    "llamacpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": { "baseURL": "http://127.0.0.1:$Port/v1" },
      "models": {
        "$Alias": {
          "name": "Qwen3.6-35B-A3B (local)",
          "tools": true,
          "reasoning": true,
          "limit": { "context": $CtxSize, "output": 32768 }
        }
      }
    }
  }
}
"@
Set-Content -Path $OpencodeJson -Value $ocCfg -Encoding UTF8
Ok "opencode.json (baseURL port $Port, context $CtxSize)"

$ValidatePy = Join-Path $Root "tests\validate.py"
if (-not (Test-Path $ValidatePy)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $Root "tests") | Out-Null
    $vpy = @'
#!/usr/bin/env python3
"""Validation for the local Qwen3.6-35B-A3B llama-server (OpenAI endpoint).
Runs: short completion, code-gen+ast.parse, tool-calling, long-context needle.
No third-party deps (urllib only). Prints a pass/fail table + tok/s."""
import ast, json, sys, time, urllib.request

BASE = "http://127.0.0.1:8080/v1"
MODEL = "qwen3.6-35b-a3b"
results = []

def chat(messages, tools=None, tool_choice=None, max_tokens=2048, temperature=0.2):
    # Qwen3.6 is a reasoning model: it emits <think> tokens (in reasoning_content) BEFORE
    # the final content. max_tokens must cover BOTH or content comes back empty.
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    if tools: body["tools"] = tools
    if tool_choice: body["tool_choice"] = tool_choice
    data = json.dumps(body).encode()
    req = urllib.request.Request(BASE + "/chat/completions", data=data,
                                 headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer local"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1200) as r:
        j = json.load(r)
    dt = time.time() - t0
    usage = j.get("usage", {}); tim = j.get("timings", {})
    ctoks = usage.get("completion_tokens", 0); ptoks = usage.get("prompt_tokens", 0)
    tps = tim.get("predicted_per_second") or (ctoks / dt if dt else 0)
    return j, {"dt": dt, "ctoks": ctoks, "ptoks": ptoks, "tps": tps,
               "prompt_tps": tim.get("prompt_per_second")}

def record(name, passed, detail, meta=None):
    tps = round(meta["tps"], 1) if meta else None
    results.append((name, passed, detail, tps))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {detail}"
          + (f"  ({tps} tok/s gen)" if tps else ""))

try:
    j, m = chat([{"role": "user", "content": "Answer in one short sentence: why is the sky blue?"}],
                max_tokens=1024)
    txt = j["choices"][0]["message"]["content"].strip()
    ok = len(txt) > 15 and any(w in txt.lower() for w in ["scatter", "light", "blue", "wavelength"])
    record("short-completion", ok, repr(txt[:160]), m)
except Exception as e:
    record("short-completion", False, f"ERROR {e}")

try:
    j, m = chat([{"role": "user", "content": "Write a Python function fib(n) that returns the nth "
                  "Fibonacci number. Include a docstring. Reply with only a python code block."}],
                max_tokens=2048)
    txt = j["choices"][0]["message"]["content"]
    fence = chr(96)*3
    code = txt.split(fence+"python")[1].split(fence)[0] if fence+"python" in txt else \
           (txt.split(fence)[1].split(fence)[0] if fence in txt else txt)
    tree = ast.parse(code)
    fns = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    has_doc = any(ast.get_docstring(f) for f in fns)
    record("code-gen", len(fns) >= 1 and has_doc, f"parsed OK, {len(fns)} func(s), docstring={has_doc}", m)
except Exception as e:
    record("code-gen", False, f"ERROR {e}")

try:
    tools = [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a location",
        "parameters": {"type": "object", "properties": {
            "location": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["location"]}}}]
    j, m = chat([{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}],
                tools=tools, tool_choice="auto", max_tokens=200)
    msg = j["choices"][0]["message"]; tcs = msg.get("tool_calls") or []
    ok = False; detail = "no tool_calls emitted"
    if tcs:
        tc = tcs[0]["function"]; args = json.loads(tc["arguments"])
        ok = tc["name"] == "get_weather" and "location" in args and "paris" in str(args["location"]).lower()
        detail = f"name={tc['name']} args={args}"
    record("tool-calling", ok, detail, m)
except Exception as e:
    record("tool-calling", False, f"ERROR {e}")

try:
    needle = "REMEMBER THIS: the secret vault access code is PURPLE-WOMBAT-4291."
    filler = ("The quick brown fox jumps over the lazy dog while the diligent engineer "
              "reviews configuration files and verifies cache behavior. ")
    lines, target_chars, i, cur = [], 145000, 0, 0
    while cur < target_chars:
        s = f"[line {i:05d}] {filler}"; lines.append(s); cur += len(s) + 1; i += 1
    lines.insert(int(len(lines) * 0.60), needle)
    document = chr(10).join(lines)
    prompt = ("Below is a long document. Somewhere inside it is a secret vault access code. "
              "Read carefully and tell me ONLY the exact code (format WORD-WORD-NNNN).\n\n"
              f"=== DOCUMENT START ===\n{document}\n=== DOCUMENT END ===\n\n"
              "What is the secret vault access code?")
    j, m = chat([{"role": "user", "content": prompt}], max_tokens=2048, temperature=0.0)
    txt = j["choices"][0]["message"]["content"]
    ok = "PURPLE-WOMBAT-4291" in txt.upper()
    record("long-context-needle", ok,
           f"prompt_tokens={m['ptoks']} needle@60pct found={ok} ans={txt.strip()[:80]!r}", m)
    print(f"##NEEDLE_META## prompt_tokens={m['ptoks']} gen_tps={m['tps']:.2f} "
          f"prompt_tps={m['prompt_tps']} dt={m['dt']:.1f}")
except Exception as e:
    record("long-context-needle", False, f"ERROR {e}")

print("\n=== PASS/FAIL TABLE ===")
allpass = True
for name, passed, detail, tps in results:
    allpass = allpass and passed
    print(f"  {'PASS' if passed else 'FAIL':4}  {name:22} {('%.1f tok/s'%tps) if tps else '':>12}")
sys.exit(0 if allpass else 1)
'@
    Set-Content -Path $ValidatePy -Value $vpy -Encoding UTF8
    Ok "tests\validate.py (generated)"
} else { Ok "tests\validate.py (present)" }

# ---------------------------------------------------------------- 1. Pre-flight
Say "Pre-flight checks"
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { Die "nvidia-smi not found. Install the NVIDIA driver." }
$gpu = (nvidia-smi --query-gpu=name,memory.total,compute_cap,driver_version --format=csv,noheader) -join ""
Ok "GPU: $gpu"
$script:Report["GPU"] = $gpu
$vramMiB = [int]((nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits) -join "")
if ($vramMiB -lt 7500) { Warn "GPU has less than 8GB VRAM ($vramMiB MiB) - the 65536 profile may not fit; consider -Profile conservative." }
$cc = ((nvidia-smi --query-gpu=compute_cap --format=csv,noheader) -join "").Trim()
if ($cc -ne $ExpectedCC) { Warn "Compute cap is $cc, not the expected value. The bundled binary still runs (fat build), but this is not the 8 GB target." }

$os = Get-CimInstance Win32_OperatingSystem
$ramFreeGB = [math]::Round($os.FreePhysicalMemory/1MB,1)
$ramTotGB  = [math]::Round($os.TotalVisibleMemorySize/1MB,1)
Ok "RAM: $ramFreeGB GB free / $ramTotGB GB total"
if ($ramFreeGB -lt 24) { Warn "Free RAM under 24GB. --cpu-moe needs ~20GB for experts; close apps or expect paging." }
$script:Report["RAM"] = "$ramFreeGB GB free / $ramTotGB GB total"

$driveLetter = (Split-Path $Root -Qualifier).TrimEnd(':')
$freeDiskGB = [math]::Round((Get-PSDrive $driveLetter).Free/1GB,1)
Ok "Free disk on ${driveLetter}: $freeDiskGB GB"
if ($freeDiskGB -lt 25 -and -not (Test-Path (Join-Path $ModelDir $GgufName))) { Die "Need ~25GB free for the model, have $freeDiskGB GB." }

# ---------------------------------------------------------------- 2. Binary
Say "Locating llama.cpp CUDA binary"
if (-not (Test-Path (Join-Path $BinDir "llama-server.exe"))) {
    Warn "No bundled binary in .\llama.cpp - downloading the latest prebuilt CUDA release."
    New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
    $rel = Invoke-RestMethod "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
    $binAsset  = $rel.assets | Where-Object { $_.name -match "^llama-.*bin-win-cuda-12.*x64\.zip$" } | Select-Object -First 1
    $cudaAsset = $rel.assets | Where-Object { $_.name -match "^cudart-.*win-cuda-12.*x64\.zip$" } | Select-Object -First 1
    if (-not $binAsset) { Die "No CUDA 12 win asset in latest release; check github.com/ggml-org/llama.cpp/releases" }
    curl.exe -L --fail -o "$Root\_bin.zip"    $binAsset.browser_download_url
    curl.exe -L --fail -o "$Root\_cudart.zip" $cudaAsset.browser_download_url
    Expand-Archive "$Root\_bin.zip"    -DestinationPath $BinDir -Force
    Expand-Archive "$Root\_cudart.zip" -DestinationPath $BinDir -Force
    Remove-Item "$Root\_bin.zip","$Root\_cudart.zip" -Force
}
$Server = Join-Path $BinDir "llama-server.exe"
if (-not (Test-Path $Server)) { Die "llama-server.exe missing after setup." }
$ver = (& $Server --version 2>&1 | Out-String).Trim()
Ok ("Binary: " + (($ver -split "`n") | Select-Object -First 1))
if (-not ((& $Server --list-devices 2>&1 | Out-String) -match "CUDA0")) { Die "Binary does not see a CUDA device." }
Ok "CUDA device visible to llama-server"
$script:Report["Binary"] = (($ver -split "`n") | Select-Object -First 1)

# ---------------------------------------------------------------- 3. Model
Say "Locating model GGUF"
if (-not $ModelPath) { $ModelPath = Join-Path $ModelDir $GgufName }
if (-not (Test-Path $ModelPath)) {
    New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
    Warn "GGUF not found at $ModelPath - downloading ~20GB (resumable)."
    curl.exe -L --fail -C - -o $ModelPath $HfUrl
    if ($LASTEXITCODE -ne 0) { Die "Model download failed (curl exit $LASTEXITCODE). Re-run to resume." }
}
$fsr = [IO.File]::OpenRead($ModelPath); $mb = New-Object byte[] 4; $null = $fsr.Read($mb,0,4); $fsr.Close()
if ([Text.Encoding]::ASCII.GetString($mb) -ne "GGUF") { Die "File at $ModelPath is not a valid GGUF (bad magic)." }
$actual = (Get-Item $ModelPath).Length
try {
    $expected = (Invoke-RestMethod $HfApi) | Where-Object { $_.path -eq $GgufName } | Select-Object -First 1 -ExpandProperty size
    if ($expected -and $actual -ne $expected) { Die "GGUF size mismatch (have $actual, expect $expected). Re-run to resume (curl -C -)." }
    Ok "Model verified: $ModelPath ($([math]::Round($actual/1GB,1)) GB, GGUF magic + size match)"
} catch {
    Ok "Model present: $ModelPath ($([math]::Round($actual/1GB,1)) GB, GGUF magic OK; size check skipped - no internet)"
}
$script:Report["Model"] = $ModelPath

# ---------------------------------------------------------------- 4. Start server
Say "Starting llama-server ($Profile profile: ctx $CtxSize, q8_0 KV, --cpu-moe, port $Port)"
Get-Process llama-server -ErrorAction SilentlyContinue | ForEach-Object { Warn "killing existing llama-server PID $($_.Id)"; Stop-Process -Id $_.Id -Force }
Start-Sleep 2
$baselineVram = [int]((nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join "")
$srvArgs = @("-m",$ModelPath,"--alias",$Alias,"--n-gpu-layers","999","--cpu-moe",
             "--ctx-size","$CtxSize","--flash-attn","on","-ctk","q8_0","-ctv","q8_0",
             "--jinja","--host","127.0.0.1","--port","$Port")
$logOut = Join-Path $Root "server.log"; $logErr = Join-Path $Root "server.err"
Start-Process $Server -ArgumentList $srvArgs -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Hidden
Say "Waiting for /health (loading ~20GB experts into RAM)..."
$up = $false
for ($i=0; $i -lt 180; $i++) {
    try { if ((curl.exe -s "http://127.0.0.1:$Port/health") -match "ok") { $up = $true; break } } catch {}
    Start-Sleep 2
}
if (-not $up) { Warn "server.err tail:"; Get-Content $logErr -Tail 20 -ErrorAction SilentlyContinue; Die "Server did not become healthy. See server.err / server.log." }
Ok "Server healthy on port $Port"
$served = (curl.exe -s "http://127.0.0.1:$Port/v1/models" | ConvertFrom-Json).data[0].id
if ($served -ne $Alias) { Die "/v1/models reports '$served', expected '$Alias'." }
Ok "/v1/models alias: $served"

# ---------------------------------------------------------------- 5. Validation + VRAM
$results = [ordered]@{}
if (-not $SkipTests) {
    $py = Get-Command python -ErrorAction SilentlyContinue
    if (-not $py) {
        if ($InstallPython) { Say "Installing Python (winget)"; winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements; $py = Get-Command python -ErrorAction SilentlyContinue }
        else { Warn "Python not found - skipping the test suite. Install:  winget install Python.Python.3.12   (or re-run with -InstallPython)." }
    }
    if ($py) {
        Say "Running validation suite + sampling VRAM/RAM"
        $vlog = Join-Path $Root "tests\vram.log"
        $sampler = Start-Job -ScriptBlock {
            param($out)
            "ts,card_mib,ws_mib" | Set-Content $out
            for ($k=0; $k -lt 900; $k++) {
                $c = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) -join ""
                $p = Get-Process llama-server -ErrorAction SilentlyContinue
                $w = if ($p) { [int]($p.WorkingSet64/1MB) } else { 0 }
                "$k,$c,$w" | Add-Content $out; Start-Sleep 1
            }
        } -ArgumentList $vlog
        Push-Location (Join-Path $Root "tests")
        $testOut = & python validate.py 2>&1 | Out-String
        Pop-Location
        Stop-Job $sampler -ErrorAction SilentlyContinue; Remove-Job $sampler -Force -ErrorAction SilentlyContinue
        Write-Host $testOut
        foreach ($ln in ($testOut -split "`n")) {
            if ($ln -match '^\[(PASS|FAIL)\]\s+([a-z0-9\-]+):') { $results[$Matches[2]] = $Matches[1] }
        }
        if (Test-Path $vlog) {
            $rows = Import-Csv $vlog
            $peakCard = [int](($rows | Measure-Object card_mib -Maximum).Maximum)
            $peakWs   = [int](($rows | Measure-Object ws_mib   -Maximum).Maximum)
            $procVram = $peakCard - $baselineVram
            $fit = if ($procVram -le $VramBudgetMiB) { "PASS" } else { "FAIL (over by $($procVram - $VramBudgetMiB) MiB)" }
            $results["vram-fit"] = $fit
            $script:Report["Peak VRAM (llama-server)"] = "$procVram MiB (card peak $peakCard - baseline $baselineVram); budget $VramBudgetMiB MiB -> $fit"
            $script:Report["Peak RAM (working set)"]   = "$peakWs MiB"
            Say "Peak VRAM (llama-server): ~$procVram MiB   Peak RAM: ~$peakWs MiB   Fit(<=6.5GB): $fit"
        }

        # ------- OpenCode setup + agentic smoke test -------
        Say "Setting up OpenCode"
        $node = Get-Command node -ErrorAction SilentlyContinue
        if (-not $node) {
            if ($InstallNode) { Say "Installing Node LTS (winget)"; winget install -e --id OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements; $node = Get-Command node -ErrorAction SilentlyContinue }
            else { Warn "Node not found - skipping OpenCode. Install:  winget install OpenJS.NodeJS.LTS   (then re-run, or pass -InstallNode)." }
        }
        if ($node) {
            npm install -g opencode-ai@latest @ai-sdk/openai-compatible 2>&1 | Select-Object -Last 3 | Out-Host
            $cfgDir = Join-Path $env:USERPROFILE ".config\opencode"
            New-Item -ItemType Directory -Force -Path $cfgDir | Out-Null
            Copy-Item (Join-Path $Root "opencode.json") (Join-Path $cfgDir "opencode.json") -Force
            Ok "opencode.json -> $cfgDir"
            $sc = Join-Path $Root "opencode-scratch"
            New-Item -ItemType Directory -Force -Path $sc | Out-Null
            Get-ChildItem $sc -ErrorAction SilentlyContinue | Remove-Item -Force -Recurse
            Say "OpenCode warm-up (first run downloads ripgrep once - needs internet)"
            $warm = Start-Job { & opencode run --auto --dir $using:sc -m "llamacpp/qwen3.6-35b-a3b" "reply with: ready" 2>&1 }
            if (-not (Wait-Job $warm -Timeout 180)) { Warn "warm-up slow (ripgrep download?) - continuing"; Stop-Job $warm }
            Receive-Job $warm | Out-Null; Remove-Job $warm -Force -ErrorAction SilentlyContinue
            Get-Process opencode -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
            Say "OpenCode agentic smoke test (create + run a python file)"
            $agent = Start-Job { & opencode run --auto --dir $using:sc -m "llamacpp/qwen3.6-35b-a3b" "Create a Python file calc.py that prints the sum of 2 and 3, then run it with 'python calc.py' and report the exact stdout." 2>&1 }
            $ocPass = "FAIL"
            if (Wait-Job $agent -Timeout 360) {
                Receive-Job $agent | Out-Null
                Start-Sleep 1
                $calc = Join-Path $sc "calc.py"
                if ((Test-Path $calc) -and ((python $calc 2>&1) -match "^5")) { $ocPass = "PASS" }
            } else { Warn "opencode agentic test timed out"; Stop-Job $agent }
            Remove-Job $agent -Force -ErrorAction SilentlyContinue
            Get-Process opencode -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
            $results["opencode-agentic"] = $ocPass
            if ($ocPass -eq "PASS") { Ok "OpenCode wrote+ran calc.py -> 5" } else { Warn "OpenCode agentic test not confirmed (see opencode-scratch). If garbage/compaction: ctx too small or template rejecting non-first system msg." }
        }
    }
}

# ---------------------------------------------------------------- 6. Report
Say "Writing RESULTS.md"
$md = @()
$md += "# RESULTS - install.ps1 run on THIS laptop"
$md += ""
$md += "Profile: **$Profile** (ctx $CtxSize, q8_0 KV, --cpu-moe), port $Port"
$md += ""
$md += "## Environment"
foreach ($k in $script:Report.Keys) { $md += "- **$k**: $($script:Report[$k])" }
$md += ""
$md += "## Pass / fail"
$md += "| Test | Result |"
$md += "|---|---|"
foreach ($k in $results.Keys) { $md += "| $k | $($results[$k]) |" }
$md += ""
$md += "_tok/s: see the validation output above; re-measure here - it does not transfer from the build rig._"
$md | Set-Content (Join-Path $Root "RESULTS.md")

Write-Host ""
Say "SUMMARY"
foreach ($k in $results.Keys) {
    $c = if ($results[$k] -eq "PASS") { "Green" } else { "Red" }
    Write-Host ("  {0,-20} {1}" -f $k, $results[$k]) -ForegroundColor $c
}
$notPass = ($results.Values | Where-Object { $_ -ne "PASS" }).Count
Write-Host ""
if ($results.Count -eq 0) { Warn "Server is up; tests were skipped." }
elseif ($notPass -eq 0) { Ok "ALL CHECKS PASSED. Endpoint: http://127.0.0.1:$Port/v1  (alias $Alias)" }
else { Warn "Some checks did not pass - see the table above and RESULTS.md." }

if ($StopWhenDone) { Say "Stopping server (-StopWhenDone)"; Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force }
else { Write-Host ""; Say "Server left running (PID $((Get-Process llama-server -ErrorAction SilentlyContinue).Id)). Stop it with:  Get-Process llama-server | Stop-Process -Force" }
Write-Host ""
Write-Host "Next: run 'opencode' in any project folder, then /models to pick Qwen3.6-35B-A3B (local)." -ForegroundColor Cyan
