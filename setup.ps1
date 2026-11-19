# setup.ps1 - one-step Windows setup. Run setup.cmd (double-click it, or type
# .\setup.cmd in PowerShell); it starts this script even when Windows blocks
# PowerShell scripts.
#
#   .\setup.cmd                                   the wizard: model, context, thinking
#   .\setup.cmd --hf https://huggingface.co/OWNER/NAME
#   .\setup.cmd --model qwen3.6-35b-a3b --think off --non-interactive
#
# Installs Python 3.12 with winget if no Python 3.9+ is found, then runs
# install.py, which installs Node + OpenCode, llama.cpp and the model.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Update-PathFromRegistry {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user"
}

function Find-Python {
    # The Microsoft Store "python" stub exits with an error instead of running,
    # so every candidate is checked by actually running it.
    foreach ($cand in @(@("py", "-3"), @("python"), @("python3"))) {
        try {
            $exe = $cand[0]
            $pre = @($cand | Select-Object -Skip 1)
            $out = & $exe @pre -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($LASTEXITCODE -eq 0 -and $out -and ([version]$out -ge [version]"3.9")) {
                return , $cand
            }
        } catch { }
    }
    return $null
}

Write-Host "== OpenCode local stack setup ==" -ForegroundColor Cyan
$py = Find-Python
if (-not $py) {
    Write-Host "Python 3.9+ not found. Installing Python 3.12 with winget ..."
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "winget is not available. Install Python from https://www.python.org/downloads/"
        Write-Host "(tick 'Add python.exe to PATH'), then run setup.cmd again."
        exit 1
    }
    winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
    Update-PathFromRegistry
    $py = Find-Python
    if (-not $py) {
        Write-Host "Python was installed but this window can't see it yet. Close it, open a new"
        Write-Host "PowerShell window in this folder and run .\setup.cmd again."
        exit 1
    }
}
$exe = $py[0]
$pre = @($py | Select-Object -Skip 1)
# setup.cmd hands its arguments over in OCL_SETUP_ARGS: on the command line,
# Windows PowerShell would take "--hf" etc. as its own parameter names.
$passArgs = @()
if ($env:OCL_SETUP_ARGS) {
    $passArgs = @([regex]::Matches($env:OCL_SETUP_ARGS, '"[^"]*"|\S+') |
                  ForEach-Object { $_.Value.Trim('"') })
}
& $exe @pre (Join-Path $PSScriptRoot "install.py") --install-node @passArgs @args
$rc = $LASTEXITCODE

# Double-clicked setup.cmd (Explorer -> cmd -> this script): keep the window
# open so the result can be read. Started from a terminal: just return.
try {
    $me = Get-CimInstance Win32_Process -Filter "ProcessId = $PID"
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($me.ParentProcessId)"
    $grand = Get-CimInstance Win32_Process -Filter "ProcessId = $($parent.ParentProcessId)"
    if ($grand.Name -eq "explorer.exe") { Read-Host "Press Enter to close" | Out-Null }
} catch { }
exit $rc
