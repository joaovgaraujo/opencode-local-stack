# Build-Installer.ps1
# Compiles the Inno Setup installer to dist\OpenCodeLocal-Setup.exe.
#   .\Build-Installer.ps1 -Version 1.1.0
# The version defaults to the one in packaging\OpenCodeLocal.iss.

[CmdletBinding()]
param(
    [string]$Version
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot

# Byte-compile check: a syntax error would otherwise only show up on the
# user's PC, after the installer has run.
$py = (Get-Command python -ErrorAction SilentlyContinue).Source
if ($py) {
    & $py -m py_compile (Join-Path $root 'install.py') (Join-Path $root 'start.py') `
        (Get-ChildItem (Join-Path $root 'installer') -Filter *.py | ForEach-Object FullName)
    if ($LASTEXITCODE -ne 0) { throw 'py_compile failed' }
}

# Locate ISCC (Inno Setup 6): PATH, per-user install, then machine-wide
$iscc = (Get-Command iscc -ErrorAction SilentlyContinue).Source
if (-not $iscc) {
    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe')
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    $iscc = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
}
if (-not $iscc) { throw 'ISCC.exe (Inno Setup 6) not found. Install with: winget install JRSoftware.InnoSetup' }

New-Item -ItemType Directory -Path (Join-Path $root 'dist') -Force | Out-Null

$isccArgs = @()
if ($Version) { $isccArgs += "/DMyAppVersion=$($Version.TrimStart('v'))" }
$isccArgs += (Join-Path $root 'packaging\OpenCodeLocal.iss')

Write-Host "[build] Compiling installer with $iscc..."
& $iscc @isccArgs
if ($LASTEXITCODE -ne 0) { throw "ISCC failed with exit code $LASTEXITCODE" }

$setup = Join-Path $root 'dist\OpenCodeLocal-Setup.exe'
$size = '{0:N1} MB' -f ((Get-Item $setup).Length / 1MB)
Write-Host "[build] Installer: $setup ($size)"
