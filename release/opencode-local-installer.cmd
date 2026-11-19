@echo off
rem opencode-local-installer.cmd - download the latest opencode-local-stack
rem release and run its setup wizard. Double-click it, or run it in a terminal.
rem Arguments are passed on to setup, e.g.:
rem   opencode-local-installer.cmd --hf https://huggingface.co/OWNER/NAME
rem The PowerShell part below the #PSSTART line is run from this same file.
setlocal
set "OCL_SELF=%~f0"
set "OCL_ARGS=%*"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s = Get-Content -Raw -LiteralPath $env:OCL_SELF; Invoke-Expression $s.Substring($s.LastIndexOf('#PS' + 'START'))"
set "RC=%ERRORLEVEL%"
pause
exit /b %RC%

#PSSTART
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # the progress bar makes downloads several times slower
$repo = "joaovgaraujo/opencode-local-stack"
$ua = @{ "User-Agent" = "opencode-local-installer" }

Write-Host "== OpenCode local stack installer ==" -ForegroundColor Cyan
Write-Host "It downloads about 20-25 GB in total (model + llama.cpp), so pick a drive with room."
$default = Join-Path $env:USERPROFILE "opencode-local-stack"
$dest = $env:OCL_DEST
if (-not $dest) {
    $dest = Read-Host "Install folder (an existing install is updated, models are kept) [$default]"
    if (-not $dest) { $dest = $default }
}
$dest = $dest.Trim('"')

if ($env:OCL_REF) {
    $label = $env:OCL_REF
    $url = "https://github.com/$repo/archive/$($env:OCL_REF).zip"
} else {
    $rel = Invoke-RestMethod "https://api.github.com/repos/$repo/releases/latest" -Headers $ua
    $label = $rel.tag_name
    $url = $rel.zipball_url
}
Write-Host "Downloading opencode-local-stack $label ..."
$tmp = Join-Path $env:TEMP ("ocl-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
$zip = Join-Path $tmp "src.zip"
Invoke-WebRequest -Uri $url -OutFile $zip -Headers $ua -UseBasicParsing
Expand-Archive -Path $zip -DestinationPath (Join-Path $tmp "x")
$src = Get-ChildItem (Join-Path $tmp "x") -Directory | Select-Object -First 1

# Copy over the release's files. Downloaded models, llama.cpp, settings.json
# and anything else not in the release stay as they are.
New-Item -ItemType Directory -Force -Path $dest | Out-Null
Copy-Item -Path (Join-Path $src.FullName "*") -Destination $dest -Recurse -Force
Remove-Item -LiteralPath $tmp -Recurse -Force
Write-Host "Files are in $dest"
Write-Host ""

$setupArgs = @()
if ($env:OCL_ARGS) { $setupArgs = $env:OCL_ARGS -split '\s+' | Where-Object { $_ } }
& (Join-Path $dest "setup.cmd") @setupArgs
exit $LASTEXITCODE
