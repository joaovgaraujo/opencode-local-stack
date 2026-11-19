@echo off
rem setup.cmd - one-step Windows setup (runs setup.ps1 even when PowerShell scripts are blocked).
rem Arguments go to install.py, e.g.  setup.cmd --hf https://huggingface.co/OWNER/NAME
cd /d "%~dp0"
if not "%~1"=="" set "OCL_SETUP_ARGS=%*"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
exit /b %ERRORLEVEL%
