@echo off
rem setup.cmd - one-step Windows setup (runs setup.ps1 even when PowerShell scripts are blocked).
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
exit /b %ERRORLEVEL%
