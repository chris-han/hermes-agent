@echo off
setlocal
set "installerPowerShell=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
if exist "%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe" set "installerPowerShell=%SystemRoot%\Sysnative\WindowsPowerShell\v1.0\powershell.exe"
"%installerPowerShell%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install.ps1"
set "installerExit=%errorlevel%"
if not "%installerExit%"=="0" echo Installation failed. Read the error above before closing this window.
pause
exit /b %installerExit%
