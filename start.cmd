@echo off
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0cone_project\start_tool.ps1"
if errorlevel 1 (
  pause
) else (
  start "" "http://127.0.0.1:8765/"
)
