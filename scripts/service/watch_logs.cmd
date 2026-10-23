@echo off
rem Opens the live server log window (read-only; closing it does not affect the server).
start "Class Assessor - live server log" powershell.exe -NoExit -NoProfile -ExecutionPolicy Bypass -File "%~dp0watch_logs.ps1" %*
