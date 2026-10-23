<#
Removes the "ClassEvalServer" scheduled task and stops the page, worker and supervisor it started.
Power settings and Tailscale unattended mode are left as they are.

    powershell -ExecutionPolicy Bypass -File scripts\service\uninstall_autostart.ps1
#>
param([string]$TaskName = 'ClassEvalServer')

$ErrorActionPreference = 'Stop'
$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) { Write-Host "No task named $TaskName."; exit 0 }
Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'scripts[\\/]run\.py|classeval\.server\.main|classeval\.worker' } |
    ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null }
Write-Host "Removed $TaskName and stopped the page and worker."
