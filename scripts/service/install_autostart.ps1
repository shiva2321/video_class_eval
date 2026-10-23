<#
One-time setup so the class-eval page is up whenever this computer is on, with nobody logged in:

  1. a scheduled task "ClassEvalServer" that runs supervisor.ps1 at boot (and at logon, and every 5 minutes if it ever stopped);
  2. no sleep or hibernate on mains power (a desktop is always on mains);
  3. Tailscale in unattended mode, so the shared address comes up at boot without anyone signing in.

Run it from any PowerShell; it asks for administrator rights itself (a UAC prompt) and writes logs\install.log.

    powershell -ExecutionPolicy Bypass -File scripts\service\install_autostart.ps1 [-Funnel]

-Funnel makes the supervisor keep the public Tailscale Funnel address pointing at the page.
Undo with uninstall_autostart.ps1.
#>
param(
    [switch]$Funnel,
    [string]$TaskName = 'ClassEvalServer',
    [int]$Port = 8765,
    [string]$RunAsUser = "$env:USERDOMAIN\$env:USERNAME",
    [switch]$Elevated
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$supervisor = Join-Path $PSScriptRoot 'supervisor.ps1'
$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    $argList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"", '-TaskName', $TaskName, '-Port', $Port, '-RunAsUser', "`"$RunAsUser`"", '-Elevated')
    if ($Funnel) { $argList += '-Funnel' }
    Start-Process powershell.exe -Verb RunAs -ArgumentList $argList
    Write-Host 'Approve the administrator prompt; the result is written to logs\install.log.'
    exit 0
}

New-Item -ItemType Directory -Force (Join-Path $root 'logs') | Out-Null
Start-Transcript -Path (Join-Path $root 'logs\install.log') -Force | Out-Null
try {
    Write-Host "== Scheduled task $TaskName (runs as $RunAsUser, with or without a login)"
    $taskArgs = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$supervisor`" -Port $Port"
    if ($Funnel) { $taskArgs += ' -Funnel' }
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $taskArgs -WorkingDirectory $root
    $repeat = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) -RepetitionInterval (New-TimeSpan -Minutes 5) -RepetitionDuration (New-TimeSpan -Days 3650)
    $triggers = @((New-ScheduledTaskTrigger -AtStartup), (New-ScheduledTaskTrigger -AtLogOn -User $RunAsUser), $repeat)
    $principal = New-ScheduledTaskPrincipal -UserId $RunAsUser -LogonType S4U -RunLevel Limited
    # -Priority 4 is normal CPU, disk and memory priority. The default (7) is "below normal", which starves the speech-model
    # load (about 10 MB/s instead of 1 GB/s) and leaves jobs looking stuck.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable `
        -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -Priority 4
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Principal $principal -Settings $settings `
        -Description 'Keeps the class-eval page, worker and Ollama running (scripts\service\supervisor.ps1)' -Force | Out-Null
    Write-Host 'registered'

    Write-Host '== Power (mains): never sleep, never hibernate'
    foreach ($setting in 'standby-timeout-ac', 'hibernate-timeout-ac') {
        powercfg /change $setting 0
    }
    Write-Host 'done (the screen may still turn off; the supervisor also asks Windows not to sleep while it runs)'

    Write-Host '== Tailscale unattended mode'
    if (Test-Path $tailscale) {
        & $tailscale set --unattended
        Start-Sleep -Seconds 5
        Write-Host ((& $tailscale status | Select-Object -First 2) -join "`n")
    } else {
        Write-Host 'Tailscale not installed; skipped'
    }

    Write-Host '== Starting it now (replacing a supervisor started by hand, if any)'
    Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
        Where-Object { $_.CommandLine -match 'supervisor\.ps1' -and $_.ProcessId -ne $PID } |
        ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null }
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep -Seconds 3
    Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo | Format-List LastRunTime, LastTaskResult, NextRunTime
    Write-Host 'INSTALL OK'
} catch {
    Write-Host "INSTALL FAILED: $_"
    exit 1
} finally {
    Stop-Transcript | Out-Null
    if ($Elevated) { Read-Host 'Press Enter to close' | Out-Null }
}
