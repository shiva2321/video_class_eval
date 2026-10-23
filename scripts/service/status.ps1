<#
Read-only check that everything the page needs is up. Needs no administrator rights.

    powershell -ExecutionPolicy Bypass -File scripts\service\status.ps1
#>
param([string]$TaskName = 'ClassEvalServer', [int]$Port = 8765)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'

function Show([string]$label, $value) { '{0,-26} {1}' -f $label, $value }

$boot = (Get-CimInstance Win32_OperatingSystem).LastBootUpTime
Show 'up since' ("$boot ({0:N1} h)" -f ((Get-Date) - $boot).TotalHours)

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($task) {
    $info = $task | Get-ScheduledTaskInfo
    Show 'scheduled task' "$($task.State), last result $($info.LastTaskResult), last run $($info.LastRunTime)"
    Show 'task triggers' (($task.Triggers | ForEach-Object { $_.CimClass.CimClassName -replace 'MSFT_Task|Trigger', '' }) -join ', ')
} else {
    Show 'scheduled task' 'NOT INSTALLED'
}

foreach ($item in @(@('page (port)', $Port), @('Ollama (port)', 11434))) {
    $listening = Get-NetTCPConnection -State Listen -LocalPort $item[1] -ErrorAction SilentlyContinue
    Show $item[0] $(if ($listening) { "$($item[1]) listening" } else { "$($item[1]) NOT listening" })
}
try {
    $code = (Invoke-WebRequest "http://127.0.0.1:$Port/healthz" -UseBasicParsing -TimeoutSec 5).StatusCode
    Show 'page /healthz' $code
} catch { Show 'page /healthz' "failed: $($_.Exception.Message)" }

if (Test-Path $tailscale) {
    $state = (& $tailscale status --json | ConvertFrom-Json).BackendState
    Show 'tailscale' $state
    Show 'funnel' ((& $tailscale funnel status | Select-Object -First 3) -join ' | ')
}

$ac = { param($sub, $name) ((powercfg /query SCHEME_CURRENT $sub $name | Select-String 'Current AC').ToString() -split ':')[-1].Trim() }
Show 'sleep after (AC)' (& $ac 'SUB_SLEEP' 'STANDBYIDLE')
Show 'hibernate after (AC)' (& $ac 'SUB_SLEEP' 'HIBERNATEIDLE')
Show '(0x0 = never)' ''

$log = Join-Path $root 'logs\supervisor.log'
if (Test-Path $log) { ''; 'last supervisor log lines:'; Get-Content $log -Tail 6 }
