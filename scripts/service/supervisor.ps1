<#
Keeps the class-eval page, its worker and Ollama running, and the computer awake while it does.
Started at boot by the "ClassEvalServer" scheduled task (see install_autostart.ps1); it can also be run by hand:

    powershell -ExecutionPolicy Bypass -File scripts\service\supervisor.ps1 [-Port 8765] [-Funnel]

-Funnel also keeps "tailscale funnel" pointing at the page (the public address). Output goes to logs\supervisor.log.
#>
param(
    [int]$Port = 8765,
    [switch]$Funnel
)

$ErrorActionPreference = 'Continue'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$logDir = Join-Path $root 'logs'
New-Item -ItemType Directory -Force $logDir | Out-Null
$logFile = Join-Path $logDir 'supervisor.log'
$python = Join-Path $root '.venv\Scripts\python.exe'
$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
$ollamaPort = 11434


function Write-Log([string]$message) {
    try {
        if ((Test-Path $logFile) -and (Get-Item $logFile).Length -gt 5MB) {
            Move-Item $logFile "$logFile.old" -Force
        }
        [IO.File]::AppendAllText($logFile, "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $message`r`n")
    } catch { }
}

function Test-Port([int]$port) {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect('127.0.0.1', $port, $null, $null)
        return ($pending.AsyncWaitHandle.WaitOne(1500) -and $client.Connected)
    } catch { return $false } finally { $client.Close() }
}

# Runs a program without a window, returns its output; gives up (and kills it) after $timeoutSec.
function Invoke-Tool([string]$exe, [string[]]$arguments, [int]$timeoutSec = 20) {
    $out = [IO.Path]::GetTempFileName()
    $err = [IO.Path]::GetTempFileName()
    try {
        $proc = Start-Process $exe -ArgumentList $arguments -WindowStyle Hidden -PassThru -RedirectStandardOutput $out -RedirectStandardError $err
        if (-not $proc.WaitForExit($timeoutSec * 1000)) {
            taskkill /PID $proc.Id /T /F | Out-Null
            return 'timed out'
        }
        return ((Get-Content $out, $err -Raw -ErrorAction SilentlyContinue) -join "`n")
    } finally { Remove-Item $out, $err -Force -ErrorAction SilentlyContinue }
}

# Stops the page and the worker, including copies left behind by an earlier run.
function Stop-Stack {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match 'scripts[\\/]run\.py|classeval\.server\.main|classeval\.worker' } |
        ForEach-Object { taskkill /PID $_.ProcessId /T /F | Out-Null }
}

function Start-Stack {
    Stop-Stack
    $script:stack = Start-Process $python -ArgumentList 'scripts\run.py', '--port', $Port -WorkingDirectory $root `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'run.log') -RedirectStandardError (Join-Path $logDir 'run.err.log')
    $script:graceUntil = (Get-Date).AddSeconds(120)
    $script:portDownChecks = 0
    Write-Log "started the page and worker (pid $($script:stack.Id)) on port $Port"
}

function Start-Ollama {
    $exe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
    if (-not $exe) { $exe = Join-Path $env:LOCALAPPDATA 'Programs\Ollama\ollama.exe' }
    if (-not (Test-Path $exe)) { Write-Log "Ollama not found at $exe"; return }
    Start-Process $exe -ArgumentList 'serve' -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir 'ollama.log') -RedirectStandardError (Join-Path $logDir 'ollama.err.log')
    Write-Log 'started Ollama'
}

function Sync-Funnel {
    if (-not (Test-Path $tailscale)) { return }
    $state = (Invoke-Tool $tailscale @('status', '--json') 10) | ConvertFrom-Json -ErrorAction SilentlyContinue
    if (-not $state -or $state.BackendState -ne 'Running') { return }
    if ((Invoke-Tool $tailscale @('funnel', 'status') 10) -match "127\.0\.0\.1:$Port") { return }
    $result = (Invoke-Tool $tailscale @('funnel', '--bg', "$Port") 30).Trim()
    Write-Log "tailscale funnel --bg $Port -> $result"
}


# One supervisor at a time: the lock file is held for as long as this process lives.
try {
    $lock = [IO.File]::Open((Join-Path $logDir 'supervisor.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch {
    Write-Log 'another supervisor is already running; exiting'
    exit 0
}

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint flags);'
$ES_CONTINUOUS_SYSTEM_REQUIRED = [uint32]2147483649  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this runs

Write-Log "supervisor started (pid $PID, port $Port, funnel $($Funnel.IsPresent))"
while (-not (Test-Path $python)) {  # the disk holding the project may come up after the task starts
    Write-Log "waiting for $python"
    Start-Sleep -Seconds 5
}

$stack = $null
$graceUntil = Get-Date
$portDownChecks = 0
$lastFunnel = [datetime]::MinValue
while ($true) {
    [void][Win32.Power]::SetThreadExecutionState($ES_CONTINUOUS_SYSTEM_REQUIRED)

    if (-not (Test-Port $ollamaPort)) { Start-Ollama }

    if ($null -eq $stack -or $stack.HasExited) {
        if ($null -ne $stack) { Write-Log 'run.py exited' }
        Start-Stack
    } elseif ((Get-Date) -gt $graceUntil) {
        if (Test-Port $Port) {
            $portDownChecks = 0
        } else {
            $portDownChecks++
            if ($portDownChecks -ge 6) {
                Write-Log "nothing is listening on port $Port; restarting"
                Start-Stack
            }
        }
    }

    if ($Funnel -and ((Get-Date) - $lastFunnel).TotalSeconds -gt 60) {
        Sync-Funnel
        $lastFunnel = Get-Date
    }
    Start-Sleep -Seconds 10
}
