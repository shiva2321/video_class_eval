<#
Live view of everything the server is doing: page requests and uploads, the worker, Ollama, the supervisor, job progress.
Open it any time (double-click watch_logs.cmd); it only reads, so closing the window never affects the server.

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\service\watch_logs.ps1 [-History 8] [-All]

-History  how many old lines of each log to show first.
-All      show every Ollama call; by default they are counted ("12 model calls in the last 10 s").
Ctrl+C or closing the window stops watching.
#>
param(
    [int]$History = 8,
    [switch]$All,
    [int]$Port = 8765
)

$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$logDir = Join-Path $root 'logs'
$jobsDir = Join-Path $root 'jobs'
$tailscale = 'C:\Program Files\Tailscale\tailscale.exe'
$Host.UI.RawUI.WindowTitle = 'Class Assessor - live server log'
try {
    $Host.UI.RawUI.BufferSize = New-Object Management.Automation.Host.Size(180, 8000)
    $Host.UI.RawUI.WindowSize = New-Object Management.Automation.Host.Size(150, 46)
} catch { }

# name, file, colour
$sources = @(
    @{ Tag = 'SUPERVISOR'; File = 'supervisor.log';  Color = 'Magenta' },
    @{ Tag = 'PAGE      '; File = 'server.log';      Color = 'Cyan' },
    @{ Tag = 'PAGE ERR  '; File = 'server.err.log';  Color = 'DarkCyan' },
    @{ Tag = 'WORKER    '; File = 'worker.err.log';  Color = 'Yellow' },
    @{ Tag = 'WORKER OUT'; File = 'worker.log';      Color = 'Yellow' },
    @{ Tag = 'OLLAMA    '; File = 'ollama.err.log';  Color = 'Green' },
    @{ Tag = 'OLLAMA OUT'; File = 'ollama.log';      Color = 'Green' }
)
foreach ($s in $sources) { $s.Path = Join-Path $logDir $s.File; $s.Pos = -1L; $s.Partial = '' }

function Write-Line([string]$tag, [string]$text, [string]$color) {
    Write-Host ('{0} {1} ' -f (Get-Date -Format 'HH:mm:ss'), $tag) -NoNewline -ForegroundColor DarkGray
    Write-Host $text -ForegroundColor $color
}

function Test-Port([int]$p) {
    $client = New-Object Net.Sockets.TcpClient
    try {
        $pending = $client.BeginConnect('127.0.0.1', $p, $null, $null)
        return ($pending.AsyncWaitHandle.WaitOne(1000) -and $client.Connected)
    } catch { return $false } finally { $client.Close() }
}

# Reads what was appended to a log since the last call; returns the complete new lines.
function Read-New($s) {
    if (-not (Test-Path $s.Path)) { $s.Pos = -1L; return @() }
    try {
        $fs = [IO.File]::Open($s.Path, 'Open', 'Read', 'ReadWrite, Delete')
    } catch { return @() }
    try {
        $len = $fs.Length
        if ($s.Pos -lt 0) {  # first look: start a little before the end
            $start = [Math]::Max(0, $len - 4096)
            $s.Pos = $start
            $first = $true
        } elseif ($len -lt $s.Pos) {  # file was truncated or replaced
            $s.Pos = 0; $s.Partial = ''; $first = $false
        } else { $first = $false }
        if ($len -eq $s.Pos) { return @() }
        $fs.Seek($s.Pos, 'Begin') | Out-Null
        $bytes = New-Object byte[] ($len - $s.Pos)
        $read = $fs.Read($bytes, 0, $bytes.Length)
        $s.Pos += $read
    } finally { $fs.Close() }
    $text = $s.Partial + [Text.Encoding]::UTF8.GetString($bytes, 0, $read)
    $parts = $text -split "`r`n|`n|`r"
    $s.Partial = $parts[-1]
    $lines = @()
    if ($parts.Count -gt 1) {
        $complete = $parts[0..($parts.Count - 2)]
        if ($first -and $start -gt 0) { $complete = $complete | Select-Object -Skip 1 }  # we started mid-line
        $lines = @($complete | Where-Object { $_ -and $_.Trim() })
    }
    if ($first -and $lines.Count -gt $History) { $lines = $lines[($lines.Count - $History)..($lines.Count - 1)] }
    return $lines
}

$lastJob = @{}
function Watch-Jobs {
    if (-not (Test-Path $jobsDir)) { return }
    foreach ($dir in Get-ChildItem $jobsDir -Directory -ErrorAction SilentlyContinue) {
        $file = Join-Path $dir.FullName 'job.json'
        try { $j = Get-Content $file -Raw -ErrorAction Stop | ConvertFrom-Json } catch { continue }
        $state = '{0}|{1}|{2}' -f $j.status, $j.stage, [int]($j.progress * 100)
        if (-not $script:jobsSeeded) { $lastJob[$j.id] = $state; continue }  # first pass: remember, don't announce history
        if ($lastJob[$j.id] -eq $state) { continue }
        $lastJob[$j.id] = $state
        $color = switch ($j.status) { 'done' { 'Green' } 'failed' { 'Red' } 'queued' { 'Gray' } default { 'White' } }
        $extra = if ($j.status -eq 'failed' -and $j.error) { "  error: $($j.error.code)" } else { '' }
        Write-Line 'JOB       ' ("{0}...  {1,-8} {2,-10} {3,3}%{4}" -f $j.id.Substring(0, 6), $j.status, $j.stage, [int]($j.progress * 100), $extra) $color
    }
    $script:jobsSeeded = $true
}

function Get-Heartbeat {
    $page = if (Test-Port $Port) { 'page up' } else { 'PAGE DOWN' }
    $oll = if (Test-Port 11434) { 'ollama up' } else { 'OLLAMA DOWN' }
    $gpu = 'gpu n/a'
    try {
        $g = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>$null) -split ',\s*'
        $gpu = 'gpu {0}% {1:N1} GB' -f $g[0], ([double]$g[1] / 1024)
    } catch { }
    $running = 0; $queued = 0
    foreach ($dir in Get-ChildItem $jobsDir -Directory -ErrorAction SilentlyContinue) {
        try { $st = (Get-Content (Join-Path $dir.FullName 'job.json') -Raw | ConvertFrom-Json).status } catch { continue }
        if ($st -eq 'running') { $running++ } elseif ($st -eq 'queued') { $queued++ }
    }
    "{0} | {1} | {2} | jobs running {3}, waiting {4}" -f $page, $oll, $gpu, $running, $queued
}

Clear-Host
Write-Host '=== Class Assessor server - live log (Ctrl+C or close this window to stop watching) ===' -ForegroundColor White
$url = ''
if (Test-Path $tailscale) {
    try { $url = ((& $tailscale funnel status 2>$null) | Select-String 'https://\S+' | Select-Object -First 1).Matches[0].Value } catch { }
}
Write-Host ("local page : http://127.0.0.1:{0}" -f $Port) -ForegroundColor Gray
if ($url) { Write-Host ("public link: {0}" -f $url) -ForegroundColor Gray }
Write-Host ("logs       : {0}" -f $logDir) -ForegroundColor Gray
Write-Host "Colours: magenta supervisor, cyan page, yellow worker, green Ollama, white job progress, grey status line every 30 s`n" -ForegroundColor DarkGray
Write-Line 'STATUS    ' (Get-Heartbeat) 'Gray'

$chatCalls = 0
$lastFlush = Get-Date
$lastBeat = Get-Date
$script:jobsSeeded = $false
Watch-Jobs
while ($true) {
    foreach ($s in $sources) {
        foreach ($line in (Read-New $s)) {
            if (-not $All -and $line -match 'httpx: HTTP Request: POST http://127\.0\.0\.1:11434/api/chat') { $chatCalls++; continue }
            $color = $s.Color
            if ($line -match '\b(ERROR|Traceback|Exception|CRITICAL|failed)\b') { $color = 'Red' }
            elseif ($line -match '\bWARNING\b') { $color = 'DarkYellow' }
            Write-Line $s.Tag $line $color
        }
    }
    Watch-Jobs
    $now = Get-Date
    if (($now - $lastFlush).TotalSeconds -ge 10) {
        if ($chatCalls -gt 0) { Write-Line 'OLLAMA    ' ("{0} model calls in the last 10 s" -f $chatCalls) 'DarkGreen'; $chatCalls = 0 }
        $lastFlush = $now
    }
    if (($now - $lastBeat).TotalSeconds -ge 30) {
        Write-Line 'STATUS    ' (Get-Heartbeat) 'Gray'
        $lastBeat = $now
    }
    Start-Sleep -Milliseconds 500
}
