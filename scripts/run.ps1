# scripts/run.ps1 - Supervisor for 8x8 Class Quality Assessor Server and Worker
param (
    [int]$Port = 8765,
    [string]$HostAddress = "127.0.0.1"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path "$PSScriptRoot\.."
Set-Location $RepoRoot

# Offline mode guards
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

$LogsDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogsDir)) {
    New-Item -ItemType Directory -Path $LogsDir | Out-Null
}

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Error "Virtual environment python not found at $PythonExe. Run scripts\bootstrap.ps1 first."
    exit 1
}

Write-Host "================================================="
Write-Host "Starting Class Quality Assessor Server & Worker"
Write-Host "Host: $HostAddress | Port: $Port"
Write-Host "Offline ML mode: HF_HUB_OFFLINE=1"
Write-Host "Logs directed to: $LogsDir"
Write-Host "Press Ctrl+C to terminate both processes."
Write-Host "================================================="

function Start-ServerProcess {
    $ServerLog = Join-Path $LogsDir "server.log"
    return Start-Process -FilePath $PythonExe `
        -ArgumentList "-m", "uvicorn", "classeval.server.main:create_app", "--factory", "--host", $HostAddress, "--port", "$Port" `
        -RedirectStandardOutput $ServerLog `
        -RedirectStandardError $ServerLog `
        -PassThru `
        -NoNewWindow
}

function Start-WorkerProcess {
    $WorkerLog = Join-Path $LogsDir "worker.log"
    return Start-Process -FilePath $PythonExe `
        -ArgumentList "-m", "classeval.worker" `
        -RedirectStandardOutput $WorkerLog `
        -RedirectStandardError $WorkerLog `
        -PassThru `
        -NoNewWindow
}

$ServerProc = Start-ServerProcess
$WorkerProc = Start-WorkerProcess

try {
    while ($true) {
        Start-Sleep -Seconds 2

        if ($ServerProc.HasExited) {
            $exitCode = $ServerProc.ExitCode
            Write-Warning "Server exited (code $exitCode). Restarting in 3 seconds..."
            Start-Sleep -Seconds 3
            $ServerProc = Start-ServerProcess
            Write-Host "Server restarted (PID $($ServerProc.Id))."
        }

        if ($WorkerProc.HasExited) {
            $exitCode = $WorkerProc.ExitCode
            Write-Warning "Worker exited (code $exitCode). Restarting in 3 seconds..."
            Start-Sleep -Seconds 3
            $WorkerProc = Start-WorkerProcess
            Write-Host "Worker restarted (PID $($WorkerProc.Id))."
        }
    }
}
finally {
    Write-Host "`nStopping server and worker processes..."
    if ($ServerProc -and -not $ServerProc.HasExited) {
        Stop-Process -Id $ServerProc.Id -Force -ErrorAction SilentlyContinue
    }
    if ($WorkerProc -and -not $WorkerProc.HasExited) {
        Stop-Process -Id $WorkerProc.Id -Force -ErrorAction SilentlyContinue
    }
    Write-Host "Stopped."
}
