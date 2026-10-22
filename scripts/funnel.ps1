# scripts/funnel.ps1 - Tailscale Funnel management for public mobile evaluation
param (
    [switch]$Reset,
    [switch]$DisableSleep,
    [int]$Port = 8765
)

$ErrorActionPreference = "Stop"

if ($DisableSleep) {
    Write-Host "Disabling standby and monitor sleep timeouts while connected to AC power..."
    Write-Host "Executing: powercfg /change standby-timeout-ac 0"
    & powercfg /change standby-timeout-ac 0
    Write-Host "Executing: powercfg /change monitor-timeout-ac 0"
    & powercfg /change monitor-timeout-ac 0
    Write-Host "Sleep settings updated."
    if (-not $Reset -and -not $PSBoundParameters.ContainsKey('Port')) {
        return
    }
}

if ($Reset) {
    Write-Host "Resetting Tailscale Funnel configuration..."
    & tailscale funnel reset
    Write-Host "Funnel reset complete."
    return
}

Write-Host "Starting Tailscale Funnel in background on port $Port..."
& tailscale funnel --bg $Port

Write-Host "Tailscale Funnel status:"
& tailscale funnel status
