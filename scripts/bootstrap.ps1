# scripts/bootstrap.ps1
param (
    [switch]$SkipOllama
)

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Checking ffmpeg & ffprobe ==="
$hasFfmpeg = $false
try {
    $res1 = & ffmpeg -version 2>&1
    $res2 = & ffprobe -version 2>&1
    if ($LASTEXITCODE -eq 0) { $hasFfmpeg = $true }
} catch {}

if (-not $hasFfmpeg) {
    if ((Test-Path "F:\tools\ffmpeg\bin\ffmpeg.exe") -and (Test-Path "F:\tools\ffmpeg\bin\ffprobe.exe")) {
        Write-Host "Found ffmpeg in F:\tools\ffmpeg\bin. Adding to session PATH..."
        $env:PATH = "F:\tools\ffmpeg\bin;" + $env:PATH
        $hasFfmpeg = $true
    } else {
        Write-Host "Attempting winget install Gyan.FFmpeg..."
        try {
            winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
            $hasFfmpeg = $true
        } catch {
            Write-Host "Winget install failed."
        }
    }
}
if ($hasFfmpeg) {
    Write-Host "ffmpeg & ffprobe are ready."
} else {
    Write-Warning "ffmpeg/ffprobe not found. Please install ffmpeg or place ffmpeg.exe and ffprobe.exe into F:\tools\ffmpeg\bin."
}

Write-Host "`n=== 2. uv sync ==="
& uv sync

Write-Host "`n=== 3. Ensuring .env file ==="
if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example"
}

function New-CryptoAccessCode {
    $chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    $bytes = New-Object byte[] 12
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($bytes)
    $rng.Dispose()
    return -join ($bytes | ForEach-Object { $chars[$_ % $chars.Length] })
}

# Update missing keys in .env without overwriting existing
$envContent = Get-Content ".env"
$existingKeys = @{}
foreach ($line in $envContent) {
    if ($line -match "^\s*([A-Za-z0-9_]+)\s*=") {
        $existingKeys[$matches[1]] = $true
    }
}

$exampleLines = Get-Content ".env.example"
$toAppend = @()
foreach ($line in $exampleLines) {
    if ($line -match "^\s*([A-Za-z0-9_]+)\s*=") {
        $k = $matches[1]
        if (-not $existingKeys.ContainsKey($k)) {
            $toAppend += $line
        }
    }
}

if ($toAppend.Count -gt 0) {
    Add-Content -Path ".env" -Value $toAppend
    Write-Host "Appended $($toAppend.Count) missing keys to .env"
}

# Ensure ACCESS_CODE has a non-placeholder value (12 chars alphanumeric)
$updatedEnv = Get-Content ".env"
$accessCodeSet = $false
$newLines = @()
foreach ($line in $updatedEnv) {
    if ($line -match "^\s*ACCESS_CODE=(.*)$") {
        $val = $matches[1].Trim()
        if ([string]::IsNullOrWhiteSpace($val) -or $val -eq "change_this_to_a_secure_token") {
            $randomCode = New-CryptoAccessCode
            $newLines += "ACCESS_CODE=$randomCode"
            $accessCodeSet = $true
        } else {
            $newLines += $line
            $accessCodeSet = $true
        }
    } else {
        $newLines += $line
    }
}
if (-not $accessCodeSet) {
    $randomCode = New-CryptoAccessCode
    $newLines += "ACCESS_CODE=$randomCode"
}
Set-Content -Path ".env" -Value $newLines
Write-Host ".env configuration validated (secrets protected)."

Write-Host "`n=== 4. Downloading Models ==="
& uv run python -c @"
import os
import sys
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()
hf_token = os.environ.get('HF_TOKEN') or None

with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

models_cfg = cfg.get('models', {})
whisper_dir = Path(models_cfg.get('whisper_dir', 'models/whisper'))
whisper_dir.mkdir(parents=True, exist_ok=True)

def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    return total / (1024 * 1024)

def repo_cache_size_mb(repo_id: str) -> float:
    cache_dir = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface' / 'hub'))
    repo_folder_name = 'models--' + repo_id.replace('/', '--')
    repo_path = cache_dir / repo_folder_name
    return dir_size_mb(repo_path)

print('--- Whisper models ---')
w3_dir = whisper_dir / 'faster-whisper-large-v3'
print('Downloading Systran/faster-whisper-large-v3...')
snapshot_download('Systran/faster-whisper-large-v3', local_dir=str(w3_dir))
print(f'Systran/faster-whisper-large-v3 total size: {dir_size_mb(w3_dir):.1f} MB')

w_turbo_dir = whisper_dir / 'large-v3-turbo'
print('Downloading large-v3-turbo (dropbox-dash/faster-whisper-large-v3-turbo)...')
snapshot_download('dropbox-dash/faster-whisper-large-v3-turbo', local_dir=str(w_turbo_dir))
print(f'large-v3-turbo resolved repo: dropbox-dash/faster-whisper-large-v3-turbo, size: {dir_size_mb(w_turbo_dir):.1f} MB')

print('--- Vision CLIP model ---')
clip_id = models_cfg.get('clip', 'openai/clip-vit-base-patch32')
print(f'Downloading {clip_id} into default HF cache...')
snapshot_download(clip_id)
print(f'{clip_id} size in HF cache: {repo_cache_size_mb(clip_id):.1f} MB')

print('--- ECAPA model ---')
ecapa_id = models_cfg.get('ecapa', 'speechbrain/spkrec-ecapa-voxceleb')
print(f'Downloading {ecapa_id} into default HF cache...')
snapshot_download(ecapa_id)
print(f'{ecapa_id} size in HF cache: {repo_cache_size_mb(ecapa_id):.1f} MB')

print('--- Pyannote models (gated) ---')
if hf_token:
    pyannote_repos = [
        'pyannote/speaker-diarization-3.1',
        'pyannote/segmentation-3.0',
        'pyannote/wespeaker-voxceleb-resnet34-LM',
    ]
    for repo in pyannote_repos:
        try:
            print(f'Attempting download of {repo} into default HF cache...')
            snapshot_download(repo, token=hf_token)
            print(f'{repo} downloaded ({repo_cache_size_mb(repo):.1f} MB)')
        except Exception as e:
            msg = str(e)
            if '401' in msg or '403' in msg or 'gated' in msg.lower():
                print(f'accept terms at huggingface.co/{repo}')
            else:
                print(f'pyannote {repo} error: {e}')
else:
    print('HF_TOKEN not set in .env; skipping pyannote download (ECAPA fallback available)')
"@

Write-Host "`n=== 5. Ensuring Ollama Vision LLM ==="
if (-not $SkipOllama) {
    Write-Host "Pulling qwen3-vl:8b (fallback qwen2.5vl:7b)..."
    $llmTag = "qwen3-vl:8b"
    $pulled = $false
    try {
        & ollama pull qwen3-vl:8b
        if ($LASTEXITCODE -eq 0) {
            $pulled = $true
            $llmTag = "qwen3-vl:8b"
        }
    } catch {}

    if (-not $pulled) {
        Write-Host "qwen3-vl:8b failed or not available, pulling qwen2.5vl:7b..."
        try {
            & ollama pull qwen2.5vl:7b
            if ($LASTEXITCODE -eq 0) {
                $pulled = $true
                $llmTag = "qwen2.5vl:7b"
            }
        } catch {}
    }

    if ($pulled) {
        Write-Host "Ollama model $llmTag ready. Updating config.yaml..."
        (Get-Content "config.yaml") -replace 'llm:\s*".*"', "llm: `"$llmTag`"" | Set-Content "config.yaml"
    } else {
        Write-Warning "Could not pull Ollama vision model. Ensure Ollama is running."
    }
}

Write-Host "`n=== Bootstrap Complete ==="
