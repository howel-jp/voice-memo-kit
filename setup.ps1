# voice-memo-kit venv setup.
# Creates the Python venv the launcher (transcribe.ps1) uses.
#
#   .\setup.ps1 -DataDir <dir>                       # build a new venv at <dir>\.venv (~7GB download)
#   .\setup.ps1 -DataDir <dir> -LinkTo <existing venv> # junction <dir>\.venv -> existing venv (no download)
#   .\setup.ps1                                        # same, at <kit-root>\.venv
#
# Requirements for a new build: Python 3.10 (py launcher or python on PATH), NVIDIA GPU with a
# CUDA 12.x driver (requirements.lock.txt pins torch cu126 wheels), ffmpeg on PATH.
#
# (Comments kept ASCII-only for Windows PowerShell 5.1.)

param(
    [string]$DataDir = "",
    [string]$LinkTo = "",
    [string]$Python = ""
)

$kitRoot = $PSScriptRoot
$item = Get-Item -LiteralPath $kitRoot -Force
if ($item.LinkType) { $t = @($item.Target)[0]; if ($t) { $kitRoot = $t } }

$base = if ($DataDir) { $DataDir } else { $kitRoot }
if (-not (Test-Path $base)) { New-Item -ItemType Directory -Force $base | Out-Null }
$venv = Join-Path $base ".venv"
$venvPy = Join-Path $venv "Scripts\python.exe"

if (Test-Path $venvPy) {
    Write-Host "venv already exists: $venv"
} elseif ($LinkTo) {
    $targetPy = Join-Path $LinkTo "Scripts\python.exe"
    if (-not (Test-Path $targetPy)) { Write-Host "LinkTo is not a venv: $LinkTo" -ForegroundColor Red; exit 1 }
    New-Item -ItemType Junction -Path $venv -Target (Resolve-Path $LinkTo).Path | Out-Null
    Write-Host "linked: $venv -> $LinkTo"
} else {
    $py = $Python
    if (-not $py) {
        if (Get-Command py -ErrorAction SilentlyContinue) { $py = "py -3.10" }
        elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
        else { Write-Host "Python 3.10 not found. Install it or pass -Python <path>." -ForegroundColor Red; exit 1 }
    }
    Write-Host "creating venv with '$py' at $venv (this downloads ~7GB)..."
    Invoke-Expression "$py -m venv `"$venv`""
    if (-not (Test-Path $venvPy)) { Write-Host "venv creation failed" -ForegroundColor Red; exit 1 }
    & $venvPy -m pip install --upgrade pip
    & $venvPy -m pip install -r (Join-Path $kitRoot "requirements.lock.txt")
    if ($LASTEXITCODE -ne 0) { Write-Host "pip install failed" -ForegroundColor Red; exit 1 }
}

# Verify
& $venvPy -c "import torch, whisperx; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())"
if ($LASTEXITCODE -ne 0) { Write-Host "verification failed: torch/whisperx not importable" -ForegroundColor Red; exit 1 }
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "WARNING: ffmpeg not found on PATH (needed for audio decoding)" -ForegroundColor Yellow
}
Write-Host "setup OK: $venvPy"
exit 0
