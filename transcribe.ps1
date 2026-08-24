# Voice-memo transcription launcher (voice-memo-kit).
# Runs scripts/transcribe.py with the kit's Python venv. All other args pass through.
# The current directory is treated as the project root (use --project to override).
#
#   .\transcribe.ps1 [-DataDir <dir>] [python args...]
#
# Python resolution order:
#   1. $env:VOICE_MEMO_PYTHON (explicit override)
#   2. <DataDir>\.venv\Scripts\python.exe   (plugin mode: pass ${CLAUDE_PLUGIN_DATA})
#   3. <kit-root>\.venv\Scripts\python.exe  (real venv or a junction to an existing one)
# If none exists, run setup.ps1 first.
#
# The kit folder may be reached through a junction, so the root is resolved from the
# junction target. (Comments ASCII-only for Windows PowerShell 5.1.)
#
# NOTE: do NOT set $ErrorActionPreference = "Stop": the Python process emits library
# warnings to stderr. Success/failure is the Python exit code.

param(
    [string]$DataDir = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$kitRoot = $PSScriptRoot
$item = Get-Item -LiteralPath $kitRoot -Force
if ($item.LinkType) { $t = @($item.Target)[0]; if ($t) { $kitRoot = $t } }

$candidates = @()
if ($env:VOICE_MEMO_PYTHON) { $candidates += $env:VOICE_MEMO_PYTHON }
if ($DataDir) { $candidates += (Join-Path $DataDir ".venv\Scripts\python.exe") }
# Auto-probe the plugin data directory (~/.claude/plugins/data/<plugin-id>/) in case the
# caller could not pass -DataDir (e.g. ${CLAUDE_PLUGIN_DATA} was not substituted).
$dataRoot = Join-Path $env:USERPROFILE ".claude\plugins\data"
if (Test-Path $dataRoot) {
    Get-ChildItem $dataRoot -Directory -Filter "*voice-memo*" -ErrorAction SilentlyContinue |
        ForEach-Object { $candidates += (Join-Path $_.FullName ".venv\Scripts\python.exe") }
}
$candidates += (Join-Path $kitRoot ".venv\Scripts\python.exe")

$py = $null
foreach ($c in $candidates) { if (Test-Path $c) { $py = $c; break } }
if (-not $py) {
    Write-Host "Python venv not found. Tried:" -ForegroundColor Red
    $candidates | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host "Run setup.ps1 (see README.md), e.g.:" -ForegroundColor Red
    if ($DataDir) { Write-Host "  & `"$kitRoot\setup.ps1`" -DataDir `"$DataDir`"" -ForegroundColor Red }
    else { Write-Host "  & `"$kitRoot\setup.ps1`"" -ForegroundColor Red }
    exit 2
}

$script = Join-Path $kitRoot "scripts\transcribe.py"
& $py $script @Rest
exit $LASTEXITCODE
