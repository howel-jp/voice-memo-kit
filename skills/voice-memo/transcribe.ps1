# Voice-memo transcription launcher (voice-memo-kit).
# Runs scripts/transcribe.py with the kit's Python venv. All args pass through.
# Works from any project directory: the script treats the current directory
# as the project root (use --project to override).
#
# Python resolution order:
#   1. $env:VOICE_MEMO_PYTHON (explicit override)
#   2. <kit-root>\.venv\Scripts\python.exe (real venv or a junction to an existing one)
#
# This skill folder may be reached through a junction (~/.claude/skills/voice-memo),
# so the kit root is resolved from the junction target, not from the link path.
#
# (Comments kept ASCII-only: Windows PowerShell 5.1 parses BOM-less .ps1 as the
#  system ANSI code page, which corrupts multibyte chars.)
#
# NOTE: do NOT set $ErrorActionPreference = "Stop" here. The Python process emits
# library warnings to stderr; under "Stop" PowerShell can treat native stderr as a
# terminating error. Success/failure is determined by the Python exit code.

$skillDir = $PSScriptRoot
$item = Get-Item -LiteralPath $skillDir -Force
if ($item.LinkType) {
    $target = @($item.Target)[0]
    if ($target) { $skillDir = $target }
}
$kitRoot = (Resolve-Path (Join-Path $skillDir "..\..")).Path

$py = $env:VOICE_MEMO_PYTHON
if (-not $py) { $py = Join-Path $kitRoot ".venv\Scripts\python.exe" }
if (-not (Test-Path $py)) {
    Write-Host "Python venv not found: $py" -ForegroundColor Red
    Write-Host "Set up <kit-root>\.venv (see README.md) or set VOICE_MEMO_PYTHON." -ForegroundColor Red
    exit 1
}

$script = Join-Path $skillDir "scripts\transcribe.py"
& $py $script @args
exit $LASTEXITCODE
