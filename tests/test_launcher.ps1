# Launcher argument pass-through test (no Python venv / GPU needed).
# Points VOICE_MEMO_PYTHON at a stub that echoes its arguments, then checks that
# "--flags" and positional args reach the script and that -DataDir is stripped.
#
#   powershell -NoProfile -File tests\test_launcher.ps1
# Exit code 0 = pass.

$ErrorActionPreference = "Stop"
$kit = Split-Path -Parent $PSScriptRoot
$stub = Join-Path $env:TEMP ("vmk_echo_" + [guid]::NewGuid().ToString("N") + ".cmd")
"@echo off`r`necho ARGS=%*" | Set-Content -Path $stub -Encoding ascii

$fail = 0
function Check([string]$name, [bool]$cond) {
    if ($cond) { Write-Host "  ok   $name" } else { Write-Host "  FAIL $name" -ForegroundColor Red; $script:fail++ }
}

try {
    $env:VOICE_MEMO_PYTHON = $stub
    $out = (& (Join-Path $kit "transcribe.ps1") -DataDir "C:\dummy\data" --no-proofread --no-global --keep "memo one.m4a" | Out-String)
    Write-Host $out.Trim()
    Check "--no-proofread forwarded" ($out -match "--no-proofread")
    Check "--no-global forwarded"    ($out -match "--no-global")
    Check "--keep forwarded"         ($out -match "--keep")
    Check "positional forwarded"     ($out -match "memo one\.m4a")
    Check "-DataDir stripped"        ($out -notmatch "DataDir")
    Check "script path forwarded"    ($out -match "transcribe\.py")

    $out2 = (& (Join-Path $kit "transcribe.ps1") --timestamps | Out-String)
    Check "works without -DataDir"   ($out2 -match "--timestamps")
} finally {
    Remove-Item Env:\VOICE_MEMO_PYTHON -ErrorAction SilentlyContinue
    Remove-Item $stub -Force -ErrorAction SilentlyContinue
}

if ($fail -gt 0) { Write-Host "$fail check(s) failed" -ForegroundColor Red; exit 1 }
Write-Host "launcher test passed"
exit 0
