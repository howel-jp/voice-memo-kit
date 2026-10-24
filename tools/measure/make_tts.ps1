# Make a 16 kHz / 16-bit / mono wav from synthetic text with Windows speech synthesis (System.Speech).
# Used by #4 (U3). Never feed user recordings or memo text: pass only text made by
# `python tools/measure/synth.py tts ...`. Kept ASCII-only so Windows PowerShell 5.1 parses it without a BOM.
#
#   powershell -NoProfile -File tools\measure\make_tts.ps1 -TextFile <dir>\dense.txt -OutFile <dir>\dense.wav -Rate 8
#
# -Rate is -10..10 (higher is faster). Prints the length in seconds at the end.
param(
    [Parameter(Mandatory = $true)][string]$TextFile,
    [Parameter(Mandatory = $true)][string]$OutFile,
    [int]$Rate = 0,
    [string]$Voice = "Microsoft Haruka Desktop"
)
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
$text = [System.IO.File]::ReadAllText($TextFile, [System.Text.Encoding]::UTF8)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $s.SelectVoice($Voice)
    $s.Rate = $Rate
    $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
    $s.SetOutputToWaveFile($OutFile, $fmt)
    $s.Speak($text)
} finally {
    $s.Dispose()
}
$bytes = (Get-Item $OutFile).Length
$seconds = [math]::Round(($bytes - 44) / (16000 * 2), 1)
Write-Output "wrote $OutFile ($seconds s, $($text.Length) chars, rate $Rate)"
