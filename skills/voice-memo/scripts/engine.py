"""WhisperX 文字起こしコア（align 省略の軽量版）。

content_pipeline/scripts/02_transcribe.py の transcribe_local を簡略移植したもの。
ボイスメモ用途では単語タイムスタンプ（align）は不要なので、align ステップを省略して
高速化・align モデルのロード回避を行う。動画パイプライン側のコードには一切手を加えない。

この関数は WhisperX が import できる Python（= 共有 venv tools/.venv）上で実行される前提。
"""
from __future__ import annotations

from pathlib import Path

# whisperx.load_audio は 16kHz モノラル float32 を返す（SAMPLE_RATE 固定）
WHISPER_SAMPLE_RATE = 16000


def transcribe_audio(
    input_path: Path,
    *,
    model_name: str = "large-v3",
    compute_type: str = "int8",
    language: str = "ja",
    batch_size: int = 8,
    chunk_size: int = 30,
    device: str = "cuda",
    initial_prompt: str = "",
) -> dict:
    """音声ファイルを WhisperX で文字起こしして dict を返す。

    Returns:
        {"segments": [...], "language": str, "engine": str, "duration": float(秒)}
        各 segment は {"start", "end", "text"}（words は align 省略のため含まない）。
    """
    import whisperx

    # 実績のある content_pipeline/02_transcribe.py と挙動を揃えるため、
    # asr_options は initial_prompt のみ渡す（他は whisperx のデフォルトに従う。
    # whisperx 3.8.x のデフォルトは condition_on_previous_text=False を含む）。
    asr_options: dict | None = None
    if initial_prompt:
        asr_options = {"initial_prompt": initial_prompt}
        preview = initial_prompt[:80] + ("..." if len(initial_prompt) > 80 else "")
        print(f"  initial_prompt: {preview}")

    print(f"[1/2] WhisperX {model_name} ({compute_type}/{device}) をロード中...")
    try:
        model = whisperx.load_model(
            model_name, device, compute_type=compute_type, language=language,
            asr_options=asr_options,
        )
    except TypeError:
        # 古い whisperx で asr_options 非対応の場合のフォールバック
        print("  [WARN] WhisperX が asr_options 非対応、既定設定で続行")
        model = whisperx.load_model(
            model_name, device, compute_type=compute_type, language=language
        )

    print(f"  音声をロード中: {input_path.name}")
    audio = whisperx.load_audio(str(input_path))
    duration = float(len(audio)) / WHISPER_SAMPLE_RATE

    print("[2/2] 文字起こし実行中...")
    try:
        result = model.transcribe(
            audio, batch_size=batch_size, language=language, chunk_size=chunk_size
        )
    except TypeError:
        result = model.transcribe(audio, batch_size=batch_size, language=language)

    return {
        "segments": result.get("segments", []),
        "language": language,
        "engine": f"whisperx-{model_name}-{compute_type}",
        "duration": duration,
    }
