"""tools/measure の共通: キットの scripts/ の import、オフラインの環境変数、HF キャッシュの tokenizer.json。

import しただけでは、重い import (torch・whisperx)、外部への接続、プロセスの起動をしない。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = KIT_ROOT / "scripts"

# WhisperX・pyannote を動かす測りで必ず付ける (#4 の計画)。
# HF_HUB_OFFLINE: Hugging Face からダウンロードしない (キャッシュに無ければ失敗させる)。
# PYANNOTE_METRICS_ENABLED: pyannote.audio 4.x の telemetry を送らない。
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "PYANNOTE_METRICS_ENABLED": "false"}

DEFAULT_MODEL_REPO = "Systran/faster-whisper-large-v3"


def use_kit_scripts() -> None:
    """キットの scripts/ (glossary.py など) を import できるようにする。"""
    p = str(SCRIPTS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


def force_offline_env() -> None:
    """WhisperX・faster-whisper・pyannote を import する前に呼ぶ。"""
    for k, v in OFFLINE_ENV.items():
        os.environ[k] = v


def hf_hub_cache() -> Path:
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(key):
            return Path(os.environ[key]).expanduser()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def find_tokenizer_json(model_repo: str = DEFAULT_MODEL_REPO) -> Path:
    """HF キャッシュにある、キットが使うモデルの tokenizer.json を返す (ダウンロードはしない)。"""
    repo_dir = hf_hub_cache() / ("models--" + model_repo.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    ref = repo_dir / "refs" / "main"
    if ref.is_file():
        cand = snapshots / ref.read_text(encoding="utf-8").strip() / "tokenizer.json"
        if cand.is_file():
            return cand
    cands = sorted(snapshots.glob("*/tokenizer.json")) if snapshots.is_dir() else []
    if not cands:
        raise FileNotFoundError(
            f"{model_repo} の tokenizer.json が HF キャッシュ ({repo_dir}) にありません。"
            "ダウンロードはしないので、--tokenizer でパスを渡してください"
        )
    return cands[-1]


def load_hf_tokenizer(path: Path | None = None):
    """tokenizers.Tokenizer を返す (faster-whisper が読むのと同じ tokenizer.json)。"""
    import tokenizers

    return tokenizers.Tokenizer.from_file(str(path or find_tokenizer_json()))


def hf_encoder(hf_tok):
    """(text) -> (ids, offsets)。faster-whisper の Tokenizer.encode と同じく特殊トークンを足さない。"""

    def encode(text: str):
        enc = hf_tok.encode(text, add_special_tokens=False)
        return list(enc.ids), [tuple(o) for o in enc.offsets]

    return encode


def median(values: list[float]) -> float:
    v = sorted(values)
    if not v:
        return 0.0
    mid = len(v) // 2
    return float(v[mid]) if len(v) % 2 else (v[mid - 1] + v[mid]) / 2.0
