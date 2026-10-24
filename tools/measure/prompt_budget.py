"""長い initial_prompt が、1 チャンクで出せるトークン数を減らすかを確かめる (#4 の U3、#3 §2 の 3b)。

whisperx 3.8.5 の batched の経路は、prompt の長さによらず ctranslate2 の generate に max_length=448 を渡す。
448 が prompt を含む長さなら、prompt を使い切る (228 = 印 5 + 前文 223) と、出せるのは 220 トークンになる。

    # (a) 仕組み: ctranslate2 の generate を直に呼び、prompt の長さ P と max_length M を変える
    python tools/measure/prompt_budget.py mechanism --audio <dir>/dense.wav --device cuda
    # (b) 影響: キットの engine.transcribe_audio で、prompt なしと長い合成の prompt を比べる
    python tools/measure/prompt_budget.py effect --audio <dir>/dense.wav --source-text <dir>/dense.txt --device cuda

音声は合成の音声だけを使う (make_tts.ps1)。どちらも HF_HUB_OFFLINE=1 と PYANNOTE_METRICS_ENABLED=false を
このプロセスに設定してから import する (ダウンロードと telemetry をしない)。モデルがキャッシュに無ければ失敗する。
import しただけでは、重い import をしない。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _kit  # noqa: E402

_kit.use_kit_scripts()

WHISPER_MAX_LENGTH = 448
KEEP = WHISPER_MAX_LENGTH // 2 - 1


def synthetic_prompt_text(min_tokens: int, encode, seed: int = 0) -> str:
    """min_tokens 以上になる、合成の辞書の initial_prompt (キットの build_initial_prompt で作る)。"""
    import glossary
    import synth

    n = 10
    while True:
        prompt = glossary.build_initial_prompt(synth.synth_entries("M", n, seed))
        ids, _ = encode(" " + prompt.strip())
        if len(ids) >= min_tokens:
            return prompt
        n += 10


def mechanism(audio: Path, device: str, compute_type: str, ps: list[int], seed: int = 0,
              small_room: int = 16, seconds: float = 30.0) -> list[dict]:
    _kit.force_offline_env()
    import numpy as np
    from faster_whisper.audio import decode_audio, pad_or_trim
    from faster_whisper.tokenizer import Tokenizer
    from faster_whisper.transcribe import WhisperModel

    model = WhisperModel("large-v3", device=device, compute_type=compute_type, local_files_only=True)
    tok = Tokenizer(model.hf_tokenizer, model.model.is_multilingual, task="transcribe", language="ja")
    wav = decode_audio(str(audio), sampling_rate=16000)
    chunk = wav[: int(seconds * 16000)]
    feats = np.ascontiguousarray(pad_or_trim(model.feature_extractor(chunk)))
    enc = model.encode(feats)

    src = synthetic_prompt_text(KEEP + 10, _kit.hf_encoder(model.hf_tokenizer), seed)
    src_ids = tok.encode(" " + src.strip())
    rows = []
    for p in ps:
        prev = src_ids[-p:] if p else []
        prompt = model.get_prompt(tok, prev, without_timestamps=True)
        for m in (WHISPER_MAX_LENGTH, len(prompt) + small_room):
            res = model.model.generate(
                enc, [prompt], beam_size=5, patience=1, length_penalty=1, repetition_penalty=1,
                no_repeat_ngram_size=0, max_length=m, suppress_blank=True, suppress_tokens=[-1],
                return_scores=True,
            )
            seq = list(res[0].sequences_ids[0])
            text_ids = [t for t in seq if t < tok.eot]
            text = tok.decode(text_ids)
            rows.append({
                "device": device,
                "P": p,
                "prompt_len": len(prompt),
                "max_length": m,
                "out_tokens": len(seq),
                "room_if_prompt_counts": m - len(prompt),
                "at_cap_if_prompt_counts": len(seq) >= m - len(prompt),
                "out_chars": len(text.replace(" ", "")),
            })
    return rows


def effect(audio: Path, source_text: Path | None, device: str, compute_type: str, seed: int = 0) -> list[dict]:
    _kit.force_offline_env()
    import engine

    encode = _kit.hf_encoder(_kit.load_hf_tokenizer())
    src_chars = len(source_text.read_text(encoding="utf-8")) if source_text else None
    long_prompt = synthetic_prompt_text(KEEP + 10, encode, seed)
    rows = []
    for label, prompt in (("none", ""), ("long", long_prompt)):
        res = engine.transcribe_audio(audio, model_name="large-v3", compute_type=compute_type, language="ja",
                                      batch_size=8, chunk_size=30, device=device, initial_prompt=prompt)
        segs = res["segments"]
        seg_tokens = [len(encode((s.get("text") or "").strip())[0]) for s in segs]
        seg_secs = [float(s.get("end", 0)) - float(s.get("start", 0)) for s in segs]
        chars = sum(len((s.get("text") or "").replace(" ", "").strip()) for s in segs)
        prompt_tokens = len(encode(" " + prompt.strip())[0]) if prompt else 0
        rows.append({
            "device": device,
            "prompt": label,
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_kept": min(prompt_tokens, KEEP),
            "duration_sec": round(res["duration"], 1),
            "segments": len(segs),
            "seg_seconds": [round(x, 1) for x in seg_secs],
            "seg_tokens": seg_tokens,
            "max_seg_tokens": max(seg_tokens) if seg_tokens else 0,
            "out_chars": chars,
            "source_chars": src_chars,
            "tokens_per_30s": round(sum(seg_tokens) / (res["duration"] / 30.0), 1) if res["duration"] else None,
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="長い prompt と 1 チャンクの出力の上限 (#4 U3)")
    ap.add_argument("--json", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mechanism")
    m.add_argument("--audio", type=Path, required=True)
    m.add_argument("--device", default="cuda")
    m.add_argument("--compute-type", default="int8")
    m.add_argument("--ps", default="0,50,100,150,223")
    m.add_argument("--seconds", type=float, default=30.0)
    e = sub.add_parser("effect")
    e.add_argument("--audio", type=Path, required=True)
    e.add_argument("--source-text", type=Path, default=None)
    e.add_argument("--device", default="cuda")
    e.add_argument("--compute-type", default="int8")
    args = ap.parse_args(argv)

    if args.cmd == "mechanism":
        rows = mechanism(args.audio, args.device, args.compute_type,
                         [int(x) for x in args.ps.split(",") if x != ""], seconds=args.seconds)
    else:
        rows = effect(args.audio, args.source_text, args.device, args.compute_type)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
