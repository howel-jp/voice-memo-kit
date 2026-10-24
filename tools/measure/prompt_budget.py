"""長い initial_prompt が、1 チャンクで出せるトークン数を減らすかを確かめる (#4 の U3、#3 §2 の 3b)。

whisperx 3.8.5 の batched の経路は、prompt の長さによらず ctranslate2 の generate に max_length=448 を渡す。
ここでは generate を直に呼び、prompt の長さと max_length M を変えて、出たトークン数を
2 つの上限の読み (M - len(prompt) + 1 と M // 2) と並べる。

    # (a) 仕組み: generate を直に呼ぶ (合成の辞書 n 項目の prompt、または長い prompt の後ろ P トークン)
    python tools/measure/prompt_budget.py mechanism --audio <dir>/dense.wav --mode full --values 0,10,20,78
    python tools/measure/prompt_budget.py mechanism --audio <dir>/dense.wav --mode tail --values 0,50,150,223
    # (b) 影響: キットの engine.transcribe_audio で、prompt の長さを変えて比べる
    python tools/measure/prompt_budget.py effect --audio <dir>/dense.wav --source-text <dir>/dense.txt --items 0,10,20,78

effect は prompt を替えるたびにモデルを読み直す。1 プロセスで続けると、2026-10-24 に 172 秒の音声で
10 分を超えても終わらなかったことがある (原因は未確認)。確実に測るときは --items を 1 つずつ別のプロセスで渡す。

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


def _prompt_sources(encode, mode: str, values: list[int], seed: int = 0) -> list[tuple[str, str | None, int]]:
    """(ラベル, initial_prompt の文字列か None, 後ろから使うトークン数か -1) の一覧。

    full: キットと同じく build_initial_prompt(合成の辞書 n 項目) をそのまま渡す (get_prompt が後ろ 223 を残す)。
    tail: 長い合成の prompt の後ろ P トークンだけを前文にする (前置きの文を含まない。仕組みの確かめ用)。
    """
    import glossary
    import synth

    out = []
    if mode == "full":
        for n in values:
            text = glossary.build_initial_prompt(synth.mixed_entries(n, seed)) if n else ""
            out.append((f"items={n}", text, -1))
    else:
        src = synthetic_prompt_text(KEEP + 10, encode, seed)
        for p in values:
            out.append((f"tail={p}", src, p))
    return out


def predicted_cap(max_length: int, prompt_len: int) -> int:
    """出せるトークン数の上限の読み (この道具の実測から): min(M - len(prompt) + 1, M // 2)。"""
    return min(max_length - prompt_len + 1, max_length // 2)


def mechanism(audio: Path, device: str, compute_type: str, mode: str, values: list[int], rooms: list[int],
              seed: int = 0, seconds: float = 30.0) -> list[dict]:
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

    rows = []
    for label, text, p in _prompt_sources(_kit.hf_encoder(model.hf_tokenizer), mode, values, seed):
        ids = tok.encode(" " + text.strip()) if text else []
        prev = ids if p < 0 else (ids[-p:] if p else [])
        prompt = model.get_prompt(tok, prev, without_timestamps=True)  # whisperx と同じ (後ろ 223 を残す)
        for m in [WHISPER_MAX_LENGTH] + [len(prompt) + r for r in rooms]:
            res = model.model.generate(
                enc, [prompt], beam_size=5, patience=1, length_penalty=1, repetition_penalty=1,
                no_repeat_ngram_size=0, max_length=m, suppress_blank=True, suppress_tokens=[-1],
                return_scores=True,
            )
            seq = list(res[0].sequences_ids[0])
            out_text = tok.decode([t for t in seq if t < tok.eot]).replace(" ", "")
            cap = predicted_cap(m, len(prompt))
            rows.append({
                "device": device,
                "prompt": label,
                "prompt_len": len(prompt),
                "max_length": m,
                "out_tokens": len(seq),
                "cap_if_prompt_counts": m - len(prompt) + 1,
                "cap_half": m // 2,
                "predicted_cap": cap,
                "at_predicted_cap": len(seq) == cap,
                "over_predicted_cap": len(seq) > cap,
                "out_chars": len(out_text),
                "out_head": out_text[:12],  # 合成の音声の文字起こし (中身の確かめ用)
            })
    return rows


def effect(audio: Path, source_text: Path | None, device: str, compute_type: str, items: list[int],
           seed: int = 0) -> list[dict]:
    """キットの engine.transcribe_audio (VAD で 30 秒までに分けて batched で生成) で、prompt の長さを変えて比べる。"""
    _kit.force_offline_env()
    import engine

    import glossary
    import synth

    encode = _kit.hf_encoder(_kit.load_hf_tokenizer())
    source = source_text.read_text(encoding="utf-8").replace("\n", "") if source_text else ""
    src_chars = len(source) if source_text else None
    rows = []
    for n in items:
        entries = synth.mixed_entries(n, seed) if n else []
        text = glossary.build_initial_prompt(entries) if entries else ""
        res = engine.transcribe_audio(audio, model_name="large-v3", compute_type=compute_type, language="ja",
                                      batch_size=8, chunk_size=30, device=device, initial_prompt=text)
        segs = res["segments"]
        seg_tokens = [len(encode((s.get("text") or "").strip())[0]) for s in segs]
        seg_secs = [float(s.get("end", 0)) - float(s.get("start", 0)) for s in segs]
        out = "".join((s.get("text") or "").replace(" ", "").strip() for s in segs)
        chars = len(out)
        prompt_tokens = len(encode(" " + text.strip())[0]) if text else 0
        # 音声に無い辞書の語が出力に出た数 (prompt の語の並びを書き写したしるし)
        prompt_only = [e["canonical"] for e in entries if e["canonical"] not in source]
        rows.append({
            "device": device,
            "prompt": f"items={n}",
            "prompt_only_words": len(prompt_only),
            "prompt_only_words_in_output": sum(1 for w in prompt_only if w in out),
            "out_head": out[:12],  # 合成の音声の文字起こし (中身の確かめ用)
            "prompt_tokens": prompt_tokens,
            "prompt_tokens_kept": min(prompt_tokens, KEEP),
            "duration_sec": round(res["duration"], 1),
            "segments": len(segs),
            "seg_seconds": [round(x, 1) for x in seg_secs],
            "seg_tokens": seg_tokens,
            "max_seg_tokens": max(seg_tokens) if seg_tokens else 0,
            "out_chars": chars,
            "source_chars": src_chars,
            "out_per_source": round(chars / src_chars, 3) if src_chars else None,
            "tokens_per_30s": round(sum(seg_tokens) / (res["duration"] / 30.0), 1) if res["duration"] else None,
        })
        _free_models(engine)
    return rows


def _free_models(engine) -> None:
    """engine は initial_prompt ごとにモデルをキャッシュするので、prompt を替えるたびに捨てる
    (捨てないと large-v3 が prompt の数だけ GPU に載り、8GB の GPU では極端に遅くなる。2026-10-24 に観測)。"""
    import gc

    engine.clear_model_cache()
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip() != ""]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="長い prompt と 1 チャンクの出力の上限 (#4 U3)")
    ap.add_argument("--json", type=Path, default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mechanism")
    m.add_argument("--audio", type=Path, required=True)
    m.add_argument("--device", default="cuda")
    m.add_argument("--compute-type", default="int8")
    m.add_argument("--mode", choices=["full", "tail"], default="full",
                   help="full: 合成の辞書 n 項目の prompt をそのまま。tail: 長い prompt の後ろ P トークン")
    m.add_argument("--values", default="0,5,10,20,30,78",
                   help="full なら項目数 n、tail なら P (カンマ区切り)")
    m.add_argument("--rooms", default="16,100,200", help="max_length = len(prompt) + room の room")
    m.add_argument("--seconds", type=float, default=30.0)
    e = sub.add_parser("effect")
    e.add_argument("--audio", type=Path, required=True)
    e.add_argument("--source-text", type=Path, default=None)
    e.add_argument("--device", default="cuda")
    e.add_argument("--compute-type", default="int8")
    e.add_argument("--items", default="0,10,20,78", help="合成の辞書の項目数 (0 は prompt なし)")
    args = ap.parse_args(argv)
    _kit.utf8_stdout()

    if args.cmd == "mechanism":
        rows = mechanism(args.audio, args.device, args.compute_type, args.mode, _ints(args.values),
                         _ints(args.rooms), seconds=args.seconds)
    else:
        rows = effect(args.audio, args.source_text, args.device, args.compute_type, _ints(args.items))
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
