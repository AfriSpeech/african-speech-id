"""Transcribe the YouVersion and JW audio into per-language parquet.

These arrive as whole Bible chapters and magazine tracks -- minutes long, where GRN gives
15-second clips. They are cut into fixed windows before decoding, both because a CTC pass
over a ten-minute file is wasteful and because the head is trained and served on short
utterances; matching that here keeps the training text distribution honest.

fp32 on CUDA, as with GRN: int8 has no CUDA kernels and drops to 7x against ~370x here.
One parquet per language per source, skipped if present, so a kill costs one language.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sherpa_onnx
import soundfile as sf

MODEL = ("/mnt/volume_d2wey28/projects/african-speech-id/models/"
         "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12")


def windows(w: np.ndarray, sr: int, secs: float, hop: float):
    n, step = round(secs * sr), round(hop * sr)
    for i in range(0, max(len(w) - n // 2, 1), step):
        seg = w[i:i + n]
        if len(seg) >= sr:                       # skip a runt tail
            yield i / sr, seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="data/src_youversion or data/src_jw")
    ap.add_argument("--tag", required=True, help="youversion | jw")
    ap.add_argument("--out-dir", default="data/src_parts")
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--hop", type=float, default=20.0)
    ap.add_argument("--max-seconds", type=float, default=3600.0,
                    help="per language, a backstop on top of the fetch budget")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--provider", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    langs = sorted(p for p in Path(args.src).iterdir() if p.is_dir())
    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=f"{MODEL}/model.onnx", tokens=f"{MODEL}/tokens.txt",
        num_threads=2, provider=args.provider)
    print(f"{len(langs)} languages in {args.src}", flush=True)

    for n, d in enumerate(langs, 1):
        iso = d.name
        dest = out / f"{args.tag}__{iso}.parquet"
        if dest.exists():
            print(f"{n}/{len(langs)} {iso}: done", flush=True)
            continue
        files = sorted(f for f in d.iterdir() if f.suffix.lower() in (".mp3", ".wav"))
        if not files:
            continue

        t0, rows, secs = time.time(), [], 0.0
        batch, meta = [], []

        def flush():
            if not batch:
                return
            streams = []
            for seg, sr in batch:
                s = rec.create_stream(); s.accept_waveform(sr, seg); streams.append(s)
            rec.decode_streams(streams)
            for (fn, off, dur), s in zip(meta, streams):
                rows.append((iso, args.tag, fn, round(off, 1), round(dur, 2), s.result.text))
            batch.clear(); meta.clear()

        try:
            for f in files:
                if secs >= args.max_seconds:
                    break
                try:
                    w, sr = sf.read(f, dtype="float32", always_2d=False)
                except Exception:
                    continue
                if w.ndim > 1:
                    w = w.mean(axis=1)
                if sr != 16000:                  # omniASR expects 16 kHz
                    m = round(len(w) * 16000 / sr)
                    w = np.interp(np.linspace(0, len(w) - 1, m),
                                  np.arange(len(w)), w).astype(np.float32)
                    sr = 16000
                for off, seg in windows(w, sr, args.window, args.hop):
                    batch.append((seg, sr))
                    meta.append((f.name, off, len(seg) / sr))
                    secs += len(seg) / sr
                    if len(batch) >= args.batch:
                        flush()
                    if secs >= args.max_seconds:
                        break
            flush()
        except Exception as exc:
            print(f"{n}/{len(langs)} {iso}: FAILED {type(exc).__name__}: {str(exc)[:100]}",
                  flush=True)
            continue

        if not rows:
            print(f"{n}/{len(langs)} {iso}: no rows", flush=True)
            continue
        c = list(zip(*rows))
        pq.write_table(pa.table({
            "iso": pa.array(c[0]), "source": pa.array(c[1]), "file": pa.array(c[2]),
            "offset": pa.array(c[3], pa.float32()), "seconds": pa.array(c[4], pa.float32()),
            "text": pa.array(c[5]),
        }), dest.as_posix())
        dt = time.time() - t0
        empty = sum(1 for r in rows if not r[5].strip())
        print(f"{n}/{len(langs)} {iso}: {len(rows)} windows, {secs/60:.1f} min, "
              f"{empty} empty, {secs/max(dt,1e-9):.0f}x RT", flush=True)


if __name__ == "__main__":
    main()
