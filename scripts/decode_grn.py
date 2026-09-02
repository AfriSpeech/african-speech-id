"""Transcribe downloaded GRN subsets with base omniASR, one parquet of text per subset.

The head that follows classifies text, so the recogniser only has to be *consistent* within
a language, not correct: it emits plausible output even for languages it plainly does not
know, and consistent-but-wrong still separates languages.

fp32 on CUDA, not int8: quantised operators have no CUDA kernels, so int8 lands on the CPU
node by node and runs at 7x against fp32's 111x. This box's CPU is usually contended and
its GPU is not. Note the mismatch -- the head is served on int8 CPU -- which cost about 1.3
points when measured on the Ghanaian model. Decode int8 for a shipping run.

One output parquet per subset, skipped if present, so a killed run resumes for free.
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import sherpa_onnx
import soundfile as sf

MODEL = ("/mnt/volume_d2wey28/projects/african-speech-id/models/"
         "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/shard_index.json")
    ap.add_argument("--out-dir", default="data/grn_parts")
    ap.add_argument("--cap-seconds", type=float, default=3600.0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--provider", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    index = json.loads(Path(args.index).read_text())
    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=f"{MODEL}/model.onnx", tokens=f"{MODEL}/tokens.txt",
        num_threads=args.threads, provider=args.provider)
    print(f"{len(index)} subsets, provider={args.provider}", flush=True)

    for n, (subset, files) in enumerate(sorted(index.items()), 1):
        dest = out / f"{subset}.parquet"
        if dest.exists():
            print(f"{n}/{len(index)} {subset}: done", flush=True)
            continue

        t0 = time.time()
        rows, secs = [], 0.0
        batch, meta = [], []

        def flush():
            if not batch:
                return
            streams = []
            for w, sr in batch:
                s = rec.create_stream(); s.accept_waveform(sr, w); streams.append(s)
            rec.decode_streams(streams)
            for (idx, lang, code, dur), s in zip(meta, streams):
                rows.append((subset, lang, code, idx, round(dur, 2), s.result.text))
            batch.clear(); meta.clear()

        try:
            for path in files:
                pf = pq.ParquetFile(path)
                for rg in range(pf.metadata.num_row_groups):
                    tbl = pf.read_row_group(rg)
                    cols = tbl.to_pydict()
                    for i in range(tbl.num_rows):
                        a = cols["audio"][i]
                        raw = a["bytes"] if isinstance(a, dict) else a
                        if not raw:
                            continue
                        w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                        if w.ndim > 1:
                            w = w.mean(axis=1)
                        dur = len(w) / sr
                        if dur < 1.0:
                            continue
                        batch.append((w, sr))
                        meta.append((i, cols.get("language", [""])[i],
                                     cols.get("code", [""])[i], dur))
                        secs += dur
                        if len(batch) >= args.batch:
                            flush()
                        if secs >= args.cap_seconds:
                            break
                    if secs >= args.cap_seconds:
                        break
                if secs >= args.cap_seconds:
                    break
            flush()
        except Exception as exc:            # one bad shard must not end the run
            print(f"{n}/{len(index)} {subset}: FAILED {type(exc).__name__}: "
                  f"{str(exc)[:120]}", flush=True)
            continue

        if not rows:
            print(f"{n}/{len(index)} {subset}: no rows", flush=True)
            continue

        c = list(zip(*rows))
        pq.write_table(pa.table({
            "subset": pa.array(c[0]), "language": pa.array(c[1]), "code": pa.array(c[2]),
            "clip": pa.array(c[3], pa.int32()), "seconds": pa.array(c[4], pa.float32()),
            "text": pa.array(c[5]),
        }), dest.as_posix())

        dt = time.time() - t0
        empty = sum(1 for r in rows if not r[5].strip())
        chars = int(np.mean([len(r[5]) for r in rows]))
        print(f"{n}/{len(index)} {subset}: {len(rows)} clips, {secs/60:.1f} min, "
              f"{chars} chars mean, {empty} empty, {secs/max(dt,1e-9):.0f}x RT", flush=True)


if __name__ == "__main__":
    main()
