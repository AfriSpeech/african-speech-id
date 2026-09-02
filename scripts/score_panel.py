"""Score the head against the tiered test panel.

Results are reported per tier, never pooled, because the tiers are not equally honest:

  A  FLEURS, Waxal    different speakers, different collection, never seen by anything
  A+ WikiTongues      spontaneous conversation -- the only unscripted source we have
  C  omniASR test     held out of ASR training, but the same collection protocol, so the
                      recogniser reads it more cleanly than it reads the wild

A single pooled number would let the easy tiers carry the hard ones.

Truth is an ISO 639-3 code. The label space is mixed -- 1034 classes are ISO codes, 352 are
display names for GRN subsets that never carried one -- so a prediction counts as correct if
the predicted class IS the truth code or is a name known to belong to it.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq
import sherpa_onnx
import soundfile as sf

ASR = ("/mnt/volume_d2wey28/projects/african-speech-id/models/"
       "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12")

FLEURS_ISO = {
    "af_za": "afr", "am_et": "amh", "ff_sn": "fuc", "ha_ng": "hau", "kam_ke": "kam",
    "lg_ug": "lug", "luo_ke": "luo", "nso_za": "nso", "ny_mw": "nya", "om_et": "gaz",
    "sn_zw": "sna", "so_so": "som", "sw_ke": "swh", "umb_ao": "umb", "wo_sn": "wol",
    "xh_za": "xho", "yo_ng": "yor", "zu_za": "zul",
}


MAX_SECONDS = 30.0      # self-attention is quadratic in length; a few-minute clip in a
                        # batch asks onnxruntime for tens of GB and kills the run


def resample(w, sr, target=16000):
    if sr != target:
        n = round(len(w) * target / sr)
        w = np.interp(np.linspace(0, len(w) - 1, n), np.arange(len(w)), w)
    return np.asarray(w, dtype=np.float32)


def load_alias(path="data/mixed_classes.json"):
    """class -> the ISO codes it may legitimately answer for."""
    alias = defaultdict(set)
    for f in glob.glob("data/grn_parts/*.parquet"):
        t = pq.read_table(f, columns=["subset", "language"]).to_pydict()
        if not t["subset"]:
            continue
        sub, disp = t["subset"][0], t["language"][0]
        pre = sub.split("_")[0]
        cls = pre if re.fullmatch(r"[a-z]{3}", pre) else disp.split(":")[0].strip()
        if re.fullmatch(r"[a-z]{3}", pre):
            alias[cls].add(pre)
    return alias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="out/mixed1386/model.joblib")
    ap.add_argument("--asr", default=ASR,
                    help="recogniser to transcribe the panel with; must match "
                         "the build the head was trained on")
    ap.add_argument("--per-lang", type=int, default=60)
    ap.add_argument("--min-seconds", type=float, default=3.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default="out/panel_scores.json")
    ap.add_argument("--provider", default="cuda",
                    help="cpu for the int8 build: quantised ops have no CUDA "
                         "kernels and fall back node by node")
    ap.add_argument("--mms", action="store_true",
                    help="also score facebook/mms-lid-4017 on the same clips")
    args = ap.parse_args()

    b = joblib.load(args.head)
    vec, clf = b["vec"], b["clf"]
    known = {str(c) for c in clf.classes_}
    alias = load_alias()

    def correct(pred: str, truth_iso: str) -> bool:
        p = str(pred)
        return p == truth_iso or truth_iso in alias.get(p, ())

    # the int8 builds ship model.int8.onnx, the fp32 ones model.onnx
    asr_model = next((f"{args.asr}/{n}" for n in ("model.onnx", "model.int8.onnx")
                      if Path(f"{args.asr}/{n}").exists()), f"{args.asr}/model.onnx")
    rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=asr_model, tokens=f"{args.asr}/tokens.txt",
        num_threads=8, provider=args.provider)

    mms = mms_fe = mms_labels = None
    if args.mms:
        import torch
        from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
        mms_fe = AutoFeatureExtractor.from_pretrained("facebook/mms-lid-4017")
        mms = Wav2Vec2ForSequenceClassification.from_pretrained(
            "facebook/mms-lid-4017").eval().to("cuda")
        # id2label keys come back as ints; indexing with str(i) KeyErrors silently
        mms_labels = {int(k): v for k, v in mms.config.id2label.items()}

    def mms_predict(waves):
        """MMS reads audio directly -- no transcript, so it sees exactly the same clips."""
        if mms is None:
            return []
        cap = round(MAX_SECONDS * 16000)
        out = []
        with torch.no_grad():
            for i in range(0, len(waves), 4):
                chunk = [w[:cap] for w in waves[i:i + 4]]
                inp = mms_fe(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
                lg = mms(**{k: v.to("cuda") for k, v in inp.items()}).logits
                out += [mms_labels[j] for j in lg.argmax(dim=-1).tolist()]
        return out

    def decode(waves):
        # Cap here, not in resample: resample also feeds the WikiTongues windowing loop,
        # and truncating there gave one 20s window per clip instead of ten.
        cap = round(MAX_SECONDS * 16000)
        waves = sorted((w[:cap] for w in waves), key=len)
        texts = []
        for i in range(0, len(waves), args.batch):
            streams = []
            for w in waves[i:i + args.batch]:
                s = rec.create_stream(); s.accept_waveform(16000, w); streams.append(s)
            rec.decode_streams(streams)
            texts += [s.result.text for s in streams]
        return texts

    # ---- assemble the panel: (tier, source, truth_iso, [waveforms]) ----
    items = []

    for cfg, iso in FLEURS_ISO.items():
        fs = glob.glob(f"data/hf_fleurs/**/{cfg}/validation/*.parquet", recursive=True)
        if fs:
            items.append(("A", "fleurs", cfg, iso, fs))
    for iso_dir in sorted(glob.glob("data/test_raw/waxal/_hf/**/data/*/*/*test*.parquet",
                                    recursive=True)):
        m = re.search(r"/data/[A-Z]+/([a-z]{2,3})/", iso_dir)
        if m:
            items.append(("A", "waxal", m.group(1), m.group(1), [iso_dir]))
    for f in sorted(glob.glob("data/test_raw/omni/**/*.parquet", recursive=True)):
        m = re.search(r"/([a-z]{3})_[A-Z][a-z]{3}/", f)
        if m:
            items.append(("C", "omni", m.group(1), m.group(1), [f]))

    grouped = defaultdict(lambda: defaultdict(list))
    for tier, src, key, iso, files in items:
        grouped[(tier, src)][iso].extend(files)

    results = []
    for (tier, src), by_iso in sorted(grouped.items()):
        for iso, files in sorted(by_iso.items()):
            if not any(correct(c, iso) for c in known):
                continue
            waves, t0 = [], time.time()
            for f in files:
                try:
                    pf = pq.ParquetFile(f)
                except Exception:
                    continue
                for rg in range(pf.metadata.num_row_groups):
                    for row in pf.read_row_group(rg).to_pylist():
                        if len(waves) >= args.per_lang:
                            break
                        a = row.get("audio")
                        raw = a.get("bytes") if isinstance(a, dict) else a
                        if not raw:
                            continue
                        try:
                            w, sr = sf.read(io.BytesIO(raw), dtype="float32",
                                            always_2d=False)
                        except Exception:
                            continue
                        if w.ndim > 1:
                            w = w.mean(axis=1)
                        if len(w) / sr < args.min_seconds:
                            continue
                        waves.append(resample(w, sr))
                    if len(waves) >= args.per_lang:
                        break
                if len(waves) >= args.per_lang:
                    break
            if not waves:
                continue
            texts = [t for t in decode(waves) if t.strip()]
            if not texts:
                continue
            pred = clf.predict(vec.transform(texts))
            hit = sum(1 for p in pred if correct(p, iso))
            mp = mms_predict(waves)
            mhit = sum(1 for p in mp if p == iso)
            results.append({"tier": tier, "source": src, "iso": iso, "n": len(texts),
                            "correct": hit, "acc": round(hit / len(texts), 4),
                            "mms_n": len(mp), "mms_correct": mhit,
                            "mms_acc": round(mhit / len(mp), 4) if mp else None})
            mtxt = (f"  mms {mhit/len(mp):.3f}" if mp else "")
            print(f"  [{tier}] {src:<12} {iso:<5} ours {hit/len(texts):.3f}{mtxt}"
                  f"   n={len(texts)}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- WikiTongues: plain wav on disk, labelled by video id ----
    wt_idx = json.loads(Path("data/wikitongues_test_index.json").read_text())
    wt = defaultdict(list)
    for vid, meta in wt_idx.items():
        p = Path(f"data/test_raw/wikitongues/{vid}.wav")
        if p.exists():
            wt[meta["our_class"]].append(p)
    for cls, paths in sorted(wt.items()):
        if cls not in known:
            continue
        waves = []
        for p in paths:
            w, sr = sf.read(p, dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=1)
            w = resample(w, sr)
            for i in range(0, max(len(w) - 16000 * 10, 1), 16000 * 20):
                seg = w[i:i + 16000 * 20]
                if len(seg) >= 16000 * args.min_seconds:
                    waves.append(seg)
        waves = waves[:args.per_lang]
        if not waves:
            continue
        texts = [t for t in decode(waves) if t.strip()]
        if not texts:
            continue
        pred = clf.predict(vec.transform(texts))
        hit = sum(1 for p in pred if str(p) == cls)
        mp = mms_predict(waves)
        wt_iso = sorted(alias.get(cls, [cls]))[0]
        mhit = sum(1 for p in mp if p == wt_iso)
        results.append({"tier": "A+", "source": "wikitongues", "iso": cls,
                        "n": len(texts), "correct": hit,
                        "acc": round(hit / len(texts), 4),
                        "mms_n": len(mp), "mms_correct": mhit,
                        "mms_acc": round(mhit / len(mp), 4) if mp else None})
        mtxt = (f"  mms {mhit/len(mp):.3f}" if mp else "")
        print(f"  [A+] wikitongues  {cls:<18} ours {hit/len(texts):.3f}{mtxt}"
              f"   n={len(texts)}", flush=True)

    Path(args.out).write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print("\n  tier   source        langs  clips    ours   macro     mms")
    by = defaultdict(list)
    for r in results:
        by[(r["tier"], r["source"])].append(r)
        by[(r["tier"], "ALL")].append(r)
    for k in sorted(by):
        rs = by[k]
        n = sum(r["n"] for r in rs); c = sum(r["correct"] for r in rs)
        mn = sum(r.get("mms_n") or 0 for r in rs)
        mc = sum(r.get("mms_correct") or 0 for r in rs)
        mtxt = f"  {mc/mn:.4f}" if mn else "       -"
        print(f"  {k[0]:<6} {k[1]:<13} {len(rs):>4} {n:>6}  {c/max(n,1):.4f}  "
              f"{np.mean([r['acc'] for r in rs]):.4f}{mtxt}")


if __name__ == "__main__":
    main()
