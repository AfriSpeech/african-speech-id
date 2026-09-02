"""Fetch YouVersion Bible audio, capped per language by the mixed-source plan.

Budget is per language across ALL sources, not per source: a language that already has an
hour of GRN takes one hour here and one from JW, so it ends with three hours in three
different voices rather than three hours of one narrator. Single-narrator data is what made
the in-domain numbers untrustworthy in the first place.

Chapters are picked spread across the inventory rather than sequentially -- Genesis 1-20
is one translator warming up; a spread samples the whole recording.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import requests
import soundfile as sf

VERSION_API = "https://nodejs.bible.com/api/bible/version/3.1"
CHAPTER_API = "https://nodejs.bible.com/api/bible/chapter/3.1"
UA = "african-speech-id research (+https://github.com/GhanaNLP/african-speech-id)"


def api(sess, url, params, tries=3):
    for a in range(tries):
        try:
            r = sess.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2 ** a)
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/fetch_plan.json")
    ap.add_argument("--versions", default="data/youversion_africa_versions_enriched.csv")
    ap.add_argument("--out", default="data/src_youversion")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())["yv"]
    by_iso = defaultdict(list)
    with open(args.versions, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("audio")).strip().lower() == "true":
                by_iso[r["lang_code"]].append(int(r["version_id"]))

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session(); sess.headers["User-Agent"] = UA
    rng = random.Random(args.seed)
    index = {}

    for n, (iso, budget_h) in enumerate(sorted(plan.items()), 1):
        dest = out / iso
        marker = dest / "done.json"
        if marker.exists():
            print(f"  [{n}/{len(plan)}] {iso}: already done", flush=True)
            continue
        versions = by_iso.get(iso, [])
        if not versions:
            print(f"  [{n}/{len(plan)}] {iso}: no audio version listed", flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)

        got, budget_s, files = 0.0, budget_h * 3600, []
        for vid in versions:
            if got >= budget_s:
                break
            inv = api(sess, VERSION_API, {"id": vid})
            if not inv:
                continue
            chapters = []
            for b in inv.get("books", []):
                if not (b.get("audio") or b.get("audio_count", 0) > 0):
                    continue
                usfm = b.get("usfm")
                for c in b.get("chapters", []):
                    num = c.get("human") or c.get("canonical")
                    if usfm and num and str(num).isdigit():
                        chapters.append((usfm, int(num)))
            if not chapters:
                continue
            rng.shuffle(chapters)                       # spread across the whole Bible
            for usfm, ch in chapters:
                if got >= budget_s:
                    break
                d = api(sess, CHAPTER_API, {"id": vid, "reference": f"{usfm}.{ch}"})
                if not d:
                    continue
                au = (d.get("audio") or [{}])[0].get("download_urls", {}).get("format_mp3_32k")
                if not au:
                    continue
                if au.startswith("//"):
                    au = "https:" + au
                try:
                    raw = sess.get(au, timeout=180).content
                    w, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
                except Exception:
                    continue
                secs = len(w) / sr
                if secs < 5:
                    continue
                fn = dest / f"{vid}_{usfm}_{ch}.mp3"
                fn.write_bytes(raw)
                files.append({"file": fn.name, "seconds": round(secs, 1), "version": vid})
                got += secs
                time.sleep(0.3)                          # be a polite guest

        marker.write_text(json.dumps({"iso": iso, "hours": round(got / 3600, 3),
                                      "files": files}, indent=1))
        index[iso] = round(got / 3600, 3)
        print(f"  [{n}/{len(plan)}] {iso}: {got/3600:.2f} h in {len(files)} chapters",
              flush=True)

    Path("data/youversion_index.json").write_text(json.dumps(index, indent=1))
    print(f"done: {len(index)} languages, {sum(index.values()):.0f} h")


if __name__ == "__main__":
    main()
