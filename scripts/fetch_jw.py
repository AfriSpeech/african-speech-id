"""Fetch JW audio, capped per language by the mixed-source plan.

The scraper that produced african-speech-public_v1 pulled only magazine audio (wp, g). That
misses most of what is there: for Twi the Bible alone is 82 h against ~50 h of magazines,
and there are eight other publications besides. This walks the whole publication list.

Taking tracks from several publications rather than one also spreads the voice talent, which
is the point of the mixed-source budget -- more speakers per language, not just more hours.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import requests

API = "https://b.jw-cdn.org/apis/pub-media/GETPUBMEDIALINKS"
# Bible first: it is the largest and most consistent. Then the study publications, which
# bring different readers and a less formal register.
PUBS = ["nwt", "lff", "jy", "bt", "rr", "bh", "lfb", "kr", "ia", "cl", "ll", "mwb", "wp", "g"]
UA = "african-speech-id research (+https://github.com/GhanaNLP/african-speech-id)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="data/fetch_plan.json")
    ap.add_argument("--langs", default="data/jw_langs_africa.csv")
    ap.add_argument("--out", default="data/src_jw")
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())["jw"]
    loc = {}
    with open(args.langs, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            loc.setdefault(r["iso639_3"], r["wtlocale"])

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sess = requests.Session(); sess.headers["User-Agent"] = UA
    index = {}

    for n, (iso, budget_h) in enumerate(sorted(plan.items()), 1):
        wt = loc.get(iso)
        dest = out / iso
        marker = dest / "done.json"
        if marker.exists():
            print(f"  [{n}/{len(plan)}] {iso}: already done", flush=True)
            continue
        if not wt:
            print(f"  [{n}/{len(plan)}] {iso}: no wtlocale", flush=True)
            continue
        dest.mkdir(parents=True, exist_ok=True)

        got, budget_s, files = 0.0, budget_h * 3600, []
        # round-robin one track per publication before taking a second from any, so a
        # language ends up with several voices rather than one long Bible reading
        pools = {}
        for pub in PUBS:
            try:
                d = sess.get(API, params={"pub": pub, "langwritten": wt,
                                          "fileformat": "MP3", "output": "json"},
                             timeout=60).json()
                tr = d.get("files", {}).get(wt, {}).get("MP3", [])
                tr = [t for t in tr if t.get("file", {}).get("url", "").endswith(".mp3")
                      and t.get("duration", 0) > 20]
                if tr:
                    pools[pub] = tr
            except Exception:
                pass
            time.sleep(0.2)

        cursor = {p: 0 for p in pools}
        while got < budget_s and any(cursor[p] < len(pools[p]) for p in pools):
            for pub in list(pools):
                if got >= budget_s or cursor[pub] >= len(pools[pub]):
                    continue
                t = pools[pub][cursor[pub]]; cursor[pub] += 1
                url = t["file"]["url"]
                try:
                    raw = sess.get(url, timeout=240).content
                except Exception:
                    continue
                fn = dest / f"{pub}_{cursor[pub]:04d}.mp3"
                fn.write_bytes(raw)
                secs = float(t.get("duration", 0))
                files.append({"file": fn.name, "seconds": round(secs, 1), "pub": pub})
                got += secs
                time.sleep(0.2)

        marker.write_text(json.dumps({"iso": iso, "hours": round(got / 3600, 3),
                                      "pubs": sorted(pools), "files": files}, indent=1))
        index[iso] = round(got / 3600, 3)
        print(f"  [{n}/{len(plan)}] {iso}: {got/3600:.2f} h from {len(pools)} publications",
              flush=True)

    Path("data/jw_index.json").write_text(json.dumps(index, indent=1))
    print(f"done: {len(index)} languages, {sum(index.values()):.0f} h")


if __name__ == "__main__":
    main()
