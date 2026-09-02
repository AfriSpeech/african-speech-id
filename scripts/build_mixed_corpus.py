"""Merge GRN, YouVersion and JW transcripts into one training corpus.

Label space: the ISO 639-3 code where the GRN subset name carries one (820 of them do), the
display name with its dialect suffix stripped otherwise. That is what lets the three sources
join -- YouVersion and JW are keyed on ISO, GRN on subset names -- and it merges GRN dialect
subsets of one language at the same time.

Sampling is per language and deliberately spread across sources. A language with GRN,
YouVersion and JW audio takes a third from each rather than filling up from whichever is
largest: GRN gives one narrator per language, and single-narrator training is what made the
in-domain numbers untrustworthy in the first place. Mixing narrators is the point of having
fetched the other two at all.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grn", default="data/grn_parts")
    ap.add_argument("--new", default="data/src_parts")
    ap.add_argument("--out", default="data/mixed_corpus.parquet")
    ap.add_argument("--max-clips", type=int, default=900,
                    help="per language, across all sources")
    ap.add_argument("--min-clips", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # class -> source -> [(text, seconds)]
    pool: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    names: dict[str, str] = {}

    for f in sorted(Path(args.grn).glob("*.parquet")):
        t = pq.read_table(f, columns=["subset", "language", "text", "seconds"]).to_pydict()
        if not t["subset"]:
            continue
        sub, disp = t["subset"][0], t["language"][0]
        pre = sub.split("_")[0]
        cls = pre if re.fullmatch(r"[a-z]{3}", pre) else disp.split(":")[0].strip()
        names.setdefault(cls, disp.split(":")[0].strip())
        for x, s in zip(t["text"], t["seconds"]):
            if x and x.strip():
                pool[cls]["grn"].append((x, float(s)))

    for f in sorted(Path(args.new).glob("*.parquet")):
        t = pq.read_table(f, columns=["iso", "source", "text", "seconds"]).to_pydict()
        if not t["iso"]:
            continue
        cls = t["iso"][0]
        names.setdefault(cls, cls)
        for x, s, src in zip(t["text"], t["seconds"], t["source"]):
            if x and x.strip():
                pool[cls][src].append((x, float(s)))

    rng = random.Random(args.seed)
    rows, dropped, stats = [], [], defaultdict(int)
    for cls, by_src in sorted(pool.items()):
        total = sum(len(v) for v in by_src.values())
        if total < args.min_clips:
            dropped.append((cls, total))
            continue
        # even split across whatever sources exist, spending any shortfall on the others
        srcs = sorted(by_src)
        take, budget = {}, args.max_clips
        for i, s in enumerate(srcs):
            share = budget // (len(srcs) - i)
            take[s] = min(share, len(by_src[s]))
            budget -= take[s]
        picked = []
        for s in srcs:
            items = by_src[s]
            picked += [(s, *it) for it in
                       (rng.sample(items, take[s]) if take[s] < len(items) else items)]
            stats[s] += take[s]
        # Shuffle before numbering: picked is grouped by source, so a contiguous split on
        # id would hold out whichever source happens to sit last rather than a fair sample
        # of the class. The honest cross-source number comes from the test panel, not here.
        rng.shuffle(picked)
        for n, (s, text, secs) in enumerate(picked):
            rows.append((n, cls, text, secs, s, names[cls]))

    c = list(zip(*rows))
    pq.write_table(pa.table({
        "id": pa.array(c[0], pa.int32()),
        "language": pa.array(c[1]),
        "text": pa.array(c[2]),
        "duration": pa.array(c[3], pa.float32()),
        "source": pa.array(c[4]),
        "lang_name": pa.array(c[5]),
    }), args.out)

    per = defaultdict(set)
    for _, cls, _, _, s, _ in rows:
        per[cls].add(s)
    multi = sum(1 for v in per.values() if len(v) > 1)
    print(f"{len(rows):,} clips, {len(per)} languages -> {args.out}")
    print(f"  clips by source: {dict(stats)}")
    print(f"  languages with >1 source (multi-narrator): {multi} "
          f"({100*multi/max(len(per),1):.0f}%)")
    print(f"  dropped under {args.min_clips} clips: {len(dropped)}")
    Path("data/mixed_classes.json").write_text(
        json.dumps({k: sorted(v) for k, v in per.items()}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
