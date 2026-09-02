"""Download the parquet shards for the chosen GRN subsets.

Shard names are not uniform -- some subsets use train-00000.parquet, others
train-00000-of-00001.parquet -- so paths come from the repo tree rather than a guessed
pattern. Downloads run in parallel because a single HTTP stream to the Hub tops out around
3 MB/s from this box, which is the bottleneck; the decode that follows is not.

Only as many shards as the per-subset audio cap needs, which for a one-hour cap is almost
always the first.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import time
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET = "AfriSpeech/grn-african-speech"
TREE = f"https://huggingface.co/api/datasets/{DATASET}/tree/main"


def shards(subset: str, tries: int = 5) -> list[tuple[str, int]]:
    """Resolve a subset's parquet paths, backing off through the Hub's rate limit.

    Nearly a thousand tree calls will be throttled without this, and a throttled call
    returns an error rather than an empty listing -- silently dropping half the corpus.
    """
    for attempt in range(tries):
        try:
            req = urllib.request.Request(f"{TREE}/{subset}")
            tok = os.environ.get("HF_TOKEN")
            if tok:
                req.add_header("Authorization", f"Bearer {tok}")
            with urllib.request.urlopen(req, timeout=90) as r:
                files = json.load(r)
            return sorted((f["path"], f.get("size", 0)) for f in files
                          if f["path"].endswith(".parquet"))
        except Exception:
            if attempt == tries - 1:
                print(f"  UNRESOLVED {subset}", flush=True)
                return []
            time.sleep(2 ** attempt)
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="pilot_subsets.json")
    ap.add_argument("--cache", default="data/hf")
    ap.add_argument("--max-shards", type=int, default=1,
                    help="per subset; one shard is far more than a one-hour cap needs")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    subsets = [s["subset"] for s in json.loads(Path(args.subsets).read_text())]

    with cf.ThreadPoolExecutor(args.workers) as ex:
        listing = dict(zip(subsets, ex.map(shards, subsets)))
    wanted = [(s, p) for s, fs in listing.items() for p, _ in fs[:args.max_shards]]
    total = sum(sz for s, fs in listing.items() for _, sz in fs[:args.max_shards])
    print(f"{len(wanted)} shards across {len(subsets)} subsets, {total/1e9:.1f} GB",
          flush=True)

    done = {"n": 0}

    def get(item):
        subset, path = item
        try:
            local = hf_hub_download(DATASET, path, repo_type="dataset", cache_dir=args.cache)
            done["n"] += 1
            print(f"  [{done['n']}/{len(wanted)}] {subset}", flush=True)
            return subset, local
        except Exception as exc:
            print(f"  FAILED {subset}: {type(exc).__name__}: {str(exc)[:100]}", flush=True)
            return subset, None

    with cf.ThreadPoolExecutor(args.workers) as ex:
        got = list(ex.map(get, wanted))

    index = {}
    for subset, local in got:
        if local:
            index.setdefault(subset, []).append(local)
    Path("data/shard_index.json").write_text(json.dumps(index, indent=1))
    print(f"downloaded {len(index)}/{len(subsets)} subsets -> data/shard_index.json")


if __name__ == "__main__":
    main()
