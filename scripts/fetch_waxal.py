"""Fetch Waxal test splits straight from the repo.

The datasets-server parquet conversion for google/WaxalNLP is failing on the Hub side
("Commit 0/1 could not be created"), so the usual /parquet endpoint returns nothing. The
repo ships its own parquet under data/ASR/<iso>/, with two naming conventions, so this
lists the repo and filters rather than guessing paths.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

REPO = "google/WaxalNLP"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isos", default="data/test_target_isos.json")
    ap.add_argument("--out", default="data/test_raw/waxal")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    want = set(json.loads(Path(args.isos).read_text()))
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    by_iso = defaultdict(list)
    for f in files:
        m = re.match(r"data/[A-Z]+/([a-z]{2,3})/.*test.*\.parquet$", f)
        if m and m.group(1) in want:
            by_iso[m.group(1)].append(f)
    print(f"  {len(by_iso)} languages with a test shard we can name", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def get(item):
        iso, paths = item
        got = []
        for p in sorted(paths)[:2]:
            try:
                got.append(hf_hub_download(REPO, p, repo_type="dataset",
                                           cache_dir=str(out / "_hf")))
            except Exception as exc:
                print(f"    {iso}: {type(exc).__name__}: {str(exc)[:70]}", flush=True)
        return iso, got

    manifest = {}
    with cf.ThreadPoolExecutor(args.workers) as ex:
        for iso, got in ex.map(get, sorted(by_iso.items())):
            if got:
                manifest[f"waxal:{iso}"] = {"source": "waxal", "tier": "A",
                                            "iso": iso, "files": got}
                print(f"    {iso}: {len(got)} shard(s)", flush=True)

    p = Path("data/test_manifest_waxal.json")
    p.write_text(json.dumps(manifest, indent=1))
    print(f"waxal manifest: {len(manifest)} languages")


if __name__ == "__main__":
    main()
