"""Download the parquet-based halves of the test panel: Waxal and the omniASR test split.

The panel is tiered by how independent each source really is, and results should be quoted
per tier rather than pooled:

  A  FLEURS, Waxal      different speakers, different collection, never seen
  A+ WikiTongues        spontaneous conversation -- the only unscripted source
  B  JW Broadcasting    separate recordings from the JW audio we train on, same publisher
  C  omniASR test       held out of ASR training, but the same collection protocol, so the
                        recogniser reads it more cleanly than wild audio

Only the test splits are taken, and only for languages our head can name.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

SOURCES = {
    "waxal": {"repo": "google/WaxalNLP", "split": "test", "tier": "A"},
    "omni":  {"repo": "facebook/omnilingual-asr-corpus", "split": "test", "tier": "C"},
}


def parquet_urls(repo: str, config: str, split: str):
    u = (f"https://datasets-server.huggingface.co/parquet?dataset="
         f"{urllib.parse.quote(repo, safe='')}&config={urllib.parse.quote(config)}")
    try:
        with urllib.request.urlopen(u, timeout=90) as r:
            fs = json.load(r)["parquet_files"]
        return [f["url"] for f in fs if f["split"] == split]
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--isos", default="data/test_target_isos.json",
                    help="ISO codes our head can name")
    ap.add_argument("--out", default="data/test_raw")
    ap.add_argument("--max-shards", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    want = set(json.loads(Path(args.isos).read_text()))
    manifest = {}

    for name, spec in SOURCES.items():
        # list configs for this repo, keep those whose ISO we can name
        u = (f"https://datasets-server.huggingface.co/splits?dataset="
             f"{urllib.parse.quote(spec['repo'], safe='')}")
        with urllib.request.urlopen(u, timeout=120) as r:
            splits = json.load(r)["splits"]
        cfgs = sorted({s["config"] for s in splits if s["split"] == spec["split"]})
        keep = []
        for c in cfgs:
            iso = c.split("_")[0] if name == "omni" else c.rsplit("_", 1)[0]
            if iso in want:
                keep.append((c, iso))
        print(f"  {name}: {len(cfgs)} configs with a {spec['split']} split, "
              f"{len(keep)} we can name", flush=True)

        dest = Path(args.out) / name
        dest.mkdir(parents=True, exist_ok=True)

        def get(item):
            cfg, iso = item
            urls = parquet_urls(spec["repo"], cfg, spec["split"])[:args.max_shards]
            got = []
            for i, url in enumerate(urls):
                try:
                    # the parquet endpoint URLs are Hub paths; hf_hub_download resumes
                    rel = url.split("/resolve/")[-1].split("/", 1)[-1]
                    p = hf_hub_download(spec["repo"], rel, repo_type="dataset",
                                        revision="refs/convert/parquet",
                                        cache_dir=str(dest / "_hf"))
                    got.append(p)
                except Exception as exc:
                    print(f"    {cfg}: {type(exc).__name__}", flush=True)
            return cfg, iso, got

        with cf.ThreadPoolExecutor(args.workers) as ex:
            for cfg, iso, got in ex.map(get, keep):
                if got:
                    manifest[f"{name}:{cfg}"] = {"source": name, "tier": spec["tier"],
                                                 "iso": iso, "files": got}
                    print(f"    {cfg} -> {iso}", flush=True)

    Path("data/test_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {len(manifest)} entries")


if __name__ == "__main__":
    import urllib.parse
    main()
