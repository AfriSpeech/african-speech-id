"""Download FLEURS validation parquet for the African languages that overlap GRN.

FLEURS is the out-of-domain test: different speakers, different recording chain, and
read-sentence content rather than religious narration. Since GRN has one narrator per
language, a GRN-internal split cannot tell us anything about generalisation and this can.

Validation rather than test only to halve the transfer -- we never train on FLEURS, so
both are equally unseen, and this box gets about 3 MB/s per connection to the Hub.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

DATASET = "google/fleurs"
REVISION = "refs/convert/parquet"

# FLEURS config -> the GRN language name it should be scored against
OVERLAP = {
    "af_za": "Afrikaans", "am_et": "Amharic", "ff_sn": "Fulfulde", "ha_ng": "Hausa",
    "kam_ke": "Kamba", "lg_ug": "Ganda", "luo_ke": "Luo", "nso_za": "Pedi",
    "ny_mw": "Nyanja", "om_et": "Oromo", "sn_zw": "Shona", "so_so": "Somali",
    "sw_ke": "Swahili", "umb_ao": "Umbundu", "wol_sn": "Wolof", "wo_sn": "Wolof",
    "xh_za": "Xhosa", "yo_ng": "Yoruba", "zu_za": "Zulu",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="validation")
    ap.add_argument("--cache", default="data/hf_fleurs")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    cfgs = sorted(set(OVERLAP) - {"wol_sn"})     # wol_sn is not a real FLEURS config

    def get(cfg):
        for name in (f"{cfg}/{args.split}/0000.parquet",
                     f"{cfg}/{args.split}/0001.parquet"):
            try:
                p = hf_hub_download(DATASET, name, repo_type="dataset",
                                    revision=REVISION, cache_dir=args.cache)
                print(f"  {cfg}", flush=True)
                return cfg, p
            except Exception as exc:
                last = f"{type(exc).__name__}: {str(exc)[:80]}"
        print(f"  FAILED {cfg}: {last}", flush=True)
        return cfg, None

    with cf.ThreadPoolExecutor(args.workers) as ex:
        got = [r for r in ex.map(get, cfgs) if r[1]]

    index = {cfg: {"path": p, "language": OVERLAP[cfg]} for cfg, p in got}
    Path("data/fleurs_index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)}/{len(cfgs)} configs -> data/fleurs_index.json")


if __name__ == "__main__":
    main()
