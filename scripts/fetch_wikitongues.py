"""Download WikiTongues audio for the languages our head can express.

WikiTongues is the only source in the test panel that is spontaneous speech: people talking
about their lives on a phone camera, several speakers per clip, occasional code-switching.
Everything else we score against is read -- religious narration, prompted sentences, news
copy -- so this is the one set that says how the model behaves on real conversation.

Labels come from the video titles ("Lucy speaking Igbo | Igboid | Nigeria | Wikitongues");
the HF mirror of this corpus ships no language labels at all.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CHANNEL_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/128.0 Safari/537.36"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="data/wikitongues_langs.json")
    ap.add_argument("--map", default="data/wt_to_class.json")
    ap.add_argument("--out", default="data/test_raw/wikitongues")
    ap.add_argument("--max-per-lang", type=int, default=3)
    ap.add_argument("--yt-dlp", default="./.venv/bin/yt-dlp")
    args = ap.parse_args()

    langs = json.loads(Path(args.langs).read_text())
    keep = json.loads(Path(args.map).read_text())      # wikitongues name -> our class
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    index, done, failed = {}, 0, 0
    for name, cls in sorted(keep.items()):
        for n, vid in enumerate(langs.get(name, [])[:args.max_per_lang]):
            dest = out / f"{vid}.wav"
            if not dest.exists():
                cmd = [args.yt_dlp, "-q", "--no-warnings", "--no-playlist",
                       "-f", "bestaudio/best", "-x", "--audio-format", "wav",
                       "--postprocessor-args", "-ar 16000 -ac 1",
                       "--user-agent", CHANNEL_UA,
                       "-o", str(out / "%(id)s.%(ext)s"),
                       f"https://www.youtube.com/watch?v={vid}"]
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
                if not dest.exists():
                    err = (r.stderr or "").strip().splitlines()
                    print(f"  FAILED {name} {vid}: {err[-1][:90] if err else '?'}", flush=True)
                    failed += 1
                    continue
            index[vid] = {"language": name, "our_class": cls}
            done += 1
            print(f"  [{done}] {name:<22} -> {cls:<22} {vid}", flush=True)

    Path("data/wikitongues_test_index.json").write_text(
        json.dumps(index, indent=1, ensure_ascii=False))
    print(f"downloaded {done} clips over {len({v['our_class'] for v in index.values()})} "
          f"classes, {failed} failed")


if __name__ == "__main__":
    main()
