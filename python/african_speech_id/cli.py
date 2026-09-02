"""``african-speech-id`` command line entry point.

    african-speech-id "obiara na enyi nyɛden dɛ ɔbɔbɔ no nkenyan"
    african-speech-id --file transcripts.txt --top 3
    african-speech-id --top 3 < transcripts.txt
"""
from __future__ import annotations

import argparse
import sys

from african_speech_id.model import DEFAULT_REPO, AfricanSpeechId


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="african-speech-id",
        description="Identify the language of a speech transcript.")
    ap.add_argument("text", nargs="*", help="transcript; omit to read stdin")
    ap.add_argument("--model", default=DEFAULT_REPO,
                    help="local model directory or Hugging Face repo id")
    ap.add_argument("--file", help="read one phoneme string per line from this file")
    ap.add_argument("--top", type=int, default=1, help="show the N most likely languages")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fp16", action="store_true", help="prefer the half-precision head")
    args = ap.parse_args(argv)

    lid = AfricanSpeechId.load(args.model, fp16=args.fp16, num_threads=args.threads)

    if args.text:
        lines = [" ".join(args.text)]
    elif args.file:
        with open(args.file, encoding="utf-8") as fh:
            lines = [l.rstrip("\n") for l in fh]
    else:
        lines = [l.rstrip("\n") for l in sys.stdin]

    for line in lines:
        if not line.strip():
            continue
        if args.top <= 1:
            r = lid.classify(line)
            print("unknown\t0.000" if r is None else f"{r.language}\t{r.confidence:.3f}")
            continue
        post = lid.posterior(line)
        if post is None:
            print("unknown\t0.000")
            continue
        top = sorted(post.items(), key=lambda kv: -kv[1])[: args.top]
        print("\t".join(f"{k}\t{v:.3f}" for k, v in top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
