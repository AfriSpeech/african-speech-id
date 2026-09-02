---
license: cc-by-nc-4.0
pipeline_tag: audio-classification
tags:
  - language-identification
  - african-languages
  - speech
  - onnx
library_name: african-speech-id
---

# african-speech-id

CPU-friendly, fast language identification for 1,386 African languages.

The model uses a fast version of Omnilingual ASR to turn speech into text, and a
classification head that reads that text and names the language. Both run on CPU, so a
phone or a laptop is enough.

## What is in this repository

| path | what it is | size |
|---|---|---|
| `300m/` | the language-ID head | 265 MB |

The head classifies text and cannot read audio. The recogniser it reads from is
[omniASR 300M CTC](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models), fetched
separately by `AfricanSpeechId.download_recogniser()`.

## Speed

The reason to use this rather than a large audio classifier. Measured on one Xeon Platinum
8558 core, int8, on held-out audio:

| | throughput |
|---|---|
| classification head alone | 0.28 ms per call, 3,600 per second |
| full pipeline, 1 thread | 3.5x realtime |
| full pipeline, 4 threads | 9.3x realtime |
| facebook/mms-lid-4017, same machine | 1.4x realtime |

A ten-second clip is identified in under three seconds on a single core. MMS-LID needs no
recogniser, but it is 970 million parameters and 3.9 GB against this head's 265 MB, and it
is slower end to end even counting the recogniser this depends on.

## Results

Evaluated against [facebook/mms-lid-4017](https://huggingface.co/facebook/mms-lid-4017) on
identical clips, same 30-second cap, both systems scored in the label space each can
produce:

| test set | languages | clips | **this model** | MMS-LID-4017 |
|---|---|---|---|---|
| [Waxal](https://huggingface.co/datasets/google/WaxalNLP) | 28 | 1,675 | **0.610** | 0.604 |
| [omniASR corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) (test split) | 76 | 4,557 | **0.429** | 0.270 |
| **combined** | **104** | **6,232** | **0.478** | **0.360** |

Per language: ahead on 57, behind on 38, tied on 9.

MMS-LID covers 4,017 languages and is strong on well-resourced ones. This model covers 1,386
and is built for the tail underneath them, which is what the omniASR test set is made of.

**Read these numbers with two things in mind.** They come from two evaluation sets chosen
because they cover long-tail African languages; other sets exist and give different answers,
and on FLEURS — 17 major languages — MMS-LID scores higher than this model does. And 104 of
1,386 languages are evaluated here at all: the remaining 1,282 are trained but unmeasured
against any independent source.

## Usage

```python
import soundfile as sf
import sherpa_onnx
from african_speech_id import AfricanSpeechId

model, tokens = AfricanSpeechId.download_recogniser()
rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(model=model, tokens=tokens)
lid = AfricanSpeechId.load()

wav, sr = sf.read("clip.wav", dtype="float32")
s = rec.create_stream(); s.accept_waveform(sr, wav); rec.decode_stream(s)
print(lid.classify(s.result.text))
```

On device there is no Python: sherpa-onnx produces the transcript and the head runs in
onnxruntime through a C API, with Kotlin and Swift bindings. See
[the repository](https://github.com/AfriSpeech/african-speech-id).

## How much audio to give it

Five seconds minimum, ten for the best result. Accuracy climbs steeply with the amount of
speech and flattens after roughly ten seconds. Check that most of the audio is speech before
transcribing — silence contributes nothing and inflates the apparent duration.

## Training data

Three sources, budgeted **per language** rather than per source, at most three hours each:

| source | languages | clips |
|---|---|---|
| [GRN African Speech](https://huggingface.co/datasets/AfriSpeech/grn-african-speech) | 1,019 | 235,212 |
| YouVersion Bible audio | 427 | 76,202 |
| JW publication audio | 218 | 32,054 |

343,468 clips in total. The budget is spread across sources on purpose: **360 languages have
two or three different narrators** rather than one. Single-narrator data is the central risk
in this kind of corpus — an earlier experiment on acoustic embeddings reached 0.976 in domain
and 0.108 out of it, having learned voices rather than languages.

All three sources are religious recordings, so the training domain is narrow. That is the
main reason to expect worse performance on conversational speech than the figures above.

## How it is built

A linear classifier over character n-grams of the transcript, with tf-idf folded into the
ONNX graph. 50,000 features over 1,386 classes; a 300,000-feature version scored 0.4833
against 0.4779, which does not justify twelve times the size. Trained on 40-character
windows rather than whole transcripts, so training matches how short utterances are served.

## Limitations

**Closed set.** The head always names one of its 1,386 languages, including for speech in a
language it has never seen. The top-1 minus top-2 margin gives a rejection signal.

**Confidence values are small.** A softmax over 1,386 classes spreads thin and a confident
answer often reads about 0.03-0.05; use the margin, not the raw score. Labels are ISO 639-3
codes where the source corpus carried one and language names otherwise.

**Dialects are not separable.** Grebo-Chedepo against Grebo-Buah, Asante Twi against Twi:
the recogniser normalises toward one orthography per language and the distinction does not
survive. Dialect subsets are merged into their language.

**Most languages are unmeasured.** 104 of 1,386 have independent evaluation.

**Narrow domain.** All training audio is religious narration.

**Transcribed with the November 2025 omniASR build.** A February 2026 v2 exists and was not
used; it changes orthographic conventions — restored diacritics, different word
segmentation, Ethiopic script instead of Latin for some Ethiopian languages — so results
here are a floor rather than a ceiling.

## Licence

Code Apache-2.0. Model and data follow the source corpora, CC BY-NC 4.0.
