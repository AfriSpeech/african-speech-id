# african-speech-id

CPU-friendly, fast language identification for 1,386 African languages.

The model uses a fast version of Omnilingual ASR to turn speech into text, and a
classification head that reads that text and names the language. Both run on CPU, so a
phone or a laptop is enough.

The inference core is C++ with a C API, so there is no Python on the device.

```sh
pip install african-speech-id
```

**How the model behaves and where it fails** is on the
[model card](https://huggingface.co/AfriSpeech/african-speech-id). This file is about using it.

## Results

Evaluated on two held-out sets, against
[facebook/mms-lid-4017](https://huggingface.co/facebook/mms-lid-4017) on identical clips:

| test set | languages | clips | **this model** | MMS-LID-4017 |
|---|---|---|---|---|
| [Waxal](https://huggingface.co/datasets/google/WaxalNLP) | 28 | 1,675 | **0.610** | 0.604 |
| [omniASR corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) (test) | 76 | 4,557 | **0.429** | 0.270 |
| **combined** | **104** | **6,232** | **0.478** | **0.360** |

Per language the model is ahead on 57, behind on 38, and tied on 9.

The gap is widest on the long tail. MMS-LID covers 4,017 languages and is strong on
well-resourced ones; this model is built for the languages underneath that, which is where
the omniASR test set sits.

## Quick start

The head classifies text and cannot read audio, so it needs a recogniser in front of it:

```python
import soundfile as sf
import sherpa_onnx
from african_speech_id import AfricanSpeechId

model, tokens = AfricanSpeechId.download_recogniser()   # omniASR, once
rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(model=model, tokens=tokens)
lid = AfricanSpeechId.load()

wav, sr = sf.read("clip.wav", dtype="float32")
s = rec.create_stream()
s.accept_waveform(sr, wav)
rec.decode_stream(s)

print(lid.classify(s.result.text))            # ewe (0.03)
```

`classify()` returns `None` when no n-gram matched, meaning there was no basis for a
decision. Report that as unknown rather than naming whichever language scored least badly.

Labels are ISO 639-3 codes (`ewe`, `hau`, `fat`) where the source corpus carried one, and
language names otherwise.

**Confidence values are small and that is expected.** A softmax over 1,386 classes spreads
thin: a confident answer often sits around 0.03–0.05. Threshold on `margin` (top-1 minus
top-2) rather than on `confidence`.

**The head is closed-set.** It always names one of its 1,386 languages, including for speech
in a language it has never seen. `margin` is the signal to threshold on if you need to
reject those.

## How much audio to give it

Five seconds minimum, ten for the best result. Accuracy climbs steeply with the amount of
speech and then flattens; past about ten seconds there is little left to gain. Check that
most of the audio is speech before transcribing — a recording that is half silence carries
half the evidence its duration suggests.

## Command line

```sh
african-speech-id "nusia nunɔ eƒe ŋuse kpledzikpɔkpɔ te"
african-speech-id --top 3 < transcripts.txt
```

## C

```c
AsidConfig cfg;
asid_config_init(&cfg);
cfg.onnx_path   = "300m/head.onnx";
cfg.ngrams_path = "300m/ngrams.txt";
cfg.labels_path = "300m/labels.txt";
cfg.config_path = "300m/head_config.txt";

char err[512];
AsidHead *h = asid_create(&cfg, err, sizeof err);

AsidResult r = asid_classify(h, transcript);
if (r.index >= 0) printf("%s %.3f\n", asid_language(h, r.index), r.confidence);
else              printf("unknown\n");

asid_destroy(h);
```

`index == -1` is the same "no basis for a decision" case as `None` in Python.

## Android and iOS

`bindings/android` has the JNI shim, a Kotlin wrapper and a CMake file.
`bindings/ios` has a Swift wrapper and a module map, so the C API imports with no
Objective-C shim and no bridging header.

## Speed

Fast enough to be practical on ordinary hardware, which is the point of building it this
way. Measured on one Xeon Platinum 8558 core, int8, on held-out audio:

| | throughput |
|---|---|
| classification head alone | 0.28 ms per call, 3,600 per second |
| full pipeline, 1 thread | 3.5x realtime |
| full pipeline, 4 threads | 9.3x realtime |
| facebook/mms-lid-4017, same machine | 1.4x realtime |

So a ten-second clip is identified in under three seconds on a single core, and in about a
second on four. MMS-LID reads language straight off the audio and needs no recogniser, but
it is a 970-million-parameter model: 3.9 GB against our 265 MB head, and slower end to end
even counting the recogniser we depend on.

No GPU path exists and none is wanted. The head is a vocabulary lookup and one sparse
gather, so a GPU would spend longer on transfers than on arithmetic.

The head is 265 MB, 1,386 classes over 50,000 features. A 300,000-feature version scored
0.4833 against 0.4779, half a point for twelve times the size, so the smaller vocabulary
ships.

## Building from source

```sh
cmake -S . -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build -j
ASID_MODEL_DIR=model ./build/asid_selftest
```

Only dependency is onnxruntime. The ONNX graph uses **opset-13 core operators only** — no
`com.microsoft` contrib ops — so it runs in mobile onnxruntime builds. The tf-idf arithmetic
is inside the graph:

```
inputs   indices int64[K], counts float32[K]
         tf = 1+log(counts) → ×idf → L2 normalise → Gather(W) → ReduceSum → +b → softmax
outputs  logits float32[C], probs float32[C]
```

Reproducing scikit-learn's `char_wb` exactly is the delicate part, and it has two traps that
fail silently rather than raising — see
[docs-char-tokenisation.md](docs-char-tokenisation.md). Every release is checked against the
trainer: sklearn and the exported graph must agree on held-out transcripts.

## Licence

Code Apache-2.0. Model and data follow the source corpora, CC BY-NC 4.0.
