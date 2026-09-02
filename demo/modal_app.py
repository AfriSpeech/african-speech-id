"""Language identification service: audio in, language out, on Modal CPU.

CPU only. The recogniser is int8, which is the right choice off a GPU -- int8 runs at 17x
realtime on CPU against fp32's 13x, and the ordering inverts on CUDA. The language head
itself costs about 0.06 ms, four orders of magnitude less than the recognition, so the whole
request is dominated by the ASR.

Cost control is the reason for the /warm endpoint. Containers scale to zero, so the first
request after an idle period pays a cold start of several seconds. The browser calls /warm
the moment the user presses record, and by the time they have spoken the four seconds the
model needs, the container is up. Nothing runs while nobody is recording.
"""
from __future__ import annotations

import io
import os
import time

import fastapi
import modal

MODELS = "/models"
ASR_DIR = f"{MODELS}/omniasr-300m"
HEAD_DIR = f"{MODELS}/head"
VAD_MODEL = f"{MODELS}/silero_vad.onnx"

# Both models are baked into the image rather than downloaded at run time: a cold start
# should be container boot plus model load, not a 350 MB download.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "bzip2", "git")   # none present in debian_slim
    .apt_install("ffmpeg")
    .pip_install("sherpa-onnx==1.13.6",
                 "african-speech-id @ git+https://github.com/AfriSpeech/african-speech-id", "fastapi[standard]",
                 "soundfile>=0.12", "numpy<2", "huggingface_hub>=0.30")
    .run_commands(
        f"mkdir -p {MODELS}",
        # omniASR CTC 300M int8 -- the front end
        f"cd {MODELS} && curl -sL -o a.tar.bz2 "
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/"
        "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2 "
        f"&& tar xjf a.tar.bz2 && rm a.tar.bz2 "
        f"&& mv sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12 {ASR_DIR}",
        # 0.6 MB, and it decides whether a recording is worth transcribing at all
        f"cd {MODELS} && curl -sL -O "
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx",
    )
    .run_commands(
        f"python -c \"from huggingface_hub import snapshot_download; "
        f"snapshot_download('AfriSpeech/african-speech-id', "
        f"local_dir='{HEAD_DIR}', allow_patterns=['*.onnx','*.txt','config.json'])\"",
    )
)

app = modal.App("african-speech-id", image=image)

# Accuracy climbs steeply with the amount of speech and flattens past roughly ten seconds,
# so the service asks for five and rejects anything below that. Short audio does not merely
# yield less text, it yields worse text -- the recogniser needs context and starves without
# it -- and silence contributes nothing while inflating the apparent duration.
#
# So the service asks for five seconds and checks that they are mostly speech. A recording
# that is half silence carries half the evidence its duration suggests.
MIN_SECONDS = 5.0
MAX_SECONDS = 20.0
MIN_SPEECH_RATIO = 0.80
MIN_SPEECH_SECONDS = 3.5

# An uploaded file supplies far more audio than someone will record by hand, and accuracy
# climbs steeply with speech, so decode a generous amount and let the VAD discard the
# silence between utterances rather than transcribing it.
UPLOAD_DECODE_SECONDS = 180   # how much of the file to decode before trimming
URL_SPEECH_TARGET = 20.0      # how much speech to keep once silence is dropped
MAX_UPLOAD_BYTES = 80 << 20
@app.cls(
    cpu=2,
    # Stay up briefly between requests so a user recording several clips pays one cold
    # start, but never idle for long: nothing should run while nobody is recording.
    scaledown_window=120,
    min_containers=0,
)
@modal.concurrent(max_inputs=4)
class Identifier:
    @modal.enter()
    def load(self):
        import sherpa_onnx
        from african_speech_id import AfricanSpeechId

        t0 = time.time()
        self.rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
            model=f"{ASR_DIR}/model.int8.onnx",
            tokens=f"{ASR_DIR}/tokens.txt",
            num_threads=2,
        )
        self.lid = AfricanSpeechId.load(HEAD_DIR, num_threads=2)

        vc = sherpa_onnx.VadModelConfig()
        vc.silero_vad.model = VAD_MODEL
        vc.silero_vad.threshold = 0.5
        vc.silero_vad.min_silence_duration = 0.25
        vc.silero_vad.min_speech_duration = 0.20
        vc.sample_rate = 16000
        self.vad_config = vc
        print(f"loaded in {time.time() - t0:.1f}s, {len(self.lid.languages)} languages, "
              f"VAD ready")

    def speech_seconds(self, audio, sr):
        """Seconds of detected speech. A fresh detector per call: it is stateful, and
        leaking state between requests would make results depend on call order."""
        import sherpa_onnx
        vad = sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=30)
        step = 512
        for i in range(0, len(audio), step):
            vad.accept_waveform(audio[i:i + step])
        vad.flush()
        total = 0.0
        while not vad.empty():
            seg = vad.front
            total += len(seg.samples) / sr
            vad.pop()
        return total

    def _run(self, wav_bytes: bytes) -> dict:
        import numpy as np
        import soundfile as sf

        t0 = time.time()
        audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        seconds = len(audio) / sr
        if seconds > MAX_SECONDS:                 # trim rather than reject
            audio = audio[: int(MAX_SECONDS * sr)]
            seconds = MAX_SECONDS

        rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
        if rms < 0.005:
            return {"ok": False, "reason": "too quiet",
                    "seconds": round(seconds, 2), "rms": round(rms, 4)}
        if seconds < MIN_SECONDS:
            return {"ok": False, "reason": "too short",
                    "seconds": round(seconds, 2), "need_seconds": MIN_SECONDS}

        speech = self.speech_seconds(audio, sr)
        ratio = speech / max(seconds, 1e-6)
        if speech < MIN_SPEECH_SECONDS or ratio < MIN_SPEECH_RATIO:
            return {"ok": False, "reason": "not enough speech",
                    "seconds": round(seconds, 2),
                    "speech_seconds": round(speech, 2),
                    "speech_ratio": round(ratio, 3),
                    "need_ratio": MIN_SPEECH_RATIO,
                    "need_speech_seconds": MIN_SPEECH_SECONDS}

        s = self.rec.create_stream()
        s.accept_waveform(sr, audio)
        self.rec.decode_stream(s)
        transcript = s.result.text.strip()
        t_asr = time.time() - t0

        if len(transcript) < 3:
            return {"ok": False, "reason": "no speech recognised",
                    "seconds": round(seconds, 2), "rms": round(rms, 4),
                    "transcript": transcript}

        t1 = time.time()
        post = self.lid.posterior(transcript) or {}
        top = sorted(post.items(), key=lambda kv: -kv[1])[:3]
        t_lid = time.time() - t1

        return {
            "ok": True,
            "language": top[0][0],
            "confidence": round(top[0][1], 4),
            "margin": round(top[0][1] - (top[1][1] if len(top) > 1 else 0.0), 4),
            "top": [{"language": k, "score": round(v, 4)} for k, v in top],
            "transcript": transcript,
            "seconds": round(seconds, 2),
            "speech_seconds": round(speech, 2),
            "speech_ratio": round(ratio, 3),
            "chars": len(transcript),
            # measured: about 20 characters is 0.51 out of domain, 40 is 0.64, 80 is 0.76
            "short": len(transcript) < 40,
            "timing_ms": {"asr": round(t_asr * 1000), "lid": round(t_lid * 1000)},
        }

    @modal.fastapi_endpoint(method="POST", docs=True)
    def identify(self, request_body: dict):
        """Accepts {"audio": "<base64 wav>"} and returns the predicted language."""
        import base64
        try:
            raw = base64.b64decode(request_body.get("audio", ""))
        except Exception:
            return {"ok": False, "reason": "audio is not valid base64"}
        if not raw:
            return {"ok": False, "reason": "no audio supplied"}
        try:
            return self._run(raw)
        except Exception as e:  # a bad upload should not take the container down
            return {"ok": False, "reason": f"{type(e).__name__}: {e}"}

    def speech_only(self, audio, sr, target):
        """Concatenate detected speech up to `target` seconds, dropping the gaps.

        Broadcast and interview audio is full of pauses, music and silence. Transcribing
        those wastes the budget: what the head needs is speech, and it gets steadily better
        with more of it.
        """
        import numpy as np
        import sherpa_onnx
        vad = sherpa_onnx.VoiceActivityDetector(self.vad_config, buffer_size_in_seconds=120)
        step = 512
        for i in range(0, len(audio), step):
            vad.accept_waveform(audio[i:i + step])
        vad.flush()
        parts, total = [], 0.0
        while not vad.empty():
            seg = vad.front
            samples = np.asarray(seg.samples, dtype=np.float32)
            parts.append(samples)
            total += len(samples) / sr
            vad.pop()
            if total >= target:
                break
        return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), total

    @modal.fastapi_endpoint(method="POST", docs=True)
    async def identify_file(self, request: fastapi.Request):
        """Accepts a raw audio or video file body and returns the predicted language.

        Decoding happens here rather than in the browser on purpose: a page can only decode
        the containers its own engine ships, which excludes most of what people actually
        have lying around, while ffmpeg reads effectively all of it. The VAD then keeps the
        speech and throws away the rest, so a long recording costs no more to classify than
        a short one -- and scores better, because accuracy climbs steeply with speech.
        """
        import subprocess
        import tempfile

        import soundfile as sf

        t0 = time.time()
        raw = await request.body()
        if not raw:
            return {"ok": False, "reason": "no file supplied"}
        if len(raw) > MAX_UPLOAD_BYTES:
            return {"ok": False, "reason": "file too large",
                    "limit_mb": MAX_UPLOAD_BYTES // (1 << 20)}

        with tempfile.TemporaryDirectory() as td:
            src, out = f"{td}/upload", f"{td}/a.wav"
            with open(src, "wb") as fh:
                fh.write(raw)
            # -t caps the decode: past the speech target there is nothing left to gain, and
            # an hour-long file would otherwise be decoded in full before the VAD trimmed it.
            r = subprocess.run(
                ["ffmpeg", "-nostdin", "-v", "error", "-y",
                 "-t", str(UPLOAD_DECODE_SECONDS), "-i", src,
                 "-vn", "-ar", "16000", "-ac", "1", out],
                capture_output=True, text=True, timeout=180)
            if not os.path.exists(out):
                err = (r.stderr or "").strip().splitlines()
                return {"ok": False, "reason": "could not read that file",
                        "detail": err[-1][:200] if err else "no audio track"}
            audio, sr = sf.read(out, dtype="float32", always_2d=False)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / sr
        t_decode = time.time() - t0

        speech, secs = self.speech_only(audio, sr, URL_SPEECH_TARGET)
        if secs < MIN_SPEECH_SECONDS:
            return {"ok": False, "reason": "not enough speech",
                    "fetched_seconds": round(duration, 1),
                    "speech_seconds": round(secs, 2),
                    "need_speech_seconds": MIN_SPEECH_SECONDS}

        buf = io.BytesIO()
        sf.write(buf, speech, sr, format="WAV", subtype="PCM_16")
        res = self._run(buf.getvalue())
        if isinstance(res, dict):
            res["source"] = "file"
            res["fetched_seconds"] = round(duration, 1)
            res["decode_ms"] = round(t_decode * 1000)
        return res

    @modal.fastapi_endpoint(method="GET", docs=True)
    def warm(self):
        """Cheap endpoint the browser hits on record-start, so the container is already up
        by the time the user has finished speaking."""
        return {"ok": True, "languages": len(self.lid.languages),
                "min_seconds": MIN_SECONDS, "max_seconds": MAX_SECONDS,
                "min_speech_ratio": MIN_SPEECH_RATIO,
                "min_speech_seconds": MIN_SPEECH_SECONDS}
