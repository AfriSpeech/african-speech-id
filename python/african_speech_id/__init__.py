"""CPU-friendly, fast language identification for 1,386 African languages.

Sits on top of Omnilingual ASR: that model turns audio into text, this one says which
language the text is in.

    audio --[sherpa-onnx + omniASR CTC]--> transcript --[this]--> language
"""
from african_speech_id.model import DEFAULT_REPO, AfricanSpeechId, Prediction

__version__ = "0.1.0"
__all__ = ["DEFAULT_REPO", "AfricanSpeechId", "Prediction"]
