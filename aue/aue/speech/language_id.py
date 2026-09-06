"""
aue/aue/speech/language_id.py
Language identification per speech segment.

faster-whisper provides language detection as part of transcription.
This module provides a thin wrapper for standalone language detection
(when transcription is not needed, e.g. for music/noise rejection).

For most use cases, the language field on SpeechSegment is already
populated by transcribe_segments() — this module is for cases where
language detection needs to run independently.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Probability threshold below which language detection is considered uncertain.
_MIN_LANGUAGE_PROBABILITY = 0.7


def detect_language(
    waveform_chunk: np.ndarray,
    sr: int,
    model_size: str = "small",
) -> tuple:
    """
    Detect the spoken language in a short audio chunk.

    Args:
        waveform_chunk: Audio slice (from preloaded waveform).
        sr:             Sample rate (expected 16000 Hz).
        model_size:     faster-whisper model size.

    Returns:
        (language_code: str, probability: float)
        e.g. ("en", 0.98), ("hi", 0.81), ("unknown", 0.0)
    """
    if len(waveform_chunk) == 0:
        return "unknown", 0.0

    try:
        from .transcribe import _probe_whisper, _get_whisper_model
        if not _probe_whisper():
            return "unknown", 0.0

        model = _get_whisper_model(model_size)
        _, info = model.transcribe(
            waveform_chunk,
            word_timestamps=False,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        lang = getattr(info, "language", "unknown") or "unknown"
        prob = float(getattr(info, "language_probability", 0.0))

        if prob < _MIN_LANGUAGE_PROBABILITY:
            logger.debug(
                "AUE language_id: low language confidence (%s @ %.2f) — returned as-is.",
                lang, prob,
            )
        return lang, prob

    except Exception as exc:
        logger.warning("AUE language_id: detection failed (%s).", exc)
        return "unknown", 0.0
