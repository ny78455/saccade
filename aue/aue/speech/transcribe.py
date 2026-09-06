"""
aue/aue/speech/transcribe.py
Speech transcription using faster-whisper (CTranslate2-backed, int8 quantized).

Why faster-whisper over vanilla openai-whisper:
  faster-whisper uses CTranslate2 for inference — typically 2–4× faster than
  the reference implementation at equivalent model size and accuracy.
  int8 quantization further reduces memory and compute with minimal WER impact.
  See DECISIONS.md §AUE-2 for benchmark numbers.

Key implementation choices:
  - Singleton model: _get_whisper_model() loads once, reused for all segments.
  - word_timestamps=True: required for WordTimestamp construction.
  - condition_on_previous_text=False: avoids the known Whisper failure mode where
    hallucinated text at the end of one segment "bleeds" into the next via
    conditioning, especially after silent or music-heavy stretches.
  - Waveform is sliced from the preloaded array — never re-decoded from disk.
  - low_confidence flag: set when avg_logprob < LOW_CONFIDENCE_THRESHOLD.
    The transcript is STILL STORED (not discarded) but flagged. Downstream
    consumers must check this flag before trusting the transcript text.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..types import SpeechSegment, WordTimestamp

logger = logging.getLogger(__name__)

# avg_logprob threshold below which we flag low_confidence.
# faster-whisper avg_logprob is typically -0.2 to -0.6 for clear speech,
# and -1.0 or below for genuinely garbled/inaudible content.
LOW_CONFIDENCE_THRESHOLD = -1.0

# faster-whisper model singleton cache: model_size -> model instance
_whisper_models: dict = {}
_whisper_available: Optional[bool] = None


def _probe_whisper() -> bool:
    """Check if faster-whisper is available."""
    global _whisper_available
    if _whisper_available is not None:
        return _whisper_available
    try:
        import faster_whisper  # noqa: F401
        _whisper_available = True
    except ImportError:
        _whisper_available = False
        logger.warning(
            "AUE speech: faster-whisper not available. "
            "Transcription will be skipped. Install via: pip install faster-whisper>=1.0.0"
        )
    return _whisper_available


def _get_whisper_model(model_size: str = "small"):
    """
    Return a singleton faster-whisper WhisperModel for the given model_size.
    Loaded once, reused for all segments — never re-instantiated per call.
    """
    if model_size in _whisper_models:
        return _whisper_models[model_size]

    from faster_whisper import WhisperModel

    logger.info(
        "AUE speech: loading faster-whisper model '%s' (int8, compute_type='int8')...",
        model_size,
    )
    try:
        model = WhisperModel(model_size, device="auto", compute_type="int8")
    except Exception:
        # Fallback: CPU with int8
        logger.warning("AUE speech: 'auto' device failed; falling back to CPU int8.")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    _whisper_models[model_size] = model
    logger.info("AUE speech: faster-whisper '%s' model ready.", model_size)
    return model


def _extract_avg_logprob(segments_list: list) -> float:
    """
    Extract average log-probability from faster-whisper segment objects.
    Returns a weighted average across all decoded segments.
    Falls back to -2.0 (low confidence) if no segments or attribute missing.
    """
    if not segments_list:
        return -2.0
    total_logprob = 0.0
    total_duration = 0.0
    for seg in segments_list:
        try:
            dur = max(0.01, float(seg.end) - float(seg.start))
            lp = float(getattr(seg, "avg_logprob", -2.0))
            total_logprob += lp * dur
            total_duration += dur
        except Exception:
            pass
    if total_duration <= 0:
        return -2.0
    return total_logprob / total_duration


def _extract_words(segments_list: list, offset_ms: float = 0.0) -> list:
    """
    Extract WordTimestamp objects from faster-whisper segment word objects.
    offset_ms is added to all timestamps to convert from within-chunk to global time.
    """
    words = []
    for seg in segments_list:
        seg_words = getattr(seg, "words", None) or []
        for w in seg_words:
            try:
                words.append(WordTimestamp(
                    word=str(w.word).strip(),
                    start_ms=float(w.start) * 1000.0 + offset_ms,
                    end_ms=float(w.end) * 1000.0 + offset_ms,
                ))
            except Exception:
                pass
    return words


def _extract_full_text(segments_list: list) -> str:
    """Concatenate text from all faster-whisper segments."""
    return " ".join(
        str(getattr(seg, "text", "")).strip()
        for seg in segments_list
        if getattr(seg, "text", "").strip()
    )


def transcribe_segments(
    waveform: np.ndarray,
    sr: int,
    speech_regions: list,
    model_size: str = "small",
) -> list:
    """
    Transcribe VAD-detected speech regions using faster-whisper.

    Args:
        waveform:       Full preloaded waveform (from audio_source.preload_audio).
                        Sliced in-memory per region — NEVER re-decoded from disk.
        sr:             Sample rate (expected 16000 Hz).
        speech_regions: List of (start_ms, end_ms) from VAD.
        model_size:     faster-whisper model size (base/small/medium/large-v3).

    Returns:
        List of SpeechSegment objects, one per VAD region.

    Honesty contract:
        - low_confidence=True segments are STILL returned (not discarded).
        - Downstream consumers must check segment.low_confidence before
          treating segment.transcript as reliable input.
        - No transcript is fabricated: if the model returns garbled text with
          very low avg_logprob, we flag it rather than silently trusting it.
    """
    if not speech_regions:
        return []

    if not _probe_whisper():
        logger.warning("AUE speech: transcription skipped (faster-whisper unavailable).")
        return [
            SpeechSegment(
                segment_id=f"SP_{i:05d}",
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_id="UNASSIGNED",
                transcript="",
                low_confidence=True,
                asr_confidence=-2.0,
            )
            for i, (start_ms, end_ms) in enumerate(speech_regions)
        ]

    model = _get_whisper_model(model_size)
    segments = []

    for idx, (start_ms, end_ms) in enumerate(speech_regions):
        start_sample = max(0, int(start_ms / 1000.0 * sr))
        end_sample = min(len(waveform), int(end_ms / 1000.0 * sr))

        if start_sample >= end_sample:
            continue

        chunk = waveform[start_sample:end_sample]

        if len(chunk) == 0:
            continue

        try:
            # condition_on_previous_text=False prevents hallucination cascade
            # across segments after music/silence stretches (see module docstring).
            result_generator, info = model.transcribe(
                chunk,
                word_timestamps=True,
                condition_on_previous_text=False,
                vad_filter=False,   # VAD already done upstream by silero; don't double-apply
            )
            result_segments = list(result_generator)

            avg_logprob = _extract_avg_logprob(result_segments)
            transcript = _extract_full_text(result_segments)
            words = _extract_words(result_segments, offset_ms=start_ms)
            language = getattr(info, "language", "unknown") or "unknown"

            is_low = avg_logprob < LOW_CONFIDENCE_THRESHOLD

            if is_low:
                logger.debug(
                    "AUE speech: low confidence segment SP_%05d "
                    "(avg_logprob=%.3f < %.1f) — flagged, not discarded.",
                    idx, avg_logprob, LOW_CONFIDENCE_THRESHOLD,
                )

            segments.append(SpeechSegment(
                segment_id=f"SP_{idx:05d}",
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_id="UNASSIGNED",
                transcript=transcript,
                words=words,
                language=language,
                asr_confidence=avg_logprob,
                low_confidence=is_low,
            ))

        except Exception as exc:
            logger.warning(
                "AUE speech: transcription failed for region [%.0f–%.0f ms]: %s",
                start_ms, end_ms, exc,
            )
            segments.append(SpeechSegment(
                segment_id=f"SP_{idx:05d}",
                start_ms=start_ms,
                end_ms=end_ms,
                speaker_id="UNASSIGNED",
                transcript="",
                low_confidence=True,
                asr_confidence=-2.0,
            ))

    logger.info(
        "AUE speech: transcribed %d / %d regions | %d low-confidence",
        len(segments),
        len(speech_regions),
        sum(1 for s in segments if s.low_confidence),
    )
    return segments
