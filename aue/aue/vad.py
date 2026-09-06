"""
aue/aue/vad.py
Voice Activity Detection using Silero-VAD v5.

This is the primary latency lever for the speech pipeline:
Whisper (ASR) and resemblyzer (diarization) only ever run on VAD-detected
speech regions — never on the full track, never on music-only or silent stretches.

On a typical movie (60% music+silence, 40% speech), VAD reduces downstream
model invocations by ~60%, which is the single largest latency win after the
single-decode rule in audio_source.py.

Silero-VAD v5 API (torch hub):
    model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', onnx=False)
    get_speech_timestamps = utils[0]

Fallback: if silero-vad is unavailable (torch hub offline, or package missing),
falls back to a simple RMS energy threshold VAD. This is significantly less
accurate but never crashes the pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Silero-VAD model singleton (loaded once)
_silero_model = None
_silero_utils = None
_silero_available: Optional[bool] = None  # None = not probed yet

# Energy threshold fallback — used when Silero is unavailable
_ENERGY_FALLBACK_THRESHOLD = 0.005   # RMS squared
_ENERGY_WINDOW_MS = 30               # 30ms windows for energy VAD fallback
_ENERGY_MERGE_GAP_MS = 300           # merge speech regions closer than this
_ENERGY_MIN_DURATION_MS = 200        # discard regions shorter than this


def _probe_silero() -> bool:
    """Try to load Silero-VAD v5. Returns True if successful."""
    global _silero_model, _silero_utils, _silero_available
    if _silero_available is not None:
        return _silero_available

    try:
        import torch
        model, utils = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            onnx=False,
            verbose=False,
            trust_repo=True,
        )
        _silero_model = model
        _silero_utils = utils
        _silero_available = True
        logger.info("AUE VAD: Silero-VAD v5 loaded successfully.")
    except Exception as exc:
        _silero_available = False
        logger.warning(
            "AUE VAD: Silero-VAD unavailable (%s). "
            "Falling back to RMS energy threshold VAD — less accurate, "
            "but pipeline will continue. Install via: pip install silero-vad>=5.0",
            exc,
        )
    return _silero_available


def _silero_detect(
    waveform: np.ndarray,
    sr: int,
    threshold: float,
) -> list:
    """Run Silero-VAD v5 and return (start_ms, end_ms) pairs."""
    import torch

    # Silero expects 16kHz mono float32 tensor
    if sr != 16000:
        try:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000
        except Exception as exc:
            logger.warning("AUE VAD: resample failed (%s); using waveform as-is.", exc)

    audio_tensor = torch.FloatTensor(waveform)

    get_speech_timestamps = _silero_utils[0]
    timestamps = get_speech_timestamps(
        audio_tensor,
        _silero_model,
        threshold=threshold,
        sampling_rate=sr,
        return_seconds=False,   # sample indices
    )

    regions = []
    for ts in timestamps:
        start_ms = float(ts["start"]) / sr * 1000.0
        end_ms = float(ts["end"]) / sr * 1000.0
        regions.append((start_ms, end_ms))

    return regions


def _energy_fallback_detect(
    waveform: np.ndarray,
    sr: int,
    threshold: float,
) -> list:
    """
    Simple RMS energy threshold VAD fallback.

    Returns (start_ms, end_ms) regions where RMS energy exceeds threshold.
    threshold is reinterpreted as an energy fraction (0.0–1.0) for this path.
    """
    window_samples = max(1, int(_ENERGY_WINDOW_MS / 1000.0 * sr))
    energy_threshold = threshold * 0.1   # scale down from speech threshold to energy threshold

    is_speech = []
    for i in range(0, len(waveform), window_samples):
        window = waveform[i: i + window_samples]
        rms = float(np.sqrt(np.mean(window ** 2))) if len(window) > 0 else 0.0
        is_speech.append(rms > energy_threshold)

    # Convert boolean mask to (start_ms, end_ms) regions
    regions = []
    in_speech = False
    start_sample = 0
    for i, active in enumerate(is_speech):
        sample_pos = i * window_samples
        if active and not in_speech:
            in_speech = True
            start_sample = sample_pos
        elif not active and in_speech:
            in_speech = False
            start_ms = start_sample / sr * 1000.0
            end_ms = sample_pos / sr * 1000.0
            if (end_ms - start_ms) >= _ENERGY_MIN_DURATION_MS:
                regions.append((start_ms, end_ms))

    if in_speech:
        start_ms = start_sample / sr * 1000.0
        end_ms = len(waveform) / sr * 1000.0
        if (end_ms - start_ms) >= _ENERGY_MIN_DURATION_MS:
            regions.append((start_ms, end_ms))

    # Merge nearby regions
    regions = _merge_close_regions(regions, _ENERGY_MERGE_GAP_MS)
    return regions


def _merge_close_regions(regions: list, gap_ms: float) -> list:
    """Merge consecutive regions with gaps smaller than gap_ms."""
    if not regions:
        return regions
    merged = [regions[0]]
    for start, end in regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_ms:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def detect_speech_segments(
    waveform: np.ndarray,
    sr: int,
    threshold: float = 0.5,
) -> list:
    """
    Detect speech regions in a waveform using Silero-VAD v5.

    Args:
        waveform:  Mono float32 audio array (from preload_audio — never re-decoded).
        sr:        Sample rate (expected 16000 Hz from preload_audio).
        threshold: VAD speech probability threshold (0.0–1.0).
                   Higher = fewer but more certain speech regions.
                   Lower  = more regions but more music/noise included.

    Returns:
        List of (start_ms, end_ms) tuples for detected speech regions.
        Empty list if waveform is empty or no speech detected.

    Guarantee:
        Only VAD-detected regions are passed to Whisper and diarization.
        Music-only and silent stretches are never processed by those models.
    """
    if len(waveform) == 0:
        logger.warning("AUE VAD: empty waveform — returning no speech regions.")
        return []

    if _probe_silero():
        try:
            regions = _silero_detect(waveform, sr, threshold)
            total_speech_ms = sum(e - s for s, e in regions)
            total_audio_ms = len(waveform) / sr * 1000.0
            logger.info(
                "AUE VAD: %d speech regions detected | speech=%.1f s / total=%.1f s (%.0f%%)",
                len(regions),
                total_speech_ms / 1000.0,
                total_audio_ms / 1000.0,
                100.0 * total_speech_ms / max(total_audio_ms, 1.0),
            )
            return regions
        except Exception as exc:
            logger.warning("AUE VAD: Silero inference failed (%s); using energy fallback.", exc)

    # Energy fallback
    regions = _energy_fallback_detect(waveform, sr, threshold)
    logger.info(
        "AUE VAD: energy fallback — %d regions (Silero unavailable)",
        len(regions),
    )
    return regions
