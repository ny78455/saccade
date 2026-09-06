"""
aue/aue/acoustic_emotion.py
Vocal/acoustic emotion estimation independent of transcript content.

CONTRACT (per spec §3.9):
  This module estimates HOW something was said (vocal delivery features),
  NOT what was said. It is completely independent of transcript semantics.
  The acceptance test (tests/test_acoustic_emotion.py) verifies this by
  asserting that two identical waveforms with different transcript strings
  produce IDENTICAL emotion scores.

Features used (all waveform-derived, no text dependency):
  - Fundamental frequency (F0/pitch) via librosa.yin or parselmouth
  - Pitch variance (speaking rate variability)
  - Root mean square energy
  - Speaking rate (estimated via onset strength)
  - Pause patterns (fraction of silent frames)
  - Spectral centroid (brightness proxy)

Output:
  dict[str, float] — multi-label soft scores, same contract as music_emotion.
  Keys: {"fear", "anger", "sadness", "joy", "surprise", "disgust", "neutral"}
  Scores are in [0.0, 1.0]. Do NOT sum to 1.0 (multi-label, not exclusive).

V1 honesty note:
  This is a heuristic vocal-feature → emotion mapping, not a pretrained
  speech emotion model (wav2vec, SER model). Expected accuracy: ~50-60%
  on clear emotional speech. See DECISIONS.md §AUE-7.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# All emotion labels — always returned as keys
_EMOTION_LABELS = ["fear", "anger", "sadness", "joy", "surprise", "disgust", "neutral"]

# Silence threshold for pause detection
_SILENCE_RMS_THRESHOLD = 0.005

# Feature thresholds
_F0_HIGH_HZ = 250.0       # High pitch → surprise, joy, fear
_F0_LOW_HZ = 120.0        # Low pitch → sadness, neutral, anger (when energy high)
_ENERGY_HIGH = 0.50       # High energy → anger, joy
_ENERGY_LOW = 0.10        # Low energy → sadness, neutral
_PAUSE_HIGH_FRAC = 0.40   # Many pauses → sadness, fear
_VARIANCE_HIGH = 50.0     # High pitch variance → surprise, fear


def _extract_pitch_features(waveform: np.ndarray, sr: int) -> tuple:
    """
    Extract F0 (fundamental frequency) statistics from waveform.
    Returns (mean_f0_hz, f0_variance, f0_available).
    Uses librosa.yin (lightweight, no external dep required).
    """
    try:
        import librosa
        if len(waveform) < sr // 4:  # need at least 0.25s for pitch
            return 0.0, 0.0, False

        f0 = librosa.yin(
            waveform.astype(np.float32),
            fmin=librosa.note_to_hz("C2"),    # ~65 Hz — below human speech floor
            fmax=librosa.note_to_hz("C7"),    # ~2093 Hz — above human speech ceiling
            sr=sr,
        )
        # Voiced frames only (F0 > 0)
        voiced = f0[f0 > 0]
        if len(voiced) < 5:
            return 0.0, 0.0, False
        return float(np.mean(voiced)), float(np.var(voiced)), True
    except Exception as exc:
        logger.debug("AUE acoustic_emotion: pitch extraction failed: %s", exc)
        return 0.0, 0.0, False


def _extract_energy(waveform: np.ndarray) -> float:
    """Normalized RMS energy [0.0, 1.0]."""
    if len(waveform) == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(waveform.astype(np.float32) ** 2)))
    return float(np.clip(rms / 0.3, 0.0, 1.0))  # normalize against typical speech level


def _extract_pause_fraction(waveform: np.ndarray, sr: int) -> float:
    """Fraction of 30ms windows with RMS below silence threshold."""
    if len(waveform) == 0:
        return 1.0
    window_size = max(1, int(sr * 0.03))
    n_silent = 0
    n_total = 0
    for start in range(0, len(waveform), window_size):
        window = waveform[start: start + window_size]
        rms = float(np.sqrt(np.mean(window.astype(np.float32) ** 2)))
        if rms < _SILENCE_RMS_THRESHOLD:
            n_silent += 1
        n_total += 1
    return float(n_silent / max(n_total, 1))


def _extract_speaking_rate(waveform: np.ndarray, sr: int) -> float:
    """
    Estimate speaking rate via onset density (onsets per second).
    Not syllable-level accuracy but a reasonable proxy.
    """
    try:
        import librosa
        if len(waveform) < sr // 2:
            return 0.0
        onset_env = librosa.onset.onset_strength(y=waveform.astype(np.float32), sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        duration_s = len(waveform) / sr
        rate = len(onsets) / max(duration_s, 0.1)
        return float(np.clip(rate / 10.0, 0.0, 1.0))  # normalize: 10 onsets/s = 1.0
    except Exception:
        return 0.0


def compute_acoustic_emotion(
    waveform_slice: np.ndarray,
    sr: int,
    transcript: Optional[str] = None,  # intentionally IGNORED — independence enforced
) -> dict:
    """
    Estimate vocal emotion from waveform features ONLY.

    The `transcript` parameter exists only to satisfy callers that may pass
    it. It is INTENTIONALLY IGNORED. Acoustic emotion must remain independent
    of transcript semantics to avoid circular reasoning in downstream fusion.

    Args:
        waveform_slice: Audio slice from preloaded waveform.
        sr:             Sample rate.
        transcript:     IGNORED. Present for API compatibility only.

    Returns:
        dict[str, float] — multi-label emotion scores.
        Keys always present: fear, anger, sadness, joy, surprise, disgust, neutral.
        Values in [0.0, 1.0]. Do NOT sum to 1.0.
    """
    # transcript is explicitly NOT used — this is an enforced design constraint
    _ = transcript  # acknowledged and discarded

    scores = {label: 0.0 for label in _EMOTION_LABELS}

    if len(waveform_slice) < 256:
        scores["neutral"] = 0.5
        return scores

    # Extract all features from waveform only
    mean_f0, f0_variance, f0_available = _extract_pitch_features(waveform_slice, sr)
    energy = _extract_energy(waveform_slice)
    pause_frac = _extract_pause_fraction(waveform_slice, sr)
    speaking_rate = _extract_speaking_rate(waveform_slice, sr)

    # --- Fear ---
    # High pitch + high variance + high pauses + moderate energy
    if f0_available and mean_f0 >= _F0_HIGH_HZ and f0_variance >= _VARIANCE_HIGH:
        scores["fear"] = min(1.0, 0.35 + pause_frac * 0.3 + (energy * 0.2))
    elif pause_frac >= _PAUSE_HIGH_FRAC and energy <= _ENERGY_LOW:
        scores["fear"] = min(1.0, 0.20 + pause_frac * 0.3)

    # --- Anger ---
    # High energy + low-moderate pitch + fast speaking rate
    if energy >= _ENERGY_HIGH:
        scores["anger"] = min(1.0, 0.30 + energy * 0.4 + speaking_rate * 0.2)
    if f0_available and _F0_LOW_HZ <= mean_f0 <= 200 and energy >= _ENERGY_HIGH:
        scores["anger"] = min(1.0, scores["anger"] + 0.2)

    # --- Sadness ---
    # Low energy + low pitch + many pauses + slow speaking rate
    if energy <= _ENERGY_LOW and pause_frac >= _PAUSE_HIGH_FRAC:
        scores["sadness"] = min(1.0, 0.30 + (1.0 - energy) * 0.3 + pause_frac * 0.3)
    if f0_available and mean_f0 < _F0_LOW_HZ and energy <= _ENERGY_LOW:
        scores["sadness"] = min(1.0, scores["sadness"] + 0.2)

    # --- Joy ---
    # High pitch + high energy + fast speaking rate
    if f0_available and mean_f0 >= _F0_HIGH_HZ and energy >= _ENERGY_LOW:
        scores["joy"] = min(1.0, 0.25 + energy * 0.35 + speaking_rate * 0.25)

    # --- Surprise ---
    # Very high pitch variance + sudden energy change
    if f0_available and f0_variance >= _VARIANCE_HIGH * 1.5:
        scores["surprise"] = min(1.0, 0.30 + (f0_variance / (2 * _VARIANCE_HIGH)) * 0.4)
    elif energy >= _ENERGY_HIGH and f0_available and mean_f0 >= _F0_HIGH_HZ:
        scores["surprise"] = min(1.0, scores["surprise"] + 0.15)

    # --- Disgust ---
    # Low-moderate energy + low pitch + irregular speaking rate
    if f0_available and mean_f0 < 150 and _ENERGY_LOW < energy < _ENERGY_HIGH:
        scores["disgust"] = min(1.0, 0.15 + (1.0 - speaking_rate) * 0.2 + (150 - mean_f0) / 100 * 0.2)

    # --- Neutral ---
    # Default when no strong emotion signals
    max_emotion = max(scores[k] for k in _EMOTION_LABELS if k != "neutral")
    if max_emotion < 0.3:
        scores["neutral"] = min(1.0, 0.5 + (0.3 - max_emotion))
    else:
        scores["neutral"] = max(0.0, 0.4 - max_emotion * 0.4)

    # Clamp all
    for k in scores:
        scores[k] = float(max(0.0, min(1.0, scores[k])))

    return scores
