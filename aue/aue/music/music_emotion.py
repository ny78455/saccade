"""
aue/aue/music/music_emotion.py
Multi-label music emotion classification.

V1 IMPLEMENTATION: Heuristic mapping from tempo + energy + chroma_mode → emotion scores.
A dedicated pretrained music-emotion model (e.g. musicnn, Music Transformer) is future work.
This V1 status is documented here and in DECISIONS.md §AUE-6 — NOT silently omitted.

CONTRACT (MANDATORY — per spec §3.7.3 and §3.21):
  - Output is ALWAYS a dict[str, float] with emotion labels as keys and probabilities [0,1] as values.
  - A single string label is NEVER returned, even when one emotion strongly dominates.
  - Scores are NOT required to sum to 1.0 (multi-label, not mutually exclusive).
  - Scores of 0.0 are included explicitly for transparency when an emotion is clearly absent.
  - The acceptance test in tests/test_music.py asserts isinstance(result, dict) always.

Emotion label set:
  action, suspense, calm, melancholic, joyful, tense, romantic

Heuristic mapping rationale:
  - Fast tempo (>120 BPM) + high energy → action / joyful
  - Slow tempo (<60 BPM) + low energy → calm / melancholic
  - High energy + low tempo → tense / suspense
  - Minor key (chroma_mode in minor positions) → melancholic / suspense
  - Major key (chroma_mode in major positions) → joyful / romantic
  Scores are additive and independently clamped to [0, 1].
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Tempo thresholds (BPM)
_BPM_SLOW = 60.0
_BPM_FAST = 120.0
_BPM_VERY_FAST = 160.0

# Energy thresholds (RMS normalized)
_ENERGY_LOW = 0.15
_ENERGY_MED = 0.40
_ENERGY_HIGH = 0.70

# Chroma mode sets: which chroma bins correspond to "major" vs "minor" tonality.
# Standard music theory: C major = 0, D major = 2, E major = 4, F major = 5,
# G major = 7, A major = 9, B major = 11 (natural major positions)
# Minor positions approximate: A minor = 9, E minor = 4, D minor = 2, etc.
# This is a rough heuristic — a real key/mode classifier would be more accurate.
_MAJOR_CHROMA_MODES = {0, 2, 4, 5, 7, 9, 11}   # natural scale positions
_MINOR_CHROMA_MODES = {1, 3, 6, 8, 10}           # non-natural positions (approx minor tendency)

# All emotion labels — always returned, even if score is 0.0
_ALL_EMOTION_LABELS = ["action", "suspense", "calm", "melancholic", "joyful", "tense", "romantic"]


def compute_music_emotion(
    tempo_bpm: Optional[float],
    rms_energy: float,
    chroma_mode: int,
) -> dict:
    """
    Compute multi-label music emotion scores from acoustic features.

    Args:
        tempo_bpm:   Estimated tempo in BPM (from acoustic_features.py). None if unknown.
        rms_energy:  Normalized RMS energy [0.0–1.0].
        chroma_mode: Dominant chroma bin [0–11] (proxy for musical key).

    Returns:
        dict[str, float] — multi-label emotion scores, each in [0.0, 1.0].
        NEVER a single string. Scores do NOT need to sum to 1.0.

    V1 honesty note:
        This is a heuristic mapping, not a pretrained music-emotion model.
        Expected accuracy: ~60-70% on obvious emotion categories (high-energy action,
        quiet calm). More nuanced emotions (romantic, melancholic) are approximated.
        See DECISIONS.md §AUE-6 for future work (musicnn, Music Transformer).
    """
    # Initialize all scores at 0.0
    scores = {label: 0.0 for label in _ALL_EMOTION_LABELS}

    # Handle unknown tempo
    bpm = tempo_bpm if tempo_bpm is not None else 90.0  # assume moderate if unknown

    # Tonality signal
    is_major_tendency = chroma_mode in _MAJOR_CHROMA_MODES
    is_minor_tendency = chroma_mode in _MINOR_CHROMA_MODES

    # --- Action ---
    # Fast tempo + high energy
    if bpm >= _BPM_FAST and rms_energy >= _ENERGY_MED:
        scores["action"] = min(1.0, 0.4 + (bpm - _BPM_FAST) / (_BPM_VERY_FAST - _BPM_FAST) * 0.4
                               + rms_energy * 0.3)

    # --- Joyful ---
    # Fast tempo + major key
    if bpm >= _BPM_FAST and is_major_tendency:
        scores["joyful"] = min(1.0, 0.35 + rms_energy * 0.4 + (bpm - _BPM_FAST) / 80.0 * 0.25)

    # --- Tense ---
    # High energy + slow-moderate tempo (builds without resolving)
    if rms_energy >= _ENERGY_MED and bpm < _BPM_FAST:
        scores["tense"] = min(1.0, rms_energy * 0.6 + 0.2)

    # --- Suspense ---
    # Moderate energy + minor tendency + moderate tempo
    if is_minor_tendency and rms_energy >= _ENERGY_LOW:
        scores["suspense"] = min(1.0, 0.3 + rms_energy * 0.4 + (0.2 if bpm < _BPM_FAST else 0.0))
    elif rms_energy >= _ENERGY_HIGH:
        scores["suspense"] = min(scores["suspense"] + 0.2, 1.0)

    # --- Calm ---
    # Slow tempo + low energy
    if bpm <= _BPM_SLOW and rms_energy <= _ENERGY_MED:
        scores["calm"] = min(1.0, 0.4 + (1.0 - rms_energy) * 0.4 + (_BPM_SLOW - bpm) / _BPM_SLOW * 0.2)

    # --- Melancholic ---
    # Slow tempo + minor key + low-moderate energy
    if bpm <= _BPM_SLOW and is_minor_tendency:
        scores["melancholic"] = min(1.0, 0.35 + (1.0 - rms_energy) * 0.35 + 0.15)
    elif is_minor_tendency and rms_energy <= _ENERGY_LOW:
        scores["melancholic"] = min(1.0, scores["melancholic"] + 0.25)

    # --- Romantic ---
    # Slow-moderate tempo + major key + moderate energy
    if _BPM_SLOW <= bpm <= _BPM_FAST and is_major_tendency and _ENERGY_LOW <= rms_energy <= _ENERGY_HIGH:
        scores["romantic"] = min(1.0, 0.30 + (1.0 - abs(rms_energy - 0.4)) * 0.3)

    # Clamp all scores to [0.0, 1.0]
    for k in scores:
        scores[k] = float(max(0.0, min(1.0, scores[k])))

    return scores
