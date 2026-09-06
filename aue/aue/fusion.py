"""
aue/aue/fusion.py
Audio Fusion Layer — combines speech, music, sound-event, and acoustic-emotion
representations into a single audio_embedding per segment.

V1 IMPLEMENTATION: temporal mean pooling of concatenated feature vectors.
This is the honest V1 choice — not an attention-pooling claim.
Mirrors AESE's pool_event_embedding (aese/event_embedding.py) which uses the
same temporal mean pooling approach with the same justification: a simple,
correct baseline before more sophisticated pooling is validated.

Embedding layout (concatenation order, documented for schema stability):
  [speech_features (64-dim)] ++ [music_features (32-dim)] ++
  [sed_features (16-dim)] ++ [acoustic_emotion_features (7-dim)]
  Total: 119-dim (padded to 128-dim with zeros for alignment)
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Embedding dimensions per component
_SPEECH_DIM = 64
_MUSIC_DIM = 32
_SED_DIM = 16
_EMOTION_DIM = 7   # matches len(_EMOTION_LABELS) in acoustic_emotion.py
_TOTAL_RAW_DIM = _SPEECH_DIM + _MUSIC_DIM + _SED_DIM + _EMOTION_DIM  # 119
_PADDED_DIM = 128  # padded to power of 2 for downstream compatibility

# Emotion label order (must match acoustic_emotion.py's _EMOTION_LABELS)
_EMOTION_LABEL_ORDER = ["fear", "anger", "sadness", "joy", "surprise", "disgust", "neutral"]


def _speech_feature_vector(speech_segments: List) -> np.ndarray:
    """
    Encode speech segments into a 64-dim feature vector via mean pooling.
    Features: presence, avg confidence, avg duration, language distribution,
              word count proxy, low-confidence fraction.
    """
    vec = np.zeros(_SPEECH_DIM, dtype=np.float32)
    if not speech_segments:
        return vec

    n = len(speech_segments)
    # [0]: speech presence (always 1 if any segments)
    vec[0] = 1.0
    # [1]: fraction of low-confidence segments
    vec[1] = sum(1 for s in speech_segments if s.low_confidence) / n
    # [2]: avg ASR confidence (shifted to [0,1]: logprob typically -2 to 0)
    avg_lp = sum(getattr(s, "asr_confidence", -1.0) for s in speech_segments) / n
    vec[2] = float(np.clip((avg_lp + 2.0) / 2.0, 0.0, 1.0))
    # [3]: avg segment duration (normalized by 30s)
    avg_dur = sum(s.end_ms - s.start_ms for s in speech_segments) / n
    vec[3] = float(np.clip(avg_dur / 30000.0, 0.0, 1.0))
    # [4]: number of segments (normalized by 20)
    vec[4] = float(np.clip(n / 20.0, 0.0, 1.0))
    # [5]: distinct speaker count (normalized by 5)
    speakers = set(getattr(s, "speaker_id", "?") for s in speech_segments)
    vec[5] = float(np.clip(len(speakers) / 5.0, 0.0, 1.0))
    # [6–15]: avg word timestamps density (words per second, normalized)
    total_words = sum(len(getattr(s, "words", [])) for s in speech_segments)
    total_dur_s = sum((s.end_ms - s.start_ms) for s in speech_segments) / 1000.0
    wps = total_words / max(total_dur_s, 0.1)
    vec[6] = float(np.clip(wps / 5.0, 0.0, 1.0))  # 5 wps = fast speech
    # Remaining dims initialized to 0 (reserved for future speech features)
    return vec


def _music_feature_vector(music_segments: List) -> np.ndarray:
    """Encode music segments into a 32-dim feature vector."""
    vec = np.zeros(_MUSIC_DIM, dtype=np.float32)
    if not music_segments:
        return vec

    n = len(music_segments)
    # [0]: music presence
    vec[0] = 1.0
    # [1]: avg music probability
    vec[1] = float(np.clip(sum(getattr(s, "music_probability", 0.0) for s in music_segments) / n, 0.0, 1.0))
    # [2]: avg intensity
    vec[2] = float(np.clip(sum(getattr(s, "intensity", 0.0) for s in music_segments) / n, 0.0, 1.0))
    # [3]: avg tension
    vec[3] = float(np.clip(sum(getattr(s, "tension", 0.0) for s in music_segments) / n, 0.0, 1.0))
    # [4]: avg tempo (normalized by 200 BPM)
    tempos = [getattr(s, "tempo_bpm", None) for s in music_segments if getattr(s, "tempo_bpm", None) is not None]
    vec[4] = float(np.clip(np.mean(tempos) / 200.0, 0.0, 1.0)) if tempos else 0.0

    # [5–11]: mean emotion scores (7 keys from music_emotion)
    music_emotion_labels = ["action", "suspense", "calm", "melancholic", "joyful", "tense", "romantic"]
    for i, label in enumerate(music_emotion_labels):
        scores = [getattr(s, "emotion", {}).get(label, 0.0) for s in music_segments]
        vec[5 + i] = float(np.clip(np.mean(scores), 0.0, 1.0)) if scores else 0.0

    return vec


def _sed_feature_vector(sound_events: List) -> np.ndarray:
    """Encode sound events into a 16-dim feature vector."""
    vec = np.zeros(_SED_DIM, dtype=np.float32)
    if not sound_events:
        return vec

    n = len(sound_events)
    vec[0] = 1.0  # SED presence
    # [1]: avg confidence
    vec[1] = float(np.clip(sum(getattr(e, "confidence", 0.0) for e in sound_events) / n, 0.0, 1.0))
    # [2]: fraction of ambiguous events
    vec[2] = sum(1 for e in sound_events if getattr(e, "event_type", "") == "ambiguous") / n
    # [3]: distinct event type count (normalized by 10)
    types = set(getattr(e, "event_type", "?") for e in sound_events if getattr(e, "event_type", "") != "ambiguous")
    vec[3] = float(np.clip(len(types) / 10.0, 0.0, 1.0))
    # [4]: high-confidence peak (max confidence across events)
    vec[4] = float(np.clip(max((getattr(e, "confidence", 0.0) for e in sound_events), default=0.0), 0.0, 1.0))

    return vec


def _emotion_feature_vector(acoustic_emotion: dict) -> np.ndarray:
    """Encode acoustic emotion dict into a 7-dim vector (fixed label order)."""
    vec = np.zeros(_EMOTION_DIM, dtype=np.float32)
    for i, label in enumerate(_EMOTION_LABEL_ORDER):
        vec[i] = float(np.clip(acoustic_emotion.get(label, 0.0), 0.0, 1.0))
    return vec


def fuse_audio_segment(
    speech_segments: List,
    music_segments: List,
    sound_events: List,
    acoustic_emotion: dict,
) -> np.ndarray:
    """
    Combine audio modality representations into a single embedding vector.

    V1: temporal mean pooling of concatenated feature vectors.
    This mirrors AESE's pool_event_embedding() approach — the honest baseline.
    Future work: attention-weighted pooling, learned fusion network.

    Args:
        speech_segments:  List of SpeechSegment objects.
        music_segments:   List of MusicSegment objects.
        sound_events:     List of SoundEvent objects.
        acoustic_emotion: dict[str, float] from acoustic_emotion.py.

    Returns:
        np.ndarray float32 of shape (_PADDED_DIM,) = (128,).
        Zero-vector if all inputs are empty.
    """
    speech_vec = _speech_feature_vector(speech_segments)     # (64,)
    music_vec = _music_feature_vector(music_segments)        # (32,)
    sed_vec = _sed_feature_vector(sound_events)              # (16,)
    emotion_vec = _emotion_feature_vector(acoustic_emotion)  # (7,)

    raw = np.concatenate([speech_vec, music_vec, sed_vec, emotion_vec])  # (119,)

    # Pad to _PADDED_DIM (128)
    padded = np.zeros(_PADDED_DIM, dtype=np.float32)
    padded[:len(raw)] = raw

    # L2-normalize (unless zero vector)
    norm = np.linalg.norm(padded)
    if norm > 0:
        padded = padded / norm

    return padded
