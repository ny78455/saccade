"""
aue/aue/types.py
Core data contracts for the Audio Understanding Engine (Module 3).

Design notes:
  - AudioPacket is built here from AUE's own single in-memory decode —
    NOT received as a packet stream from Module 1 (ASVL). ASVL does not
    expose raw waveform data downstream. See DECISIONS.md §AUE-1.
  - Multi-label fields (MusicSegment.emotion, AudioUnderstanding.acoustic_emotion,
    SoundEvent.top_predictions) are NEVER collapsed to a single label even when
    one label dominates, per spec §3.21.
  - AUEConfig.importance_weights sum check runs at import time (mirroring the
    AESE spec bug catch from §4 — the spec's weights happen to sum to 1.0 here,
    but we verify explicitly rather than assuming).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# AudioPacket — built from AUE's own single in-memory decode
# ---------------------------------------------------------------------------

@dataclass
class AudioPacket:
    """
    Corrected from the source spec's assumption — built here from AUE's own
    single in-memory decode, not received from Module 1.

    Module 1 (ASVL) only exposes scalar audio_energy per 250ms window;
    it does not stream raw waveforms to downstream modules.
    See DECISIONS.md §AUE-1 for the full deviation record.
    """
    timestamp_ms: float
    audio_energy: float          # cross-checked against Module 1's own value where available
    waveform_slice: np.ndarray   # in-memory slice, never re-decoded from disk


# ---------------------------------------------------------------------------
# WordTimestamp — word-level ASR output
# ---------------------------------------------------------------------------

@dataclass
class WordTimestamp:
    word: str
    start_ms: float
    end_ms: float


# ---------------------------------------------------------------------------
# SpeechSegment — one VAD-detected + transcribed speech region
# ---------------------------------------------------------------------------

@dataclass
class SpeechSegment:
    segment_id: str
    start_ms: float
    end_ms: float
    speaker_id: str                     # anonymous until SpeakerCharacterBinder resolves, e.g. "SPK_01"
    character_id: Optional[str] = None  # only set with real single-subject evidence — never guessed
    transcript: str = ""
    words: list = field(default_factory=list)   # list[WordTimestamp]
    language: str = "unknown"
    asr_confidence: float = 0.0         # derived from faster-whisper's avg_logprob
    low_confidence: bool = False        # True => transcript may be unreliable, FLAGGED NOT HIDDEN
    # low_confidence=True does NOT discard the transcript. Downstream consumers
    # must check this flag before treating the transcript as reliable input.


# ---------------------------------------------------------------------------
# MusicSegment — one detected music region
# ---------------------------------------------------------------------------

@dataclass
class MusicSegment:
    start_ms: float
    end_ms: float
    music_probability: float
    # MULTI-LABEL — never a single string, never forced to one dominant label.
    # Scores are NOT required to sum to 1 (labels are not mutually exclusive).
    # Per spec §3.7.3 and §3.21.
    emotion: dict = field(default_factory=dict)   # e.g. {"suspense": 0.87, "action": 0.41}
    intensity: float = 0.0
    tension: float = 0.0
    tempo_bpm: Optional[float] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# SoundEvent — one detected sound event region
# ---------------------------------------------------------------------------

@dataclass
class SoundEvent:
    event_type: str                # "ambiguous" when top confidence < threshold
    start_ms: float
    end_ms: float
    confidence: float
    # Populated when event_type == "ambiguous" (per spec §3.21 "gunshot vs firecracker"
    # example). Also populated for non-ambiguous cases as a transparency record.
    top_predictions: list = field(default_factory=list)  # list[tuple[str, float]]


# ---------------------------------------------------------------------------
# AudioUnderstanding — primary output object, one per rolling buffer window
# ---------------------------------------------------------------------------

@dataclass
class AudioUnderstanding:
    start_ms: float
    end_ms: float
    speech_segments: list = field(default_factory=list)   # list[SpeechSegment]
    speakers: list = field(default_factory=list)           # list[str], anonymous IDs
    music_segments: list = field(default_factory=list)     # list[MusicSegment]
    sound_events: list = field(default_factory=list)       # list[SoundEvent]
    # MULTI-LABEL soft scores — same contract as MusicSegment.emotion.
    # Per spec §3.9: this estimates HOW something was said, not WHAT was said.
    # It is combined with visual/dialogue/music emotion later, not used as a
    # final label on its own.
    acoustic_emotion: dict = field(default_factory=dict)   # e.g. {"fear": 0.71, "anger": 0.43}
    audio_embedding: Optional[np.ndarray] = None
    confidence: float = 0.0


# ---------------------------------------------------------------------------
# AUEConfig — runtime configuration
# ---------------------------------------------------------------------------

# Spec's default importance_weights — verified to sum to 1.0 below.
_DEFAULT_IMPORTANCE_WEIGHTS = {
    "speech": 0.30,
    "music": 0.20,
    "sound_event": 0.20,
    "dialogue_semantic": 0.15,
    "context": 0.15,
}
_WEIGHTS_SUM = sum(_DEFAULT_IMPORTANCE_WEIGHTS.values())  # must == 1.0


@dataclass
class AUEConfig:
    rolling_buffer_seconds: float = 20.0
    vad_speech_threshold: float = 0.5
    # "small" is the chosen tradeoff: meaningfully faster than "medium" or "large"
    # with acceptable WER on movie dialogue. See DECISIONS.md §AUE-2 for benchmark.
    whisper_model_size: str = "small"
    speaker_cluster_threshold: float = 0.45   # mirrors CharacterClusterer DISTANCE_THRESHOLD_CLIP
    sound_event_confidence_threshold: float = 0.5
    music_detection_threshold: float = 0.5
    music_window_seconds: float = 2.0          # PANNs sliding window
    sed_window_seconds: float = 1.0            # SED sliding window
    panns_checkpoint_dir: Optional[str] = None # None => default ~/.panns_data/
    importance_weights: dict = field(
        default_factory=lambda: dict(_DEFAULT_IMPORTANCE_WEIGHTS)
    )
    # RTF gate: pipeline logs a WARNING if total_processing_time / audio_duration >= 1.0
    rtf_warn_threshold: float = 1.0


def _assert_importance_weights_sum(cfg: AUEConfig) -> None:
    """
    §4 acceptance test: assert importance_weights sum to ~1.0.
    Mirrors the AESE spec bug catch (AESE's weights summed to 1.05).
    The spec's AUE weights happen to sum to 1.00, but we verify explicitly
    rather than assuming — same discipline, same guard.
    Raises AssertionError if the sum drifts (e.g. due to config file override).
    """
    total = sum(cfg.importance_weights.values())
    assert abs(total - 1.0) < 1e-4, (
        f"AUEConfig.importance_weights sum to {total:.6f}, expected 1.0. "
        "Check DECISIONS.md §AUE-1 for weight rationale. "
        f"Keys: {list(cfg.importance_weights.keys())}"
    )


# Run the check at import time on the default config.
# This is the same pattern as AESE's types.py — if we ever accidentally
# edit the defaults to be wrong, this catches it at startup not at test time.
_assert_importance_weights_sum(AUEConfig())
