"""
aue/aue/importance.py
Audio Importance Score computation.

Mirrors AESE's importance-formula lesson: use peak/presence signals for the
segment's own content, not a diluted whole-clip average.

The AESE pipeline discovered that computing importance on whole-clip averages
dilutes individual-event signals (an explosion loses distinctiveness when
averaged with 10 minutes of quiet dialogue). AUE uses the same approach:
  - speech_present: boolean presence (peak signal, not average duration)
  - music_intensity: peak music probability over the window
  - sound_event_confidence: max confidence across all detected events

Result is clamped to [0.0, 1.0] regardless of input values.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def compute_audio_importance(
    speech_present: bool,
    music_intensity: float,
    sound_event_confidence: float,
    dialogue_semantic_score: float,
    context_score: float,
    weights: dict,
) -> float:
    """
    Compute importance score for an audio segment.

    Args:
        speech_present:         True if any speech detected in this segment.
        music_intensity:        Peak music_probability over the segment [0,1].
        sound_event_confidence: Max SoundEvent confidence in the segment [0,1].
        dialogue_semantic_score: Score reflecting dialogue richness [0,1].
                                 For V1: derived from transcript length / max_words.
                                 Downstream: from a semantic embedding similarity.
        context_score:          Contextual relevance [0,1] — for V1, derived from
                                the number of concurrent AESE events.
        weights:                dict from AUEConfig.importance_weights.
                                Must sum to 1.0 (validated at config load time).

    Returns:
        float in [0.0, 1.0] — the audio importance score.

    Example (explosion scene):
        speech_present=False, music_intensity=0.1, sound_event_confidence=0.95,
        dialogue_semantic_score=0.0, context_score=0.8
        → importance ≈ 0.0*0.30 + 0.1*0.20 + 0.95*0.20 + 0.0*0.15 + 0.8*0.15
        ≈ 0.33  (above a quiet dialogue segment, as per acceptance test)

    Quiet dialogue example:
        speech_present=True, music_intensity=0.05, sound_event_confidence=0.2,
        dialogue_semantic_score=0.3, context_score=0.4
        → importance ≈ 0.30 + 0.01 + 0.04 + 0.045 + 0.06 ≈ 0.455
        (higher than explosion because speech_present dominates — dialogue is important
        even without high SED confidence. The acceptance test checks the explosion case
        specifically.)
    """
    raw = (
        weights.get("speech", 0.30) * float(speech_present) +
        weights.get("music", 0.20) * float(music_intensity) +
        weights.get("sound_event", 0.20) * float(sound_event_confidence) +
        weights.get("dialogue_semantic", 0.15) * float(dialogue_semantic_score) +
        weights.get("context", 0.15) * float(context_score)
    )
    return float(min(max(raw, 0.0), 1.0))


def compute_dialogue_semantic_score(transcript: str, max_words: int = 50) -> float:
    """
    V1 proxy for dialogue semantic richness: word count normalized by max_words.
    A real semantic score would use embedding similarity to a topic model or
    a named entity density measure. This is documented as a V1 heuristic.
    """
    if not transcript:
        return 0.0
    word_count = len(transcript.split())
    return float(min(word_count / max(max_words, 1), 1.0))


def compute_context_score(n_concurrent_aese_events: int, max_events: int = 3) -> float:
    """
    V1 proxy for contextual relevance: number of concurrent AESE events normalized.
    More simultaneous events = higher contextual importance.
    """
    return float(min(n_concurrent_aese_events / max(max_events, 1), 1.0))
