"""
aue/aue/aese_association.py
Temporal overlap join between AudioUnderstanding segments and AESE Event objects.

Implements Section 3.12: for each AESE Event, find all AudioUnderstanding
segments whose time range overlaps the event's [start_time_ms, end_time_ms].
Attach matching speech, music, and sound-event summaries to each event.

This is a pure interval-overlap join — no fuzzy matching, no interpolation.
Two intervals [a, b] and [c, d] overlap iff a < d AND c < b.

AESE Event import:
  Events are loaded from events.jsonl (CLI path) or passed directly from the
  AESE pipeline. The module works with either plain dicts (from JSONL) or
  AESE Event dataclass objects — both expose start_time_ms / end_time_ms.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _get_start(event: Any) -> float:
    """Extract start_time_ms from an AESE Event dict or dataclass."""
    if isinstance(event, dict):
        return float(event.get("start_time_ms", 0.0))
    return float(getattr(event, "start_time_ms", 0.0))


def _get_end(event: Any) -> float:
    """Extract end_time_ms from an AESE Event dict or dataclass."""
    if isinstance(event, dict):
        return float(event.get("end_time_ms", 0.0))
    return float(getattr(event, "end_time_ms", 0.0))


def _get_event_id(event: Any) -> int:
    """Extract event_id from an AESE Event dict or dataclass."""
    if isinstance(event, dict):
        return int(event.get("event_id", -1))
    return int(getattr(event, "event_id", -1))


def _overlaps(seg_start: float, seg_end: float, ev_start: float, ev_end: float) -> bool:
    """True iff [seg_start, seg_end) overlaps [ev_start, ev_end)."""
    return seg_start < ev_end and ev_start < seg_end


def _summarize_speech_segment(seg) -> dict:
    """Convert a SpeechSegment to a compact JSON-serializable summary."""
    return {
        "segment_id": seg.segment_id,
        "start_ms": seg.start_ms,
        "end_ms": seg.end_ms,
        "speaker_id": seg.speaker_id,
        "character_id": seg.character_id,
        "transcript": seg.transcript,
        "language": seg.language,
        "asr_confidence": round(seg.asr_confidence, 4),
        "low_confidence": seg.low_confidence,
    }


def _summarize_music_segment(seg) -> dict:
    """Convert a MusicSegment to a compact JSON-serializable summary."""
    return {
        "start_ms": seg.start_ms,
        "end_ms": seg.end_ms,
        "music_probability": round(seg.music_probability, 4),
        "emotion": {k: round(v, 4) for k, v in seg.emotion.items()},
        "intensity": round(seg.intensity, 4),
        "tension": round(seg.tension, 4),
        "tempo_bpm": seg.tempo_bpm,
        "confidence": round(seg.confidence, 4),
    }


def _summarize_sound_event(ev) -> dict:
    """Convert a SoundEvent to a compact JSON-serializable summary."""
    return {
        "event_type": ev.event_type,
        "start_ms": ev.start_ms,
        "end_ms": ev.end_ms,
        "confidence": round(ev.confidence, 4),
        "top_predictions": [(label, round(conf, 4)) for label, conf in ev.top_predictions[:3]],
    }


def associate_with_events(
    audio_understandings: List,
    aese_events: List,
) -> List[Dict]:
    """
    Join AudioUnderstanding segments with AESE Events by temporal overlap.

    Args:
        audio_understandings: List of AudioUnderstanding objects from the pipeline.
        aese_events:          List of AESE Event objects (dataclasses or dicts from JSONL).

    Returns:
        List of dicts, one per AESE event, with attached audio context:
        {
          "event_id": int,
          "event_start_ms": float,
          "event_end_ms": float,
          "speech_segments": [...],     # overlapping SpeechSegments
          "speakers": [...],            # unique speaker IDs in this event
          "music_segments": [...],      # overlapping MusicSegments
          "sound_events": [...],        # overlapping SoundEvents
          "acoustic_emotion": {...},    # mean of overlapping acoustic emotions
          "audio_importance": float,    # computed from overlapping data
          "audio_embedding": [...],     # mean of overlapping embeddings (as list)
        }

    Acceptance test:
        - Overlapping segments attach correctly.
        - Adjacent (non-overlapping) segments do NOT attach.
    """
    if not aese_events:
        logger.warning("AUE aese_association: no AESE events provided — nothing to associate.")
        return []

    if not audio_understandings:
        logger.warning("AUE aese_association: no AudioUnderstanding objects — returning empty summaries.")
        return [
            {
                "event_id": _get_event_id(ev),
                "event_start_ms": _get_start(ev),
                "event_end_ms": _get_end(ev),
                "speech_segments": [],
                "speakers": [],
                "music_segments": [],
                "sound_events": [],
                "acoustic_emotion": {},
                "audio_importance": 0.0,
                "audio_embedding": None,
            }
            for ev in aese_events
        ]

    results = []

    for event in aese_events:
        ev_start = _get_start(event)
        ev_end = _get_end(event)
        ev_id = _get_event_id(event)

        matched_speech = []
        matched_music = []
        matched_sed = []
        matched_emotions = []
        matched_embeddings = []

        for au in audio_understandings:
            au_start = getattr(au, "start_ms", 0.0)
            au_end = getattr(au, "end_ms", 0.0)

            if not _overlaps(au_start, au_end, ev_start, ev_end):
                continue

            # Collect speech segments that overlap the event
            for seg in getattr(au, "speech_segments", []):
                if _overlaps(seg.start_ms, seg.end_ms, ev_start, ev_end):
                    matched_speech.append(seg)

            # Collect music segments that overlap
            for mseg in getattr(au, "music_segments", []):
                if _overlaps(mseg.start_ms, mseg.end_ms, ev_start, ev_end):
                    matched_music.append(mseg)

            # Collect sound events that overlap
            for sev in getattr(au, "sound_events", []):
                if _overlaps(sev.start_ms, sev.end_ms, ev_start, ev_end):
                    matched_sed.append(sev)

            # Acoustic emotion (per AudioUnderstanding, not per sub-segment)
            emotion = getattr(au, "acoustic_emotion", {})
            if emotion:
                matched_emotions.append(emotion)

            # Embedding
            emb = getattr(au, "audio_embedding", None)
            if emb is not None and hasattr(emb, "__len__") and len(emb) > 0:
                matched_embeddings.append(emb)

        # Mean acoustic emotion across matched AUs
        mean_emotion = {}
        if matched_emotions:
            all_keys = set(k for e in matched_emotions for k in e.keys())
            for k in all_keys:
                vals = [e.get(k, 0.0) for e in matched_emotions]
                mean_emotion[k] = round(float(sum(vals) / len(vals)), 4)

        # Mean embedding
        mean_embedding = None
        if matched_embeddings:
            try:
                import numpy as np
                mean_embedding = np.mean(matched_embeddings, axis=0).tolist()
            except Exception:
                mean_embedding = None

        # Audio importance for this event
        from .importance import compute_audio_importance, compute_dialogue_semantic_score
        from .types import AUEConfig
        default_weights = AUEConfig().importance_weights

        transcript_all = " ".join(
            seg.transcript for seg in matched_speech if not seg.low_confidence
        )
        dialogue_score = compute_dialogue_semantic_score(transcript_all)
        music_intensity = max(
            (getattr(ms, "music_probability", 0.0) for ms in matched_music), default=0.0
        )
        sed_confidence = max(
            (getattr(se, "confidence", 0.0) for se in matched_sed), default=0.0
        )
        context_score = min(
            (len(matched_speech) + len(matched_music) + len(matched_sed)) / 10.0, 1.0
        )

        importance = compute_audio_importance(
            speech_present=len(matched_speech) > 0,
            music_intensity=music_intensity,
            sound_event_confidence=sed_confidence,
            dialogue_semantic_score=dialogue_score,
            context_score=context_score,
            weights=default_weights,
        )

        results.append({
            "event_id": ev_id,
            "event_start_ms": ev_start,
            "event_end_ms": ev_end,
            "speech_segments": [_summarize_speech_segment(s) for s in matched_speech],
            "speakers": sorted(set(s.speaker_id for s in matched_speech)),
            "music_segments": [_summarize_music_segment(m) for m in matched_music],
            "sound_events": [_summarize_sound_event(e) for e in matched_sed],
            "acoustic_emotion": mean_emotion,
            "audio_importance": round(importance, 4),
            "audio_embedding": mean_embedding,
        })

        logger.debug(
            "AUE association: event %d [%.0f–%.0f ms] → "
            "%d speech, %d music, %d SED",
            ev_id, ev_start, ev_end,
            len(matched_speech), len(matched_music), len(matched_sed),
        )

    logger.info(
        "AUE association: associated %d audio segments with %d AESE events.",
        len(audio_understandings), len(aese_events),
    )
    return results
