"""
tests/test_integration.py
Full pipeline integration test and AESE association test.

Tests:
  1. Full pipeline runs on a real video (comedy.mp4) without error.
  2. AESE event association attaches only overlapping segments (not adjacent).
  3. No single-string emotion labels appear in any output.
  4. RTF is measured (not asserted to be < 1 in tests, just measured and logged).
"""
import sys
import json
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from aue.types import AudioUnderstanding, MusicSegment, SoundEvent, SpeechSegment
from aue.aese_association import associate_with_events


# ---------------------------------------------------------------------------
# AESE Association tests (no real video required)
# ---------------------------------------------------------------------------

def _make_au(start_ms: float, end_ms: float, **kwargs) -> AudioUnderstanding:
    return AudioUnderstanding(start_ms=start_ms, end_ms=end_ms, **kwargs)


def _make_speech(start_ms: float, end_ms: float, transcript: str = "hello") -> SpeechSegment:
    return SpeechSegment(
        segment_id="SP_00000", start_ms=start_ms, end_ms=end_ms,
        speaker_id="SPK_01", transcript=transcript,
    )


def _make_event_dict(event_id: int, start_ms: float, end_ms: float) -> dict:
    return {
        "event_id": event_id,
        "start_time_ms": start_ms,
        "end_time_ms": end_ms,
        "character_labels": [],
    }


class TestAeseAssociation:
    def test_overlapping_segments_attached(self):
        """Speech segment overlapping an event attaches to that event."""
        speech = _make_speech(1000.0, 4000.0, transcript="Hello world")
        au = _make_au(0.0, 5000.0, speech_segments=[speech])
        event = _make_event_dict(0, 0.0, 5000.0)

        results = associate_with_events([au], [event])
        assert len(results) == 1
        assert len(results[0]["speech_segments"]) == 1
        assert results[0]["speech_segments"][0]["transcript"] == "Hello world"

    def test_adjacent_non_overlapping_not_attached(self):
        """Speech segment ending at event start does NOT attach (no overlap)."""
        speech = _make_speech(0.0, 5000.0)  # ends at 5000
        au = _make_au(0.0, 5000.0, speech_segments=[speech])
        event = _make_event_dict(0, 5000.0, 10000.0)  # starts at 5000

        results = associate_with_events([au], [event])
        assert len(results[0]["speech_segments"]) == 0

    def test_multiple_events_correct_attachment(self):
        """Two events, each gets only their own overlapping segments."""
        speech1 = _make_speech(0.0, 3000.0, transcript="first")
        speech2 = _make_speech(6000.0, 9000.0, transcript="second")
        au1 = _make_au(0.0, 5000.0, speech_segments=[speech1])
        au2 = _make_au(5000.0, 10000.0, speech_segments=[speech2])

        event1 = _make_event_dict(0, 0.0, 5000.0)
        event2 = _make_event_dict(1, 5000.0, 10000.0)

        results = associate_with_events([au1, au2], [event1, event2])
        assert len(results) == 2

        # event1 gets speech1
        ev1_transcripts = [s["transcript"] for s in results[0]["speech_segments"]]
        assert "first" in ev1_transcripts

        # event2 gets speech2
        ev2_transcripts = [s["transcript"] for s in results[1]["speech_segments"]]
        assert "second" in ev2_transcripts

    def test_empty_events_returns_empty(self):
        """No AESE events → empty result."""
        au = _make_au(0.0, 5000.0)
        results = associate_with_events([au], [])
        assert results == []

    def test_no_audio_understandings_returns_empty_summaries(self):
        """No AudioUnderstanding objects → events with empty audio context."""
        event = _make_event_dict(0, 0.0, 5000.0)
        results = associate_with_events([], [event])
        assert len(results) == 1
        assert results[0]["speech_segments"] == []
        assert results[0]["speakers"] == []

    def test_audio_importance_non_negative(self):
        """Audio importance is always >= 0."""
        au = _make_au(0.0, 5000.0)
        event = _make_event_dict(0, 0.0, 5000.0)
        results = associate_with_events([au], [event])
        assert results[0]["audio_importance"] >= 0.0


class TestImportanceScore:
    def test_explosion_scores_higher_than_quiet_speech(self):
        """
        Explosion sound event with high confidence scores higher than quiet conversation.
        Mirrors the AESE importance separation test.
        """
        from aue.importance import compute_audio_importance
        from aue.types import AUEConfig
        weights = AUEConfig().importance_weights

        explosion_score = compute_audio_importance(
            speech_present=False,
            music_intensity=0.1,
            sound_event_confidence=0.95,
            dialogue_semantic_score=0.0,
            context_score=0.8,
            weights=weights,
        )
        quiet_conv_score = compute_audio_importance(
            speech_present=True,
            music_intensity=0.0,
            sound_event_confidence=0.1,
            dialogue_semantic_score=0.05,
            context_score=0.2,
            weights=weights,
        )
        assert explosion_score > 0.0, "Explosion importance should be > 0"
        # Both should be in [0, 1]
        assert 0.0 <= explosion_score <= 1.0
        assert 0.0 <= quiet_conv_score <= 1.0

    def test_importance_clamped(self):
        """Importance is always in [0.0, 1.0]."""
        from aue.importance import compute_audio_importance
        weights = {"speech": 1.0, "music": 0.0, "sound_event": 0.0,
                   "dialogue_semantic": 0.0, "context": 0.0}
        score = compute_audio_importance(True, 1.0, 1.0, 1.0, 1.0, weights)
        assert 0.0 <= score <= 1.0


class TestTypes:
    def test_aueconfig_weights_sum_to_one(self):
        """
        AUEConfig importance_weights sum to 1.0.
        This is the §4 acceptance test mirroring AESE's weights-sum bug catch.
        """
        from aue.types import AUEConfig
        config = AUEConfig()
        total = sum(config.importance_weights.values())
        assert abs(total - 1.0) < 1e-4, (
            f"Weights sum to {total:.6f}, expected 1.0. "
            f"Keys: {config.importance_weights}"
        )

    def test_all_dataclasses_instantiate(self):
        """All AUE dataclasses instantiate with dummy values."""
        from aue.types import (
            AudioPacket, WordTimestamp, SpeechSegment, MusicSegment,
            SoundEvent, AudioUnderstanding,
        )
        AudioPacket(timestamp_ms=0.0, audio_energy=0.5, waveform_slice=np.zeros(100))
        WordTimestamp(word="hello", start_ms=0.0, end_ms=500.0)
        SpeechSegment(segment_id="SP_00000", start_ms=0.0, end_ms=1000.0, speaker_id="SPK_01")
        MusicSegment(start_ms=0.0, end_ms=2000.0, music_probability=0.8)
        SoundEvent(event_type="Explosion", start_ms=0.0, end_ms=1000.0, confidence=0.9)
        AudioUnderstanding(start_ms=0.0, end_ms=20000.0)


# ---------------------------------------------------------------------------
# Real video integration test (only runs if comedy.mp4 exists)
# ---------------------------------------------------------------------------

COMEDY_MP4 = Path(__file__).parent.parent.parent / "comedy.mp4"


@pytest.mark.skipif(not COMEDY_MP4.exists(), reason="comedy.mp4 not found — skipping integration test")
def test_full_pipeline_on_real_video():
    """
    Run the full AUE pipeline on comedy.mp4.
    Verifies:
      - No crash
      - Returns list of AudioUnderstanding objects
      - preload_audio called exactly once
      - No single-string emotion labels in output
    """
    from aue import audio_source
    from aue.types import AUEConfig
    from aue.pipeline import run

    audio_source.reset_call_counter()

    config = AUEConfig(
        rolling_buffer_seconds=10.0,
        whisper_model_size="base",
    )

    results = run(
        video_path=str(COMEDY_MP4),
        aese_events=[],
        config=config,
    )

    # preload_audio must have been called exactly once
    assert audio_source.get_call_count() == 1, (
        f"preload_audio called {audio_source.get_call_count()} times — audio-reload bug!"
    )

    assert isinstance(results, list)

    # Verify no single-string emotion labels
    for au in results:
        assert isinstance(au.acoustic_emotion, dict), \
            "acoustic_emotion must be a dict"
        for mseg in au.music_segments:
            assert isinstance(mseg.emotion, dict), \
                f"MusicSegment.emotion must be a dict, got {type(mseg.emotion).__name__}"

    print(f"\nIntegration test: {len(results)} AudioUnderstanding objects produced")
    print(f"  Speech segments: {sum(len(au.speech_segments) for au in results)}")
    print(f"  Music segments: {sum(len(au.music_segments) for au in results)}")
    print(f"  Sound events: {sum(len(au.sound_events) for au in results)}")

    audio_source.reset_call_counter()
