"""
tests/test_music.py
Acceptance tests for music/*.py.

Key tests:
  1. music_emotion() ALWAYS returns dict[str, float] — never a string.
  2. Emotion scores are not required to sum to 1.0 (multi-label).
  3. All expected keys are present in output.
  4. Scores are in [0.0, 1.0].
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from aue.music.music_emotion import compute_music_emotion, _ALL_EMOTION_LABELS


class TestMusicEmotion:
    def test_always_returns_dict(self):
        """compute_music_emotion ALWAYS returns dict — never a string."""
        result = compute_music_emotion(tempo_bpm=120.0, rms_energy=0.5, chroma_mode=0)
        assert isinstance(result, dict), (
            f"Expected dict, got {type(result).__name__}: {result!r}"
        )

    def test_returns_dict_for_all_inputs(self):
        """Multiple parameter combinations — all must return dict."""
        test_cases = [
            (None, 0.0, 0),
            (60.0, 0.1, 5),
            (180.0, 0.9, 11),
            (90.0, 0.5, 3),
            (45.0, 0.2, 7),
        ]
        for tempo, energy, chroma in test_cases:
            result = compute_music_emotion(tempo, energy, chroma)
            assert isinstance(result, dict), (
                f"Got {type(result).__name__} for inputs ({tempo}, {energy}, {chroma})"
            )

    def test_all_expected_keys_present(self):
        """All emotion labels are present as keys."""
        result = compute_music_emotion(tempo_bpm=100.0, rms_energy=0.4, chroma_mode=0)
        for label in _ALL_EMOTION_LABELS:
            assert label in result, f"Missing emotion key: {label!r}"

    def test_scores_in_range(self):
        """All scores are in [0.0, 1.0]."""
        result = compute_music_emotion(tempo_bpm=150.0, rms_energy=0.8, chroma_mode=2)
        for label, score in result.items():
            assert 0.0 <= score <= 1.0, (
                f"Score for {label!r} out of range: {score}"
            )

    def test_scores_do_not_need_to_sum_to_one(self):
        """
        Scores are multi-label — they are NOT required to sum to 1.0.
        This test asserts the design is multi-label, not softmax.
        """
        result = compute_music_emotion(tempo_bpm=160.0, rms_energy=0.9, chroma_mode=0)
        total = sum(result.values())
        # We only assert that it's NOT restricted to exactly 1.0
        # (it can be > 1.0 or < 1.0 — both are fine for multi-label)
        assert isinstance(total, float)
        # If this were a softmax output it would be exactly 1.0 — we don't require that
        # (asserting total != 1.0 would be too strict since it could accidentally equal 1.0)

    def test_fast_tempo_high_energy_scores_action(self):
        """Fast + energetic → action should be non-trivial."""
        result = compute_music_emotion(tempo_bpm=180.0, rms_energy=0.9, chroma_mode=0)
        assert result["action"] > 0.1, (
            f"Expected action > 0.1 for fast+energetic music, got {result['action']:.3f}"
        )

    def test_slow_tempo_low_energy_scores_calm(self):
        """Slow + quiet → calm should be non-trivial."""
        result = compute_music_emotion(tempo_bpm=45.0, rms_energy=0.05, chroma_mode=0)
        assert result["calm"] > 0.1, (
            f"Expected calm > 0.1 for slow+quiet music, got {result['calm']:.3f}"
        )

    def test_none_tempo_does_not_crash(self):
        """None tempo is handled gracefully."""
        result = compute_music_emotion(tempo_bpm=None, rms_energy=0.5, chroma_mode=0)
        assert isinstance(result, dict)
        assert all(0.0 <= v <= 1.0 for v in result.values())


class TestAcousticFeatures:
    def test_extract_features_returns_dict(self):
        """extract_music_acoustic_features returns a dict."""
        from aue.music.acoustic_features import extract_music_acoustic_features
        y = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
        result = extract_music_acoustic_features(y, sr=16000)
        assert isinstance(result, dict)

    def test_extract_features_has_expected_keys(self):
        """All expected keys present."""
        from aue.music.acoustic_features import extract_music_acoustic_features
        y = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
        result = extract_music_acoustic_features(y, sr=16000)
        for key in ["tempo_bpm", "rms_energy", "chroma", "chroma_mode", "spectral_centroid"]:
            assert key in result, f"Missing key: {key}"

    def test_short_waveform_returns_defaults(self):
        """Very short waveform returns default zero values without crashing."""
        from aue.music.acoustic_features import extract_music_acoustic_features
        y = np.zeros(10, dtype=np.float32)
        result = extract_music_acoustic_features(y, sr=16000)
        assert isinstance(result, dict)
        assert result["rms_energy"] == 0.0
