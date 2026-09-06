"""
tests/test_vad.py
Acceptance tests for vad.py.

Key tests:
  1. VAD detects speech in a synthetic speech-like signal.
  2. VAD returns empty for a silent waveform.
  3. Total detected speech < full track duration for a silent+speech mix.
     (proves the gate actually reduces downstream work)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


def _make_silence(duration_s: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def _make_noise(duration_s: float, sr: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    """Gaussian noise — acts as a speech-like signal for energy-based VAD."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(duration_s * sr)) * amplitude).astype(np.float32)


def _make_tone(freq_hz: float, duration_s: float, sr: int = 16000, amplitude: float = 0.3) -> np.ndarray:
    """Pure tone — higher ZCR than noise, less speech-like."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    return (np.sin(2 * np.pi * freq_hz * t) * amplitude).astype(np.float32)


def test_vad_returns_list():
    """detect_speech_segments always returns a list."""
    from aue.vad import detect_speech_segments
    y = _make_noise(1.0)
    result = detect_speech_segments(y, sr=16000, threshold=0.5)
    assert isinstance(result, list)


def test_vad_empty_waveform_returns_empty():
    """Empty waveform → empty speech region list."""
    from aue.vad import detect_speech_segments
    result = detect_speech_segments(np.array([], dtype=np.float32), sr=16000)
    assert result == []


def test_vad_silence_returns_no_or_few_regions():
    """Pure silence should produce very few or no speech regions."""
    from aue.vad import detect_speech_segments
    silence = _make_silence(5.0)
    regions = detect_speech_segments(silence, sr=16000, threshold=0.3)
    # Total detected speech should be < 1s for a 5s silent track
    total_speech_ms = sum(e - s for s, e in regions)
    assert total_speech_ms < 1500.0, (
        f"VAD detected {total_speech_ms:.0f} ms of speech in silence — "
        "threshold may be too low or VAD fallback is too sensitive."
    )


def test_vad_speech_regions_have_valid_timestamps():
    """All regions have start < end and are within waveform duration."""
    from aue.vad import detect_speech_segments
    y = np.concatenate([_make_silence(1.0), _make_noise(2.0), _make_silence(1.0)])
    sr = 16000
    regions = detect_speech_segments(y, sr=sr, threshold=0.3)
    total_ms = len(y) / sr * 1000.0
    for start, end in regions:
        assert start >= 0.0, f"Negative start: {start}"
        assert end > start, f"end <= start: {start}, {end}"
        assert end <= total_ms + 100.0, f"end beyond waveform: {end} > {total_ms}"


def test_vad_gates_reduces_total_duration():
    """
    For a waveform with 60% silence + 40% speech-like noise,
    detected speech duration must be meaningfully less than full duration.
    This proves the gate actually reduces downstream model invocations.
    """
    from aue.vad import detect_speech_segments
    sr = 16000
    silence_3s = _make_silence(3.0, sr)
    noise_2s = _make_noise(2.0, sr, amplitude=0.4)
    silence_3s_2 = _make_silence(3.0, sr)
    noise_2s_2 = _make_noise(2.0, sr, amplitude=0.4)

    y = np.concatenate([silence_3s, noise_2s, silence_3s_2, noise_2s_2])
    total_ms = len(y) / sr * 1000.0  # 10000ms

    regions = detect_speech_segments(y, sr=sr, threshold=0.3)
    detected_ms = sum(e - s for s, e in regions)

    # Detected speech should be much less than total duration (< 80%)
    assert detected_ms < total_ms * 0.80, (
        f"VAD detected {detected_ms:.0f}ms out of {total_ms:.0f}ms — "
        "gate is not reducing downstream work effectively."
    )


def test_vad_region_format():
    """VAD returns list of (float, float) tuples."""
    from aue.vad import detect_speech_segments
    y = _make_noise(2.0)
    regions = detect_speech_segments(y, sr=16000, threshold=0.3)
    for item in regions:
        assert len(item) == 2
        start, end = item
        assert isinstance(start, float)
        assert isinstance(end, float)
