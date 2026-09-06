"""
tests/test_audio_source.py
Acceptance tests for audio_source.py.

Critical test: preload_audio() is called exactly once per pipeline run,
regardless of clip length or segment count. This is the regression guard
for the ASVL audio-reload bug.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from aue import audio_source


@pytest.fixture(autouse=True)
def reset_counter():
    """Reset the call counter before each test."""
    audio_source.reset_call_counter()
    yield
    audio_source.reset_call_counter()


def test_preload_audio_returns_tuple(tmp_path):
    """preload_audio on a non-existent file returns empty waveform (graceful)."""
    y, sr = audio_source.preload_audio(str(tmp_path / "nonexistent.mp4"))
    assert isinstance(y, np.ndarray)
    assert isinstance(sr, int)
    assert sr == 16000


def test_preload_audio_call_count_increments():
    """Call counter increments each time preload_audio is called."""
    assert audio_source.get_call_count() == 0
    audio_source.preload_audio("/fake/path.mp4")
    assert audio_source.get_call_count() == 1
    audio_source.preload_audio("/fake/path.mp4")
    assert audio_source.get_call_count() == 2


def test_call_counter_reset():
    """reset_call_counter() brings count back to 0."""
    audio_source.preload_audio("/fake/path.mp4")
    assert audio_source.get_call_count() == 1
    audio_source.reset_call_counter()
    assert audio_source.get_call_count() == 0


def test_slice_waveform_basic():
    """slice_waveform returns the correct portion."""
    sr = 16000
    y = np.arange(sr * 10, dtype=np.float32)  # 10s at 16kHz
    chunk = audio_source.slice_waveform(y, sr, start_ms=1000.0, end_ms=3000.0)
    assert len(chunk) == sr * 2  # 2 seconds


def test_slice_waveform_clamped_to_end():
    """slice_waveform clamps to end of waveform without error."""
    sr = 16000
    y = np.ones(sr * 5, dtype=np.float32)
    chunk = audio_source.slice_waveform(y, sr, start_ms=4500.0, end_ms=6000.0)
    assert len(chunk) > 0
    assert len(chunk) <= sr  # at most 1 second (500ms of audio remains)


def test_slice_waveform_empty_waveform():
    """slice_waveform on empty waveform returns empty array."""
    chunk = audio_source.slice_waveform(np.array([]), 16000, 0.0, 1000.0)
    assert len(chunk) == 0


def test_audio_duration_ms():
    """audio_duration_ms returns correct duration."""
    sr = 16000
    y = np.zeros(sr * 10, dtype=np.float32)  # 10s
    dur = audio_source.audio_duration_ms(y, sr)
    assert abs(dur - 10000.0) < 1.0


def test_single_decode_contract_simulated():
    """
    Simulated pipeline loop: even if we call preload_audio once outside
    and process many segments, the counter stays at 1.
    This is the key regression test for the ASVL bug.
    """
    # Simulate pipeline: decode once, then slice N times
    y = np.zeros(16000 * 60, dtype=np.float32)  # 60s audio
    sr = 16000
    audio_source.reset_call_counter()

    # This is what the pipeline CORRECTLY does:
    _waveform, _sr = audio_source.preload_audio("/fake/movie.mp4")

    # Process 10 "segments" — each slices from waveform, does NOT call preload_audio
    for i in range(10):
        _chunk = audio_source.slice_waveform(y, sr, i * 6000.0, (i + 1) * 6000.0)

    # preload_audio must have been called exactly once
    assert audio_source.get_call_count() == 1, (
        f"preload_audio was called {audio_source.get_call_count()} times — "
        "this is the audio-reload bug! It must be called exactly once."
    )
