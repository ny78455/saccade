"""
tests/test_adapters.py
Acceptance tests for all Feature Adapters (§5.0).

Tests:
  - All adapters run on 3 dummy frames without crash
  - Every field of TemporalFeature is populated (no None for required fields)
  - STUB adapters return the correct stub values
"""
import numpy as np
import pytest


def _make_dummy_frame(h=64, w=64) -> np.ndarray:
    """Create a random RGB frame."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


def _make_black_frame(h=64, w=64) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# action_stub
# ---------------------------------------------------------------------------
from aese.adapters.action_stub import label_action


def test_action_stub_static():
    assert label_action(0.0) == "static"
    assert label_action(0.19) == "static"


def test_action_stub_walking():
    assert label_action(0.2) == "walking"
    assert label_action(0.49) == "walking"


def test_action_stub_fast():
    assert label_action(0.5) == "fast_action"
    assert label_action(1.0) == "fast_action"


# ---------------------------------------------------------------------------
# music_mood
# ---------------------------------------------------------------------------
from aese.adapters.music_mood import label_mood, estimate_spectral_flux


def test_music_mood_calm():
    assert label_mood(0.03, 0.01) == "calm"


def test_music_mood_energetic():
    assert label_mood(0.30, 0.0) == "energetic"


def test_music_mood_tense():
    assert label_mood(0.15, 0.0) == "tense"


def test_spectral_flux_estimate():
    assert estimate_spectral_flux(None, 0.5) == 0.0
    assert abs(estimate_spectral_flux(0.3, 0.5) - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# camera_cues
# ---------------------------------------------------------------------------
from aese.adapters.camera_cues import detect_camera_cue


def test_camera_cut():
    frame = _make_dummy_frame()
    result = detect_camera_cue(scene_change=True, image=frame, prev_was_black=False)
    assert result == "cut"


def test_camera_black_frame():
    black = _make_black_frame()
    result = detect_camera_cue(scene_change=False, image=black, prev_was_black=False)
    assert result == "black"


def test_camera_fade():
    frame = _make_dummy_frame()
    result = detect_camera_cue(scene_change=False, image=frame, prev_was_black=True)
    assert result == "fade"


def test_camera_none():
    frame = _make_dummy_frame()
    result = detect_camera_cue(scene_change=False, image=frame, prev_was_black=False)
    assert result is None


# ---------------------------------------------------------------------------
# character_stub
# ---------------------------------------------------------------------------
from aese.adapters.character_stub import count_characters


def test_character_none_image():
    assert count_characters(None) == 0


def test_character_black_frame():
    assert count_characters(_make_black_frame()) == 0


def test_character_returns_int():
    frame = _make_dummy_frame()
    result = count_characters(frame)
    assert isinstance(result, int)
    assert result >= 0


# ---------------------------------------------------------------------------
# scene_label
# ---------------------------------------------------------------------------
from aese.adapters.scene_label import label_scene, _SCENE_LABELS


def test_scene_label_returns_valid():
    frame = _make_dummy_frame()
    result = label_scene(frame)
    assert result in _SCENE_LABELS


def test_scene_label_black_frame():
    # §21.4: pure black now returns 'graphics/end card' via Path 2
    # (dark_fraction=1.0 > 0.70, color_std=0 < 30). Not 'unknown'.
    result = label_scene(_make_black_frame())
    assert result == "graphics/end card"


def test_scene_label_none():
    # Should not crash on None
    result = label_scene(None)
    assert result == "unknown"


# ---------------------------------------------------------------------------
# embedding
# ---------------------------------------------------------------------------
from aese.adapters.embedding import compute_embedding


def test_embedding_returns_array():
    frame = _make_dummy_frame()
    emb = compute_embedding(frame, subtitle_text=None)
    assert isinstance(emb, np.ndarray)
    assert emb.ndim == 1
    assert emb.dtype == np.float32
    assert len(emb) > 0


def test_embedding_with_text():
    frame = _make_dummy_frame()
    emb = compute_embedding(frame, subtitle_text="Hello world")
    assert isinstance(emb, np.ndarray)
    assert len(emb) > 0


def test_embedding_black_frame():
    """Black frame should not crash — returns valid embedding (zeros or stub)."""
    emb = compute_embedding(_make_black_frame(), subtitle_text=None)
    assert isinstance(emb, np.ndarray)
    assert len(emb) > 0


def test_embedding_different_inputs():
    """Two very different frames should produce different embeddings."""
    frame1 = _make_black_frame()
    frame1[:, :, 0] = 255  # pure red
    frame2 = _make_black_frame()
    frame2[:, :, 2] = 255  # pure blue
    emb1 = compute_embedding(frame1)
    emb2 = compute_embedding(frame2)
    # Should not be identical
    assert not np.allclose(emb1, emb2, atol=1e-6)
