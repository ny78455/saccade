"""
tests/test_scene_label_regression.py
Regression and contract tests for aese/adapters/scene_label.py.

Covers:
  Fix 2: label_scene() always returns a str (never None), CLIP availability check
  Fix 3: expanded vocabulary contract (15 labels, public SCENE_LABELS constant)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest.mock as mock
import numpy as np
import pytest

from aese.adapters.scene_label import label_scene, _clip_available, SCENE_LABELS


# ---------------------------------------------------------------------------
# Fix 2 -- label_scene always returns str
# ---------------------------------------------------------------------------

def test_label_scene_none_image_returns_unknown():
    """label_scene(None) must return 'unknown', never None."""
    result = label_scene(None)
    assert result == "unknown", f"Expected 'unknown', got {result!r}"


def test_label_scene_black_image_returns_endcard():
    """
    A black frame (all zeros) must return 'graphics/end card' (§21.4).

    Prior behaviour was 'unknown' (image.max() < 5 early return).
    §21.4 changes this: a pure black frame has dark_fraction=1.0 and color_std=0,
    which fires Path 2 of is_graphics_or_endcard(), correctly classifying a
    fade-to-black as a graphics card rather than a real scene location.
    """
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    result = label_scene(black)
    assert result == "graphics/end card", (
        f"Expected 'graphics/end card' for pure black frame (§21.4), got {result!r}"
    )


def test_label_scene_always_returns_str():
    """label_scene() must always return a non-None str in all code paths."""
    test_images = [
        None,
        np.zeros((64, 64, 3), dtype=np.uint8),                         # black -> "graphics/end card" (§21.4)
        np.full((64, 64, 3), 128, dtype=np.uint8),                     # mid-grey
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),        # noise
    ]
    for img in test_images:
        result = label_scene(img)
        assert isinstance(result, str) and result, (
            f"label_scene() returned {result!r} (not a non-empty str) for image {img}"
        )


def test_clip_unavailable_still_returns_str():
    """With CLIP mocked as unavailable, label_scene() must still return a valid str."""
    with mock.patch("aese.adapters.scene_label._load_clip_text_features", return_value=False):
        with mock.patch("aese.adapters.scene_label._clip_labels_loaded", True):
            img = np.random.randint(100, 200, (64, 64, 3), dtype=np.uint8)
            result = label_scene(img)
            assert isinstance(result, str) and result, (
                f"Expected a non-empty str when CLIP is unavailable, got {result!r}"
            )


def test_clip_available_is_bool():
    """_clip_available() must always return a bool, never raise."""
    result = _clip_available()
    assert isinstance(result, bool), f"Expected bool, got {type(result)}"


# ---------------------------------------------------------------------------
# Fix 3 -- expanded vocabulary contract
# ---------------------------------------------------------------------------

_EXPECTED_LABELS = {
    "kitchen", "living room", "bedroom", "office", "hallway",
    "street", "village", "forest", "beach", "outdoor field",
    "vehicle interior", "rooftop", "restaurant", "stage/studio",
    "unknown",
}


def test_scene_labels_has_all_expected_entries():
    """SCENE_LABELS must contain all 15 expected vocabulary entries."""
    assert set(SCENE_LABELS) == _EXPECTED_LABELS, (
        f"SCENE_LABELS mismatch.\n"
        f"  Missing: {_EXPECTED_LABELS - set(SCENE_LABELS)}\n"
        f"  Extra:   {set(SCENE_LABELS) - _EXPECTED_LABELS}"
    )


def test_scene_labels_is_public():
    """SCENE_LABELS must be importable as a public name (no leading underscore)."""
    from aese.adapters.scene_label import SCENE_LABELS as sl
    assert isinstance(sl, list) and len(sl) > 0


def test_scene_labels_contains_unknown():
    """'unknown' must be in SCENE_LABELS as the guaranteed fallback label."""
    assert "unknown" in SCENE_LABELS


def test_label_scene_output_in_vocabulary():
    """
    Any non-None output from label_scene() must be in SCENE_LABELS OR equal to
    "graphics/end card" (the pre-check bypass label added in Fix 3, §20.3).

    "graphics/end card" is intentionally NOT added to SCENE_LABELS -- it is
    returned by is_graphics_or_endcard() before any VLM or CLIP call, so it
    bypasses the model vocabulary entirely.  Flat monochrome images (like
    np.full(200)) legitimately trigger this check.
    """
    _VALID_LABELS = set(SCENE_LABELS) | {"graphics/end card"}
    test_images = [
        np.full((64, 64, 3), 200, dtype=np.uint8),   # bright neutral (may hit endcard check)
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),  # noise
        np.zeros((64, 64, 3), dtype=np.uint8),        # black
    ]
    for img in test_images:
        result = label_scene(img)
        assert result in _VALID_LABELS, (
            f"label_scene() returned {result!r}, which is not in valid label set "
            f"(SCENE_LABELS ∪ {{'graphics/end card'}})"
        )


def test_heuristic_fallback_output_in_vocabulary():
    """The heuristic path must also return a label in SCENE_LABELS."""
    from aese.adapters.scene_label import _heuristic_scene_label
    imgs = [
        np.full((64, 64, 3), 180, dtype=np.uint8),  # warm-ish mid-tone
        np.zeros((64, 64, 3), dtype=np.uint8),       # black
    ]
    for img in imgs:
        result = _heuristic_scene_label(img)
        assert result in SCENE_LABELS, (
            f"_heuristic_scene_label() returned {result!r}, not in SCENE_LABELS"
        )