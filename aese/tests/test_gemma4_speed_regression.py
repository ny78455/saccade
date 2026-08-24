"""
tests/test_gemma4_speed_regression.py
Regression guards for the Gemma-4 call-frequency bug + inference efficiency fixes.

Fixes covered
-------------
Fix 1 (aggregator.py §18):
    label_scene() must be called exactly once per 1-second window regardless of
    how many raw FramePackets land in that second (was: once per packet).

Fix 2 (gemma4.py):
    describe_scene_and_caption() parses the combined SCENE/CAPTION response
    correctly and degrades gracefully on malformed output.

Fix 3 (gemma4.py::_prepare_image):
    _prepare_image() downscales images with longest dimension > 512px; leaves
    images that are already small untouched.

These tests use only mocks and numpy — no model weights or GPU required.

History:
    2026-08-24: Initial version (Gemma-4 call-frequency + efficiency fixes).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aese.types import AESEConfig, FramePacket


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _CallCounter:
    """Side-effect callable that counts invocations and returns a fixed value."""
    def __init__(self, return_value: str = "office") -> None:
        self.count = 0
        self._return = return_value

    def __call__(self, *args, **kwargs) -> str:
        self.count += 1
        return self._return

    def reset(self) -> None:
        self.count = 0


def _make_packet(
    frame_id: int,
    timestamp_ms: float,
    *,
    has_image: bool = True,
    motion_score: float = 0.05,
    audio_energy: float = 0.05,
    novelty_score: float = 0.10,
) -> FramePacket:
    image = np.full((64, 64, 3), 128, dtype=np.uint8) if has_image else None
    return FramePacket(
        frame_id=frame_id,
        timestamp_ms=timestamp_ms,
        fps_used=8.0,
        motion_score=motion_score,
        scene_change=False,
        audio_energy=audio_energy,
        novelty_score=novelty_score,
        decision_reason="test",
        subtitle_text=None,
        image=image,
    )


# ---------------------------------------------------------------------------
# Fix 1 — label_scene called exactly once per second, not once per packet
# ---------------------------------------------------------------------------

def test_scene_label_called_once_per_second_not_per_frame():
    """
    Regression guard for the confirmed aggregator.py bug (DECISIONS.md §18).

    Simulates an 8-fps action window: 8 FramePackets all within the same
    wall-clock second.  label_scene() must be called exactly ONCE regardless of
    packet count.

    Before Fix 1:  call count == 8 (once per packet)
    After  Fix 1:  call count == 1 (representative middle-frame only)
    """
    from aese.aggregator import FeatureAggregator

    counter = _CallCounter(return_value="office")
    config = AESEConfig()
    agg = FeatureAggregator(config)

    # Push 8 packets into second 0 (timestamps 0..875ms at ~8fps)
    for i in range(8):
        agg.push(_make_packet(i, float(i * 125)))

    # Trigger window close by pushing a packet into second 1
    with patch("aese.aggregator.label_scene", side_effect=counter):
        agg.push(_make_packet(8, 1000.0))

    assert counter.count == 1, (
        f"label_scene() was called {counter.count} times for 8 packets in the same "
        f"second.  Expected exactly 1 (representative frame).  "
        f"Fix 1 in aggregator.py may be broken."
    )


def test_scene_label_called_once_with_single_packet():
    """
    Edge case: single packet in window.  Still exactly 1 call.
    (Middle-frame index of a 1-element list is index 0 - no off-by-one.)
    """
    from aese.aggregator import FeatureAggregator

    counter = _CallCounter(return_value="kitchen")
    config = AESEConfig()
    agg = FeatureAggregator(config)

    agg.push(_make_packet(0, 0.0))  # buffered in second 0

    with patch("aese.aggregator.label_scene", side_effect=counter):
        agg.push(_make_packet(1, 1000.0))  # closes second 0

    assert counter.count == 1, (
        f"label_scene() was called {counter.count} times for a single-packet window."
    )


def test_scene_label_not_called_when_no_real_images():
    """
    If all packets in a window lack real pixel data (image=None), label_scene()
    must never be called and scene_label must be 'unknown'.
    """
    from aese.aggregator import FeatureAggregator

    counter = _CallCounter()
    config = AESEConfig()
    agg = FeatureAggregator(config)

    # 3 packets with no image
    for i in range(3):
        agg.push(_make_packet(i, float(i * 200), has_image=False))

    with patch("aese.aggregator.label_scene", side_effect=counter):
        tfs = agg.flush()

    assert counter.count == 0, (
        f"label_scene() was called {counter.count} times despite no real images in window."
    )
    assert tfs[0].scene_label == "unknown", (
        f"Expected scene_label='unknown' with no real images, got {tfs[0].scene_label!r}"
    )


# ---------------------------------------------------------------------------
# Fix 2 — _parse_scene_and_caption: correct parse + graceful fallback
# ---------------------------------------------------------------------------

def test_combined_scene_and_caption_parses_correctly():
    """
    Happy-path: a well-formed SCENE/CAPTION response is parsed correctly.
    """
    from aese.adapters.gemma4 import _parse_scene_and_caption
    from aese.adapters.scene_label import SCENE_LABELS

    raw = 'SCENE: "kitchen"\nCAPTION: Two chefs argue over a boiling pot while the kitchen fills with steam.'
    scene, caption = _parse_scene_and_caption(raw, fallback_labels=SCENE_LABELS)

    assert scene == "kitchen", f"Expected scene='kitchen', got {scene!r}"
    assert "chefs" in caption, f"Caption does not contain expected content: {caption!r}"


def test_combined_scene_and_caption_case_insensitive():
    """
    SCENE/CAPTION keys are matched case-insensitively (model may vary casing).
    """
    from aese.adapters.gemma4 import _parse_scene_and_caption
    from aese.adapters.scene_label import SCENE_LABELS

    raw = "scene: office\ncaption: A detective studies evidence spread across a desk."
    scene, caption = _parse_scene_and_caption(raw, fallback_labels=SCENE_LABELS)

    assert scene == "office"
    assert "detective" in caption


def test_combined_scene_and_caption_malformed_falls_back():
    """
    Malformed response (missing CAPTION line): must return ("unknown", "") without
    raising, rather than crashing.  The caller's template-summary fallback handles "".
    """
    from aese.adapters.gemma4 import _parse_scene_and_caption
    from aese.adapters.scene_label import SCENE_LABELS

    # Missing CAPTION line entirely
    raw_no_caption = "SCENE: beach"
    scene, caption = _parse_scene_and_caption(raw_no_caption, fallback_labels=SCENE_LABELS)
    assert scene == "beach"
    assert caption == "", f"Expected empty caption, got {caption!r}"

    # Missing SCENE line entirely
    raw_no_scene = "CAPTION: Waves crash against a rocky shore under a grey sky."
    scene2, caption2 = _parse_scene_and_caption(raw_no_scene, fallback_labels=SCENE_LABELS)
    assert scene2 == "unknown", f"Expected 'unknown' when SCENE missing, got {scene2!r}"
    assert "Waves" in caption2

    # Completely empty response
    scene3, caption3 = _parse_scene_and_caption("", fallback_labels=SCENE_LABELS)
    assert scene3 == "unknown"
    assert caption3 == ""


def test_combined_scene_label_not_in_vocabulary_falls_back_to_unknown():
    """
    If the model returns a SCENE label not in the vocabulary, it must degrade to
    'unknown' rather than propagating a free-text hallucination.
    """
    from aese.adapters.gemma4 import _parse_scene_and_caption
    from aese.adapters.scene_label import SCENE_LABELS

    raw = "SCENE: spaceship interior\nCAPTION: Astronauts float weightlessly."
    scene, caption = _parse_scene_and_caption(raw, fallback_labels=SCENE_LABELS)
    assert scene == "unknown", (
        f"Out-of-vocabulary scene label should fall back to 'unknown', got {scene!r}"
    )
    assert "Astronauts" in caption


# ---------------------------------------------------------------------------
# Fix 3 — _prepare_image: downscales large frames, preserves small ones
# ---------------------------------------------------------------------------

def test_prepare_image_downscales_large_frame():
    """
    A 1920x1080 synthetic image must be downscaled so its longest dimension
    is <= _MAX_DIM (512px).
    """
    from aese.adapters.gemma4 import _prepare_image, _MAX_DIM

    large = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
    pil = _prepare_image(large)
    assert max(pil.size) <= _MAX_DIM, (
        f"_prepare_image() produced {pil.size}, longest dim should be <= {_MAX_DIM}"
    )


def test_prepare_image_preserves_aspect_ratio():
    """
    After downscaling a 1920x1080 image the output must be 512x288 (or as close
    as integer rounding allows), not 512x512.
    """
    from aese.adapters.gemma4 import _prepare_image, _MAX_DIM

    large = np.zeros((1080, 1920, 3), dtype=np.uint8)
    pil = _prepare_image(large)
    w, h = pil.size
    # 1920 -> 512, so 1080 -> 1080*(512/1920) = 288
    assert w == 512, f"Expected width=512, got {w}"
    assert h == 288, f"Expected height=288, got {h}"


def test_prepare_image_leaves_small_frame_unchanged():
    """
    An image already smaller than _MAX_DIM must not be resized (no upscaling).
    """
    from aese.adapters.gemma4 import _prepare_image, _MAX_DIM

    small = np.zeros((64, 64, 3), dtype=np.uint8)
    pil = _prepare_image(small)
    assert pil.size == (64, 64), (
        f"Small image should not be resized, got {pil.size}"
    )


def test_prepare_image_exact_boundary_unchanged():
    """
    An image whose longest dimension equals exactly _MAX_DIM must not be resized.
    """
    from aese.adapters.gemma4 import _prepare_image, _MAX_DIM

    img = np.zeros((_MAX_DIM, _MAX_DIM // 2, 3), dtype=np.uint8)  # e.g. 512x256
    pil = _prepare_image(img)
    assert pil.size == (_MAX_DIM // 2, _MAX_DIM), (
        f"Image at exactly _MAX_DIM should be unchanged, got {pil.size}"
    )
