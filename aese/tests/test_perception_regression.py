"""
tests/test_perception_regression.py
Regression guards for the four perception failures identified on the
Andhadhun clip stress test. See DECISIONS.md §20.

Fixes covered
-------------
Fix 1 (keyframe.py §20.1):
    Salience-based keyframe selection picks the motion+novelty peak, not the
    calmest/sharpest frame (lowest_blur default was the root cause of the
    missed-gunman failure in Event 9).

Fix 2 (keyframe.py, pipeline.py §20.2):
    Gated secondary keyframe for long events with a high-salience spike that
    the primary frame didn't cover. Gate conditions are deliberately narrow:
    duration > 15s AND salience > 0.7 AND spike > 8s away from primary.
    Returns None for short events and events with no qualifying distant spike.

Fix 3 (scene_label.py §20.3):
    Deterministic graphics/end-card detection runs before any VLM/CLIP call.
    Flat background + low edge density → "graphics/end card" (no model cost).
    Verified NOT to fire on legitimately dark/detailed film frames.

Fix 5 (caption_person_count.py, pipeline.py §20.5):
    Caption-text person-count reconciliation raises max_characters_seen when
    the VLM's own text describes more people than the face detector saw.
    Never lowers a detector-confirmed count.

All tests use only mocks and numpy -- no model weights or GPU required.

History:
    2026-09-03: Initial version (perception fixes §20).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aese.types import TemporalFeature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(size: int = 32) -> np.ndarray:
    return np.zeros(size, dtype=np.float32)


def _make_feature(
    timestamp_ms: float,
    motion_score: float = 0.1,
    novelty_score: float = 0.1,
    image: np.ndarray | None = None,
) -> TemporalFeature:
    """Create a minimal TemporalFeature for keyframe selection tests."""
    return TemporalFeature(
        timestamp_ms=timestamp_ms,
        scene_label="unknown",
        character_count=0,
        action_label="static",
        dialogue_present=False,
        dialogue_text=None,
        camera_cue=None,
        music_mood="calm",
        multimodal_embedding=_make_embedding(),
        motion_score=motion_score,
        novelty_score=novelty_score,
        representative_image=image,
    )


# ---------------------------------------------------------------------------
# Fix 1 — Salience-based keyframe selection (DECISIONS.md §20.1)
# ---------------------------------------------------------------------------

def test_salient_keyframe_picks_motion_novelty_peak():
    """
    A synthetic 6-feature event where the salience spike (motion+novelty=1.5)
    is at index 4, not the center (index 3) or lowest-motion (index 0).
    select_keyframe("most_salient") must return the frame at index 4.
    """
    from aese.keyframe import select_keyframe

    # Create 6 synthetic frames with distinct pixel values so we can identify
    # which one was returned
    frames = [np.full((8, 8, 3), i * 40, dtype=np.uint8) for i in range(6)]
    features = [
        _make_feature(float(i * 1000), motion_score=0.05, novelty_score=0.05, image=frames[i])
        for i in range(6)
    ]
    # Spike at index 4: motion=0.9 + novelty=0.6 = 1.5 (highest)
    features[4] = _make_feature(4000.0, motion_score=0.9, novelty_score=0.6, image=frames[4])

    result = select_keyframe(features, strategy="most_salient")

    # The result should be frame[4] (all pixels = 4*40 = 160)
    assert result is not None, "select_keyframe returned None unexpectedly"
    assert result.shape == (8, 8, 3)
    assert result[0, 0, 0] == 160, (
        f"Expected frame at spike index (pixel=160), got pixel={result[0,0,0]}. "
        f"most_salient must pick peak motion+novelty, not center or lowest-motion."
    )


def test_salient_keyframe_not_lowest_blur():
    """
    Confirm most_salient does NOT pick the lowest-motion frame (the old default).
    """
    from aese.keyframe import select_keyframe

    frames = [np.full((8, 8, 3), i * 30, dtype=np.uint8) for i in range(5)]
    features = [
        _make_feature(float(i * 1000), motion_score=float(i) * 0.2, novelty_score=0.1,
                      image=frames[i])
        for i in range(5)
    ]
    # Lowest motion = index 0 (motion=0.0); highest salience = index 4 (motion=0.8+novelty=0.1)
    most_salient = select_keyframe(features, strategy="most_salient")
    lowest_blur   = select_keyframe(features, strategy="lowest_blur")

    assert most_salient is not None
    assert lowest_blur is not None
    # They should NOT be the same frame (different pixels)
    assert not np.array_equal(most_salient, lowest_blur), (
        "most_salient and lowest_blur returned the same frame — "
        "most_salient must pick the high-motion peak, not the static frame."
    )


def test_default_strategy_is_most_salient():
    """
    Calling select_keyframe() without a strategy argument must use most_salient
    (the new default after §20.1 fix, changed from lowest_blur).
    """
    from aese.keyframe import select_keyframe

    frames = [np.full((8, 8, 3), i * 40, dtype=np.uint8) for i in range(4)]
    features = [
        _make_feature(float(i * 1000), motion_score=float(i) * 0.3, novelty_score=0.1,
                      image=frames[i])
        for i in range(4)
    ]
    # index 3 has highest salience
    default_result = select_keyframe(features)
    salient_result = select_keyframe(features, strategy="most_salient")

    assert default_result is not None
    assert salient_result is not None
    assert np.array_equal(default_result, salient_result), (
        "Default strategy must be most_salient. "
        "If this fails, the default= argument in select_keyframe() was not updated."
    )


# ---------------------------------------------------------------------------
# Fix 2 — Gated secondary keyframe (DECISIONS.md §20.2)
# ---------------------------------------------------------------------------

def _make_26s_event_with_late_spike():
    """
    26-second synthetic event:
    - Seconds 0-17: calm (motion=0.05, novelty=0.05)
    - Second 20: high-salience spike (motion=0.8, novelty=0.6)
    - Seconds 21-25: calm again
    Primary keyframe will be at second 20 (highest salience).
    Secondary frame should be None — the primary IS the spike.

    Actually for needs_secondary_frame() to fire we need the PRIMARY to be
    the calm section AND a spike elsewhere that is far enough away.
    Let's put primary salience at t=0 (calm) and spike at t=20s.
    """
    features = []
    for i in range(26):
        motion = 0.05
        novelty = 0.05
        if i == 20:
            motion = 0.85
            novelty = 0.65  # salience = 1.5 >> threshold 0.7
        features.append(_make_feature(float(i * 1000), motion_score=motion, novelty_score=novelty))
    return features


def test_needs_secondary_long_event_with_spike():
    """
    A 26s event where the salience peak (t=20s) is far from the primary keyframe.
    We force the primary to be the calm frame at t=0 (primary_idx=0) so the spike
    at t=20s is > 8s away and above the 0.7 threshold.
    needs_secondary_frame() must return the index of the spike (20).
    """
    from aese.keyframe import needs_secondary_frame

    features = _make_26s_event_with_late_spike()
    # Force primary_idx=0 (calm frame), duration=26s
    secondary_idx = needs_secondary_frame(features, primary_idx=0, duration_s=26.0)

    assert secondary_idx is not None, (
        "needs_secondary_frame() returned None for a 26s event with a "
        "high-salience spike at t=20s that is 20s away from primary. "
        "The gate should have fired."
    )
    assert secondary_idx == 20, (
        f"Expected secondary_idx=20 (the spike), got {secondary_idx}."
    )


def test_needs_secondary_short_event_returns_none():
    """
    A 10s event — gate condition 1 (duration > 15s) is NOT met.
    must always return None regardless of salience.
    """
    from aese.keyframe import needs_secondary_frame

    features = [
        _make_feature(float(i * 1000), motion_score=0.9, novelty_score=0.9)
        for i in range(10)
    ]
    result = needs_secondary_frame(features, primary_idx=0, duration_s=10.0)
    assert result is None, (
        f"needs_secondary_frame() returned {result} for a 10s event. "
        f"Gate condition 1 (duration > 15s) must prevent it from firing."
    )


def test_needs_secondary_no_qualifying_distant_spike_returns_none():
    """
    A 20s event with a high-salience spike at t=5s (only 5s from primary at t=0).
    The spike fails the >8s distance condition, so gate must stay closed.
    """
    from aese.keyframe import needs_secondary_frame

    features = []
    for i in range(20):
        motion = 0.05
        novelty = 0.05
        if i == 5:
            motion = 0.9
            novelty = 0.9  # salience = 1.8, but only 5s from primary
        features.append(_make_feature(float(i * 1000), motion_score=motion, novelty_score=novelty))

    # primary_idx=0 at t=0, spike at t=5000ms (5s < 8s threshold)
    result = needs_secondary_frame(features, primary_idx=0, duration_s=20.0)
    assert result is None, (
        f"needs_secondary_frame() returned {result} for a spike only 5s from "
        f"primary. The >8s distance gate must prevent it from firing."
    )


def test_needs_secondary_below_salience_threshold_returns_none():
    """
    A 20s event with a spike at t=15s (far enough) but salience = 0.5 (< 0.7 threshold).
    Gate must stay closed — low absolute salience is not worth an extra VLM call.
    """
    from aese.keyframe import needs_secondary_frame

    features = []
    for i in range(20):
        motion = 0.05
        novelty = 0.05
        if i == 15:
            motion = 0.3
            novelty = 0.2  # salience = 0.5 < threshold 0.7
        features.append(_make_feature(float(i * 1000), motion_score=motion, novelty_score=novelty))

    result = needs_secondary_frame(features, primary_idx=0, duration_s=20.0)
    assert result is None, (
        f"needs_secondary_frame() returned {result} for a salience=0.5 spike. "
        f"Absolute salience > 0.7 is required to trigger the gate."
    )


# ---------------------------------------------------------------------------
# Fix 3 — Deterministic graphics/end-card detection (DECISIONS.md §20.3)
# ---------------------------------------------------------------------------

def _make_endcard_frame(h=64, w=64) -> np.ndarray:
    """
    Synthetic end-card: near-black background with a subtle dark-grey logo block.
    Characteristic of flat title/end-card frames:
      - Very low color variance (near-monochrome)
      - Very low edge density (no real detail outside the minimal logo mark)

    Uses a subtle logo (not high-contrast white-on-black) to match real end-cards
    like 'THE END' on grey, a watermark, or a faint production company card.
    High-contrast logo-on-black is a different class (handled by the VLM).
    """
    img = np.full((h, w, 3), 5, dtype=np.uint8)  # near-black background
    # Subtle logo block: slightly lighter than background (delta ~15 gray levels)
    cy, cx = h // 2, w // 2
    img[cy-4:cy+4, cx-10:cx+10, :] = 20  # faint logo mark
    return img


def _make_dark_film_frame(h=64, w=64) -> np.ndarray:
    """
    Realistic dark film frame: low overall brightness but with genuine edge
    structure from actors, furniture, and architectural elements.
    Represents a night scene or dimly lit interior.

    Edge density must be comfortably above the 0.015 threshold even at low
    brightness -- real scenes have spatial structure that flat cards lack.
    """
    rng = np.random.default_rng(seed=42)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Multiple distinct regions creating hard edges (actor silhouettes, furniture)
    img[10:30, 5:25, :] = rng.integers(20, 60, (20, 20, 3)).astype(np.uint8)
    img[15:40, 30:55, :] = rng.integers(15, 50, (25, 25, 3)).astype(np.uint8)
    img[45:60, 10:50, :] = rng.integers(10, 40, (15, 40, 3)).astype(np.uint8)
    img[0:5, :, :] = rng.integers(5, 20, (5, w, 3)).astype(np.uint8)  # ceiling
    return img


def test_endcard_detected_without_vlm_call():
    """
    Synthetic end-card frame (flat dark bg + subtle logo) must be classified as
    "graphics/end card" by label_scene() without invoking the VLM or CLIP.
    The pre-check short-circuits before label_scene() even reaches the VLM import.
    """
    from aese.adapters.scene_label import label_scene, is_graphics_or_endcard

    frame = _make_endcard_frame()

    # 1. Confirm the pre-check itself fires on the synthetic end-card
    assert is_graphics_or_endcard(frame), (
        "is_graphics_or_endcard() returned False for a synthetic end-card frame. "
        "Check the color_std and edge_density thresholds."
    )

    # 2. Confirm label_scene() returns "graphics/end card"
    #    The pre-check short-circuits before any VLM/CLIP call.
    #    We verify by patching the vlm_router at the source to detect any call.
    vlm_call_count = [0]

    def _counting_describe_scene(image):
        vlm_call_count[0] += 1
        return "office"

    # Patch at the vlm_router module (where describe_scene is defined)
    with patch("aese.adapters.vlm_router.describe_scene", side_effect=_counting_describe_scene):
        result = label_scene(frame)

    assert result == "graphics/end card", (
        f"Expected 'graphics/end card' for synthetic end-card, got {result!r}. "
        f"The pre-check must short-circuit before any VLM or CLIP call."
    )
    assert vlm_call_count[0] == 0, (
        f"VLM describe_scene was called {vlm_call_count[0]} time(s) for an end-card frame. "
        f"The is_graphics_or_endcard() pre-check must bypass the VLM entirely."
    )


def test_dark_film_frame_not_classified_as_endcard():
    """
    A dark but detailed synthetic film frame must NOT trigger the graphics
    pre-check. Dark scenes have real edge structure; end-cards do not.
    """
    from aese.adapters.scene_label import is_graphics_or_endcard

    frame = _make_dark_film_frame()
    result = is_graphics_or_endcard(frame)

    assert result is False, (
        f"is_graphics_or_endcard() returned True for a dark film frame. "
        f"The heuristic is over-firing — tune the edge_density threshold "
        f"(currently 0.015) against real dark-but-detailed frames."
    )


def test_pure_black_frame_not_endcard():
    """
    A pure black frame (max pixel = 0) must be caught by the existing
    image.max() < 5 guard and return "unknown", not "graphics/end card".
    Pure black is a technical artifact (fade-to-black), not an end-card.
    """
    from aese.adapters.scene_label import label_scene

    black = np.zeros((64, 64, 3), dtype=np.uint8)
    result = label_scene(black)
    assert result == "unknown", (
        f"Pure black frame should return 'unknown', got {result!r}. "
        f"Fade-to-black is not an end-card."
    )


# ---------------------------------------------------------------------------
# Fix 5 — Caption-text person-count reconciliation (DECISIONS.md §20.5)
# ---------------------------------------------------------------------------

def test_caption_raises_count_when_face_detector_missed():
    """
    Event with max_characters_seen=1 (face detector saw one person) but the
    VLM caption describes "a man who is partially visible on the right" alongside
    an already-visible person → caption estimate = 2 → count raised to 2.
    """
    from aese.adapters.caption_person_count import estimate_person_count_from_caption

    summary = (
        "A woman with dark hair gestures while speaking to "
        "a man who is partially visible on the right."
    )
    estimate = estimate_person_count_from_caption(summary)

    assert estimate is not None, "Expected a non-None estimate for a two-person caption."
    assert estimate >= 2, (
        f"Expected count >= 2 (woman + partially-visible man), got {estimate}. "
        f"Caption reconciliation must detect the second person."
    )


def test_caption_explicit_two_people():
    """
    Caption contains "two people" → estimate must return exactly 2.
    """
    from aese.adapters.caption_person_count import estimate_person_count_from_caption

    summary = "Two people stand in a brightly lit hallway, facing each other."
    estimate = estimate_person_count_from_caption(summary)

    assert estimate == 2, (
        f"Expected estimate=2 for 'two people', got {estimate}."
    )


def test_caption_explicit_three_individuals():
    """
    Caption contains "three individuals" → estimate must return exactly 3.
    """
    from aese.adapters.caption_person_count import estimate_person_count_from_caption

    summary = "Three individuals are seated around a round table in a restaurant."
    estimate = estimate_person_count_from_caption(summary)

    assert estimate == 3, (
        f"Expected estimate=3 for 'three individuals', got {estimate}."
    )


def test_caption_no_people_returns_none():
    """
    Caption with no person mentions → estimate must return None (not 0).
    None means "no signal", not "zero people observed".
    """
    from aese.adapters.caption_person_count import estimate_person_count_from_caption

    summary = "A blue armchair is visible in a room with patterned wallpaper."
    estimate = estimate_person_count_from_caption(summary)

    assert estimate is None, (
        f"Expected None for a caption with no person mentions, got {estimate}. "
        f"None means 'no signal' — never return 0 from the caption estimator."
    )


def test_caption_never_lowers_detector_count():
    """
    When the caption describes no people but the face detector found 2,
    the caption estimate (None) must not lower max_characters_seen.
    Simulated via the estimate_person_count_from_caption return value.
    """
    from aese.adapters.caption_person_count import estimate_person_count_from_caption

    # Caption with no person mentions
    summary = "A grand piano sits in the corner of a warmly decorated room."
    estimate = estimate_person_count_from_caption(summary)

    # Simulate the reconciliation logic in pipeline._finalize_event()
    detector_count = 2
    if estimate is not None:
        reconciled = max(detector_count, estimate)
    else:
        reconciled = detector_count  # estimate is absent — do not lower

    assert reconciled == 2, (
        f"Detector count of 2 was lowered to {reconciled} by an absent caption "
        f"signal. The reconciliation must never lower a detector-confirmed count."
    )


def test_caption_empty_string_returns_none():
    """
    Empty summary → None (no signal). Defensive edge case.
    """
    from aese.adapters.caption_person_count import estimate_person_count_from_caption

    assert estimate_person_count_from_caption("") is None
    assert estimate_person_count_from_caption(None) is None
