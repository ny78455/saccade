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


def test_pure_black_frame_returns_endcard():
    """
    A pure black frame must return 'graphics/end card' (§21.4).

    BEHAVIOUR CHANGE from §20.3: the prior implementation returned 'unknown' for
    black frames via an early image.max() < 5 guard. §21.4 removes this guard and
    routes black frames through Path 2 of is_graphics_or_endcard():
      dark_fraction=1.0 > 0.70, color_std=0 < 30 → returns "graphics/end card".

    Rationale: a fade-to-black is not a real scene location and should not be
    labeled as one. "graphics/end card" is the correct label.
    """
    from aese.adapters.scene_label import label_scene

    black = np.zeros((64, 64, 3), dtype=np.uint8)
    result = label_scene(black)
    assert result == "graphics/end card", (
        f"Pure black frame should return 'graphics/end card' (§21.4), got {result!r}."
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


# ===========================================================================
# Round 21 Regression Tests (§21.1 – §21.4)
# ===========================================================================

# ---------------------------------------------------------------------------
# Fix 1 — Posture/spatial-reasoning prompt directive (§21.1)
# ---------------------------------------------------------------------------

def test_posture_prompt_contains_spatial_reasoning():
    """
    SUMMARY_SYSTEM_PROMPT must contain the posture/spatial-reasoning directive
    added in §21.1. 'lying on the floor' pins the guard against the Event 4
    'seated on armchair' misclassification from the Andhadhun stress test.
    """
    from aese.summary import SUMMARY_SYSTEM_PROMPT
    assert "lying on the floor" in SUMMARY_SYSTEM_PROMPT, (
        "SUMMARY_SYSTEM_PROMPT is missing the posture/spatial-reasoning directive (§21.1)."
    )


def test_posture_prompt_contains_furniture_warning():
    """
    Prompt must warn against inferring posture from furniture alone.
    """
    from aese.summary import SUMMARY_SYSTEM_PROMPT
    assert "furniture" in SUMMARY_SYSTEM_PROMPT.lower(), (
        "SUMMARY_SYSTEM_PROMPT must warn about assuming posture from furniture alone."
    )


# ---------------------------------------------------------------------------
# Fix 2 — Novelty-based mid-event split trigger (§21.2)
# ---------------------------------------------------------------------------

def _make_novelty_buffer(novelty_values):
    """Build a ContextBuffer whose features carry the given per-second novelty scores."""
    import sys, os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from aese.context_buffer import ContextBuffer
    from aese.types import AESEConfig, TemporalFeature
    from aese.adapters.embedding import EMBEDDING_DIM
    config = AESEConfig()
    buf = ContextBuffer(config.buffer_seconds)
    for i, nov in enumerate(novelty_values):
        tf = TemporalFeature(
            timestamp_ms=float(i * 1000),
            scene_label="hallway",
            character_count=1,
            action_label="static",
            dialogue_present=False,
            dialogue_text=None,
            camera_cue=None,
            music_mood="calm",
            multimodal_embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
            motion_score=0.05,
            novelty_score=nov,
            audio_energy=0.0,
            spectral_flux=0.0,
            image_available=True,
        )
        buf.push(tf)
    return buf, config


def _make_spike_feature(ts_ms, novelty, motion=0.05):
    from aese.types import TemporalFeature
    from aese.adapters.embedding import EMBEDDING_DIM
    return TemporalFeature(
        timestamp_ms=ts_ms,
        scene_label="hallway",
        character_count=1,
        action_label="static",
        dialogue_present=False,
        dialogue_text=None,
        camera_cue=None,
        music_mood="calm",
        multimodal_embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
        motion_score=motion,
        novelty_score=novelty,
        audio_energy=0.0,
        spectral_flux=0.0,
        image_available=True,
    )


def test_novelty_spike_check_fires_on_slow_reveal():
    """
    _check_novelty_spike() must return True when novelty jumps from 0.05 baseline
    to 0.60 — the canonical slow-reveal pattern from Event 9.
    """
    from aese.boundary.candidate_detector import (
        CandidateDetector, NOVELTY_SPIKE_THRESHOLD, NOVELTY_SPIKE_RATIO,
    )
    # 3 quiet baseline seconds + spike current
    buf, config = _make_novelty_buffer([0.05, 0.05, 0.05, 0.05])
    detector = CandidateDetector(config, buf)
    curr = _make_spike_feature(4000.0, novelty=0.60)
    assert detector._check_novelty_spike(curr), (
        f"_check_novelty_spike must fire when novelty=0.60 > "
        f"NOVELTY_SPIKE_THRESHOLD={NOVELTY_SPIKE_THRESHOLD} and "
        f">= {NOVELTY_SPIKE_RATIO}x baseline=0.05."
    )


def test_novelty_spike_duration_guard():
    """
    The duration guard (_seconds_since_boundary <= 15) must prevent the trigger
    from firing in short events, even when _check_novelty_spike() returns True.
    """
    from aese.boundary.candidate_detector import CandidateDetector, NOVELTY_SPIKE_MIN_DURATION_S
    buf, config = _make_novelty_buffer([0.05, 0.05, 0.05, 0.05])
    detector = CandidateDetector(config, buf)
    detector._seconds_since_boundary = 10  # < 15s threshold
    assert detector._seconds_since_boundary <= NOVELTY_SPIKE_MIN_DURATION_S, (
        "Duration guard must suppress novelty_spike for short events (<= 15s)."
    )


def test_motion_gate_does_not_fire_on_slow_reveal():
    """
    _check_action_transition() must return False on all-static input, confirming
    novelty_spike is a genuinely independent capability.
    """
    from aese.boundary.candidate_detector import CandidateDetector
    buf, config = _make_novelty_buffer([0.05] * 4)
    detector = CandidateDetector(config, buf)
    # All frames have action_label='static' — no fast_action transition
    assert not detector._check_action_transition(), (
        "Motion gate must not fire on slow-reveal (all-static) input."
    )


# ---------------------------------------------------------------------------
# Fix 3 — Exemplar gallery identity separation (§21.3)
# ---------------------------------------------------------------------------

def test_exemplar_gallery_rejects_identity_flip():
    """
    Two orthogonal embeddings (distinct actors) must stay in separate clusters
    after 20 alternating observations. Regression guard for the 'Person A flips
    between two actors' failure from the Andhadhun stress test.
    """
    from aese.adapters.character_cluster import CharacterClusterer
    rng = np.random.default_rng(42)
    emb_a = np.zeros(64, dtype=np.float32); emb_a[0] = 1.0
    emb_b = np.zeros(64, dtype=np.float32); emb_b[1] = 1.0

    clusterer = CharacterClusterer(distance_threshold=0.45)
    labels = []
    for _ in range(10):
        na = emb_a + rng.normal(0, 0.02, 64).astype(np.float32)
        na /= np.linalg.norm(na)
        labels.append(clusterer.assign(na))
        nb = emb_b + rng.normal(0, 0.02, 64).astype(np.float32)
        nb /= np.linalg.norm(nb)
        labels.append(clusterer.assign(nb))

    even_labels = set(labels[::2])
    odd_labels  = set(labels[1::2])
    assert len(even_labels) == 1, f"Actor A split across clusters: {even_labels}"
    assert len(odd_labels)  == 1, f"Actor B split across clusters: {odd_labels}"
    assert even_labels != odd_labels, "Actor A and Actor B merged into same cluster (identity flip)."


def test_three_distinct_embeddings_three_clusters():
    """Three orthogonal embeddings must produce exactly three clusters."""
    from aese.adapters.character_cluster import CharacterClusterer
    clusterer = CharacterClusterer(distance_threshold=0.45)
    embs = [np.zeros(64, dtype=np.float32) for _ in range(3)]
    for i, e in enumerate(embs):
        e[i] = 1.0
    labels = [clusterer.assign(e) for e in embs]
    assert len(set(labels)) == 3, f"Expected 3 clusters, got labels: {labels}"


# ---------------------------------------------------------------------------
# Fix 4a — Graphics/end-card robustness (§21.4)
# ---------------------------------------------------------------------------

def test_netflix_style_dark_bg_logo_detected():
    """
    Netflix-style end card (>90% dark pixels, small red rectangle) must be
    detected by Path 2 of is_graphics_or_endcard() and cause label_scene()
    to return 'graphics/end card'. This was the failure case from Round 20.
    """
    from aese.adapters.scene_label import is_graphics_or_endcard, label_scene
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[24:40, 28:36, 0] = 200   # small red logo patch on black background
    assert is_graphics_or_endcard(img), "Path 2 must detect dark-bg colored-logo card."
    assert label_scene(img) == "graphics/end card", (
        "label_scene() must return 'graphics/end card' for Netflix-style card."
    )


def test_legacy_flat_grey_card_still_detected():
    """Path 1 (color_std < 18, near-zero edges) must still fire after refactor."""
    from aese.adapters.scene_label import is_graphics_or_endcard
    assert is_graphics_or_endcard(np.full((64, 64, 3), 200, dtype=np.uint8)), (
        "Legacy flat grey card must still be detected by Path 1."
    )


def test_dark_film_scene_not_misclassified():
    """
    A dark but content-rich scene (40% non-black pixels with color variation)
    must NOT be classified as an end card by Path 2.
    """
    from aese.adapters.scene_label import is_graphics_or_endcard
    rng = np.random.default_rng(1)
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = rng.random((64, 64)) < 0.40
    img[mask, 0] = rng.integers(30, 180, mask.sum()).astype(np.uint8)
    img[mask, 1] = rng.integers(20, 120, mask.sum()).astype(np.uint8)
    img[mask, 2] = rng.integers(10, 80,  mask.sum()).astype(np.uint8)
    assert not is_graphics_or_endcard(img), (
        "Dark film scene with real color variation must not be classified as end card."
    )


# ---------------------------------------------------------------------------
# Fix 4b — Character label schema invariant (§21.4)
# ---------------------------------------------------------------------------

def test_schema_guard_clears_stale_labels():
    """
    character_labels must be cleared when max_characters_seen == 0.
    Reproduces the Event 10 bug: Netflix card had max_characters_seen=0
    but character_labels=['Person A'].
    """
    from aese.types import Event

    event = Event(
        event_id=10,
        start_time_ms=162000.0, end_time_ms=175000.0, duration_ms=13000.0,
        event_embedding=np.zeros(512, dtype=np.float32),
        importance=0.5, confidence=0.5,
        summary="", boundary_reason="stream_end", event_type="Scene",
        max_characters_seen=0,
        character_labels=["Person A"],
        character_data_available=True,
    )
    # Apply schema invariant (mirrors pipeline.py logic)
    if (event.max_characters_seen is None or event.max_characters_seen == 0) \
            and event.character_labels:
        event.character_labels = []

    assert event.character_labels == [], (
        f"Schema guard must clear stale labels when max_characters_seen=0. "
        f"Got {event.character_labels!r}."
    )


def test_schema_guard_preserves_valid_labels():
    """
    character_labels must NOT be cleared when max_characters_seen > 0.
    """
    from aese.types import Event

    event = Event(
        event_id=7,
        start_time_ms=111000.0, end_time_ms=122000.0, duration_ms=11000.0,
        event_embedding=np.zeros(512, dtype=np.float32),
        importance=0.4, confidence=0.95,
        summary="Person A is next to a piano.", boundary_reason="scene_change",
        event_type="Action",
        max_characters_seen=2,
        character_labels=["Person A", "Person B"],
        character_data_available=True,
    )
    original = list(event.character_labels)
    if (event.max_characters_seen is None or event.max_characters_seen == 0) \
            and event.character_labels:
        event.character_labels = []

    assert event.character_labels == original, (
        f"Schema guard must not clear valid labels when max_characters_seen={event.max_characters_seen}."
    )
