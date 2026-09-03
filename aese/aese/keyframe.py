"""
aese/keyframe.py
Key frame selection from a list of TemporalFeatures.

Implements §5.10 — five named strategies behind a single interface.
Default changed to "most_salient" (§20) — selects the peak motion+novelty frame,
i.e. the moment most likely to contain a meaningful change, not the calmest one.

Strategies:
  "most_salient"    — frame with highest (motion_score + novelty_score)   [DEFAULT]
  "lowest_blur"     — frame with lowest motion_score (static = sharpest)
                      Use for thumbnail generation where visual clarity matters
                      more than narrative content. NOT the right default for
                      caption/summary pipelines (see DECISIONS.md §20.1).
  "center"          — middle frame in the event window
  "highest_novelty" — frame with highest novelty_score
  "most_motion"     — frame with highest motion_score (most dynamic)
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .types import TemporalFeature


def select_keyframe(
    features: List[TemporalFeature],
    strategy: str = "most_salient",
) -> Optional[np.ndarray]:
    """
    Select a representative 'key frame' embedding from an event's TemporalFeatures.

    In manifest-replay mode, no raw images are available — this function returns
    the representative *embedding vector* of the selected feature instead of
    a pixel image, since pixel data requires the live pipeline.

    Args:
        features: List of TemporalFeatures spanning the event.
        strategy: One of "most_salient" | "lowest_blur" | "center" |
                  "highest_novelty" | "most_motion".
                  Defaults to "most_salient" — picks the peak motion+novelty
                  frame (the most narratively eventful moment), which is the
                  correct default for summary/caption generation.

    Returns:
        Optional[np.ndarray]: Pixel image (live mode) or embedding (replay mode)
        of the selected feature, or None if features is empty.
    """
    if not features:
        return None

    if strategy == "most_salient":
        # Peak (motion_score + novelty_score) — the most narratively eventful
        # moment, chosen for summary/caption generation (DECISIONS.md §20.1).
        salience = [f.motion_score + f.novelty_score for f in features]
        selected = features[int(np.argmax(salience))]

    elif strategy == "lowest_blur":
        # Lowest motion_score proxy for sharpness — good for thumbnail generation
        # where visual clarity matters more than narrative content, but NOT the
        # right default for caption/summary pipelines.
        selected = min(features, key=lambda tf: tf.motion_score)

    elif strategy == "center":
        selected = features[len(features) // 2]

    elif strategy == "highest_novelty":
        selected = max(features, key=lambda tf: tf.novelty_score)

    elif strategy == "most_motion":
        selected = max(features, key=lambda tf: tf.motion_score)

    else:
        raise ValueError(f"Unknown keyframe strategy: {strategy!r}. "
                         "Choose from: most_salient, lowest_blur, center, "
                         "highest_novelty, most_motion")

    # Return the real pixel image if available (live/--video mode).
    # Fall back to the embedding vector as a proxy only in manifest-replay mode
    # where no pixel data was ever extracted.
    if selected.representative_image is not None:
        return selected.representative_image.copy()
    return selected.multimodal_embedding.copy()


def select_keyframe_salient(
    features: List[TemporalFeature],
) -> Tuple[Optional[np.ndarray], float]:
    """
    Select the frame at peak (motion_score + novelty_score) and return both
    the frame and the timestamp_ms of the selected feature.

    Used internally to identify the primary keyframe index for
    needs_secondary_frame() gating in Fix 2.

    Args:
        features: List of TemporalFeatures spanning the event.

    Returns:
        (frame_or_embedding, timestamp_ms):
            frame_or_embedding — pixel image in live mode, embedding in replay mode,
                                 or None if features is empty.
            timestamp_ms — timestamp of the selected feature, or 0.0 if empty.
    """
    if not features:
        return None, 0.0

    salience = [f.motion_score + f.novelty_score for f in features]
    peak_idx = int(np.argmax(salience))
    selected = features[peak_idx]

    frame = (
        selected.representative_image.copy()
        if selected.representative_image is not None
        else selected.multimodal_embedding.copy()
    )
    return frame, selected.timestamp_ms


def needs_secondary_frame(
    features: List[TemporalFeature],
    primary_idx: int,
    duration_s: float,
) -> Optional[int]:
    """
    Decide whether a second VLM call is warranted for a long event with a
    high-salience spike the primary keyframe didn't cover.

    Gate conditions (ALL must be true to fire — deliberately narrow):
      1. Event duration > 15 s
      2. At least one other frame has salience (motion+novelty) > 0.7 in absolute
         terms AND its timestamp is > 8 s away from the primary frame's timestamp.

    This is designed NOT to fire on the vast majority of events.  For a typical
    ~10s dialogue event it always returns None.  For a 26s event with a calm
    first half and a dramatic reveal at second 20 it returns the reveal index —
    the one real case it is designed for (DECISIONS.md §20.2).

    Args:
        features:    List of TemporalFeatures for the event.
        primary_idx: Index of the primary keyframe within features.
        duration_s:  Total event duration in seconds.

    Returns:
        Index into features of the best secondary candidate, or None if the
        gate conditions are not met.
    """
    if duration_s < 15.0 or not features:
        return None

    primary_time = features[primary_idx].timestamp_ms
    salience = [f.motion_score + f.novelty_score for f in features]

    candidates = [
        (i, s)
        for i, s in enumerate(salience)
        if i != primary_idx
        and s > 0.7
        and abs(features[i].timestamp_ms - primary_time) > 8000
    ]
    if not candidates:
        return None

    # Return the candidate with the highest salience score
    return max(candidates, key=lambda c: c[1])[0]
