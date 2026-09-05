"""
aese/boundary/candidate_detector.py
Boundary Candidate Detector — §5.7, §16.

Every second, computes all boundary signals and fuses them into a score.
Implements the 2-second confidence hold to prevent boundary flapping:

  If fused_score crosses boundary_threshold but confidence is low (score near
  the margin), the detector holds the decision for up to 2 more seconds, collecting
  fresh signals before committing. This prevents spurious boundaries from
  momentary threshold crossings.

Non-functional requirements:
  - Never holds more than 2000ms before committing (§8)
  - No flapping on scores that oscillate around the threshold
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple

import numpy as np

from ..context_buffer import ContextBuffer
from ..types import AESEConfig, BoundaryDecision, BoundarySignal, TemporalFeature
from .confidence import compute_confidence, is_high_confidence
from .embedding_change import embedding_distance
from .fusion import dominant_signal_name, fuse
from .prediction_error import compute_prediction_error
from .signals import (
    camera_signal,
    character_signal,
    dialogue_signal,
    emotion_signal,
    music_signal,
    scene_signal,
)

logger = logging.getLogger(__name__)

_MAX_HOLD_MS = 2000.0  # Non-functional requirement: max decision delay

# Novelty-spike split trigger (Fix 2 — DECISIONS.md §21.2)
# Fires when novelty jumps sharply within a long, already-open event.
# Deliberately decoupled from the motion-based triggers:
#   - slow reveals don't produce motion spikes, only novelty spikes
#   - this is a semantically direct signal for "something new appeared in frame"
NOVELTY_SPIKE_THRESHOLD = 0.35  # absolute floor: novelty must be at least this
NOVELTY_SPIKE_RATIO = 2.5       # current must be >= 2.5x the 3-second baseline
NOVELTY_SPIKE_MIN_DURATION_S = 15.0  # guard: only fire in long events

# Action labels that constitute "fast action" for hard-trigger purposes.
# Must match the buckets in adapters/action_stub.py.
_ACTION_TRIGGER_LABELS: frozenset = frozenset({"fast_action"})


class CandidateDetector:
    """
    Stateful boundary candidate detector.

    Usage:
        detector = CandidateDetector(config, context_buffer)
        # Each second:
        result = detector.update(curr_feature, prev_feature)
        if result is not None:
            # A boundary decision was made (is_boundary may be True or False)
    """

    def __init__(self, config: AESEConfig, buffer: ContextBuffer) -> None:
        self.config = config
        self.buffer = buffer
        # Hold state: track pending low-confidence candidates
        self._hold_start_ms: Optional[float] = None
        self._hold_scores: List[float] = []
        self._hold_signals: List[BoundarySignal] = []
        self._hold_available: List[dict] = []  # availability dict per hold slot
        # Novelty spike gate: count seconds since the last boundary commitment.
        # Reset to 0 on every boundary (is_boundary=True return).
        # Provides the event-duration proxy needed by check_novelty_spike()
        # without requiring a reference to EventConstructor.
        self._seconds_since_boundary: int = 0

    def update(
        self,
        curr: TemporalFeature,
        prev: Optional[TemporalFeature],
    ) -> BoundaryDecision:
        """
        Process the current second's TemporalFeature and return a BoundaryDecision.

        Args:
            curr: Current second's TemporalFeature.
            prev: Previous second's TemporalFeature, or None for the first second.

        Returns:
            BoundaryDecision with is_boundary, confidence, dominant_signal, fused_score.
        """
        if prev is None:
            # First second — no decision possible yet
            return BoundaryDecision(
                is_boundary=False,
                confidence=0.0,
                dominant_signal="none",
                fused_score=0.0,
            )

        # --- Compute all signals ---
        signals = self._compute_signals(curr, prev)

        # Build availability map from image_available flag.
        # scene, character, and embedding all require real pixel data.
        # prediction_error is embedding-based — also unavailable without real images.
        # camera, dialogue, and music are derived from non-image signals (always available).
        image_ok = getattr(curr, "image_available", True)  # default True for backwards compat
        available = {
            "scene": image_ok,
            "character": image_ok,
            "embedding": image_ok,
            "prediction_error": image_ok,
            "dialogue": True,
            "camera": True,
            "music": True,
            "emotion": True,
        }

        fused = fuse(signals, self.config.weights, available)
        confidence = compute_confidence(
            fused, self.config.boundary_threshold, self.config.confidence_margin
        )
        dominant = dominant_signal_name(signals, self.config.weights, available)

        threshold = self.config.boundary_threshold
        margin = self.config.confidence_margin
        curr_ts = curr.timestamp_ms

        # ---------------------------------------------------------------
        # HARD TRIGGERS — checked before the weighted-fusion path.
        # These cover unambiguous, near-deterministic cases where a quiet
        # surrounding clip should not dilute a confirmed boundary.
        # The weighted-fusion path is NOT removed — it still runs for the
        # common ambiguous case (gradual mood/topic shift, etc.).
        # ---------------------------------------------------------------

        # HARD TRIGGER 1 — real camera cut is near-deterministic boundary evidence.
        # Module 1 flags scene_change; camera_cues adapter maps it to "cut".
        # A confirmed hard cut must not need to out-vote an otherwise quiet clip.
        if curr.camera_cue == "cut":
            self._clear_hold()
            logger.info(
                "AESE HARD TRIGGER (scene_change) at ts=%.0f ms: "
                "camera_cue='cut' — boundary committed at confidence=0.95",
                curr_ts,
            )
            return BoundaryDecision(
                is_boundary=True,
                confidence=0.95,
                dominant_signal="scene_change",
                fused_score=fused,
            )

        # HARD TRIGGER 2 — sustained transition into fast action.
        # Requires 2 consecutive fast_action seconds following a non-action second
        # to avoid triggering on a single noisy motion spike (e.g. camera shake).
        if self._check_action_transition():
            self._clear_hold()
            logger.info(
                "AESE HARD TRIGGER (motion_spike) at ts=%.0f ms: "
                "2 consecutive fast_action seconds after non-action — boundary committed at confidence=0.85",
                curr_ts,
            )
            return BoundaryDecision(
                is_boundary=True,
                confidence=0.85,
                dominant_signal="motion_spike",
                fused_score=fused,
            )

        # HARD TRIGGER 3 — novelty-only spike inside a long event (Fix 2 — §21.2).
        # Fires when novelty jumps sharply relative to the recent 3-second baseline,
        # decoupled from motion. Catches slow, cut-free reveals that a motion-based
        # trigger cannot see (a deliberate entrance with no fast movement).
        # Gate: event must be > 15 s old (prevents it becoming a general-purpose
        # high-frequency trigger on short events).
        self._seconds_since_boundary += 1
        if (
            self._seconds_since_boundary > NOVELTY_SPIKE_MIN_DURATION_S
            and self._check_novelty_spike(curr)
        ):
            self._clear_hold()  # resets _seconds_since_boundary to 0
            logger.info(
                "AESE HARD TRIGGER (novelty_spike) at ts=%.0f ms: "
                "novelty=%.3f spiked above %.1f× baseline in a %.0fs event — "
                "boundary committed at confidence=0.80",
                curr_ts,
                curr.novelty_score,
                NOVELTY_SPIKE_RATIO,
                self._seconds_since_boundary,
            )
            return BoundaryDecision(
                is_boundary=True,
                confidence=0.80,
                dominant_signal="novelty_spike",
                fused_score=fused,
            )

        if fused >= threshold + margin:
            self._clear_hold()
            logger.debug(
                "AESE boundary confirmed at ts=%.0f ms: score=%.3f confidence=%.3f signal=%s",
                curr_ts, fused, confidence, dominant,
            )
            return BoundaryDecision(
                is_boundary=True,
                confidence=confidence,
                dominant_signal=dominant,
                fused_score=fused,
            )

        # --- In the low-confidence zone: threshold - margin ≤ score < threshold + margin ---
        if fused >= threshold - margin:
            if self._hold_start_ms is None:
                # Start a new hold
                self._hold_start_ms = curr_ts
                self._hold_scores = [fused]
                self._hold_signals = [signals]
                self._hold_available = [available]
                logger.debug(
                    "AESE boundary hold started at ts=%.0f ms: score=%.3f",
                    curr_ts, fused,
                )
                return BoundaryDecision(
                    is_boundary=False,  # not yet committed
                    confidence=confidence,
                    dominant_signal=dominant,
                    fused_score=fused,
                )
            else:
                # Continue the hold — accumulate scores
                self._hold_scores.append(fused)
                self._hold_signals.append(signals)
                self._hold_available.append(available)
                hold_duration = curr_ts - self._hold_start_ms

                # Commit after 2s of holding OR if we have ≥ 2 scores
                if hold_duration >= _MAX_HOLD_MS or len(self._hold_scores) >= 2:
                    avg_score = float(np.mean(self._hold_scores))
                    avg_confidence = compute_confidence(avg_score, threshold, margin)
                    # Re-evaluate the combined signal, passing stored availability per slot
                    all_signals_fused = [
                        fuse(s, self.config.weights, av)
                        for s, av in zip(self._hold_signals, self._hold_available)
                    ]
                    final_score = max(all_signals_fused)  # take the peak
                    final_conf = compute_confidence(final_score, threshold, margin)
                    is_boundary = final_score >= threshold - margin  # lenient after hold

                    # Find the dominant signal from the peak window
                    peak_idx = int(np.argmax(all_signals_fused))
                    final_dominant = dominant_signal_name(
                        self._hold_signals[peak_idx], self.config.weights,
                        self._hold_available[peak_idx],
                    )

                    self._clear_hold()
                    logger.debug(
                        "AESE hold committed at ts=%.0f ms: score=%.3f is_boundary=%s "
                        "(hold=%.0f ms)",
                        curr_ts, final_score, is_boundary, hold_duration,
                    )
                    return BoundaryDecision(
                        is_boundary=is_boundary,
                        confidence=final_conf,
                        dominant_signal=final_dominant,
                        fused_score=final_score,
                    )
                else:
                    # Still within hold window — keep waiting
                    return BoundaryDecision(
                        is_boundary=False,
                        confidence=confidence,
                        dominant_signal=dominant,
                        fused_score=fused,
                    )

        # --- Below threshold (and below low-confidence zone) ---
        # If we were in a hold and score dropped away, cancel the hold
        if self._hold_start_ms is not None:
            hold_duration = curr_ts - self._hold_start_ms
            if hold_duration >= _MAX_HOLD_MS:
                # 2s elapsed without confirmation — commit as non-boundary
                self._clear_hold()
            # else: keep holding (score may recover next second)

        return BoundaryDecision(
            is_boundary=False,
            confidence=confidence,
            dominant_signal=dominant,
            fused_score=fused,
        )

    def _compute_signals(
        self, curr: TemporalFeature, prev: TemporalFeature
    ) -> BoundarySignal:
        """Compute all boundary signals for the current second."""
        # Embedding distance: current vs previous
        emb_dist = embedding_distance(
            curr.multimodal_embedding, prev.multimodal_embedding, metric="cosine"
        )

        # Prediction error: current embedding vs. predicted from buffer history
        recent_embs = self.buffer.recent_embeddings(n=3)
        pred_error = compute_prediction_error(recent_embs, curr.multimodal_embedding)

        return BoundarySignal(
            scene=scene_signal(curr, prev),
            character=character_signal(curr, prev),
            dialogue=dialogue_signal(curr, prev),
            camera=camera_signal(curr),
            emotion=emotion_signal(curr, prev),  # always 0.0 — see DECISIONS.md §8
            music=music_signal(curr, prev),
            embedding_distance=emb_dist,
            prediction_error=pred_error,
        )

    def _check_action_transition(self) -> bool:
        """
        Return True if the context buffer shows a sustained transition into fast action:
          - The feature immediately before the 2-second window was NOT fast_action
          - The most recent 2 features (including current) are BOTH fast_action

        Uses recent(3) so we have [before, prev, curr] — the 'before' second
        tells us the pre-transition state without looking at event_type (which
        doesn't exist until after an event closes).

        Returns False if fewer than 3 features exist in the buffer.
        """
        recent = self.buffer.recent(3)  # [before, prev, curr]
        if len(recent) < 3:
            return False
        was_non_action = recent[0].action_label not in _ACTION_TRIGGER_LABELS
        now_action = all(f.action_label in _ACTION_TRIGGER_LABELS for f in recent[-2:])
        return was_non_action and now_action

    def _check_novelty_spike(self, curr: TemporalFeature) -> bool:
        """
        Return True if novelty jumps sharply above the recent baseline.

        Logic:
          - Compute the mean novelty of the 3 features BEFORE the current one
            (baseline = what 'normal' looks like for this event)
          - Fire if curr.novelty_score >= NOVELTY_SPIKE_THRESHOLD (absolute floor)
            AND curr.novelty_score >= baseline * NOVELTY_SPIKE_RATIO (relative jump)

        Requires at least 4 features in the buffer (3 for baseline + current).
        Returns False if buffer is too small.

        Why the RATIO guard matters: if baseline is already high (a busy, dynamic
        event), a further jump should NOT fire — the ratio ensures only
        stepwise changes trigger, not sustained high-novelty periods.
        """
        recent = self.buffer.recent(4)  # [t-3, t-2, t-1, curr]
        if len(recent) < 4:
            return False
        baseline_features = recent[:-1]  # the 3 seconds before current
        baseline = sum(f.novelty_score for f in baseline_features) / len(baseline_features)
        return (
            curr.novelty_score >= NOVELTY_SPIKE_THRESHOLD
            and curr.novelty_score >= baseline * NOVELTY_SPIKE_RATIO
        )

    def _clear_hold(self) -> None:
        """Reset hold state and the event-duration counter."""
        self._hold_start_ms = None
        self._hold_scores = []
        self._hold_signals = []
        self._hold_available = []
        self._seconds_since_boundary = 0
