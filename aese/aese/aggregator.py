"""
aese/aggregator.py
Feature Aggregator — converts a variable-rate stream of FramePackets into
one TemporalFeature per wall-clock second.

Aggregation rules per §5.1:
  - Embeddings/motion/novelty/audio: mean pooling over all frames in [t, t+1000ms)
  - Categorical labels (scene_label, action_label, music_mood): majority vote
  - Booleans (dialogue_present): OR-reduce (any frame in the window triggers it)
  - camera_cue: first non-None value in the window (highest priority cue wins)
  - dialogue_text: concatenation of unique non-empty texts in the window

Streaming continuity rule (§5.1):
  If zero FramePackets fall in a given second, carry forward the previous
  TemporalFeature rather than emitting a gap. This prevents silent timeline
  gaps from corrupting downstream boundary signals.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Iterator, List, Optional

import numpy as np

from .adapters.action_stub import label_action
from .adapters.camera_cues import detect_camera_cue
from .adapters.character_cluster import extract_face_embeddings
from .adapters.character_stub import count_characters, detect_faces_with_boxes
from .adapters.embedding import compute_embedding
from .adapters.music_mood import estimate_spectral_flux, label_mood
from .adapters.scene_label import label_scene
from .types import AESEConfig, FramePacket, TemporalFeature

logger = logging.getLogger(__name__)

# Default placeholder image for manifest-replay mode (black frame)
_BLACK_FRAME = np.zeros((64, 64, 3), dtype=np.uint8)


def _make_black_frame() -> np.ndarray:
    return np.zeros((64, 64, 3), dtype=np.uint8)


class FeatureAggregator:
    """
    Collects FramePackets at variable rates and emits one TemporalFeature per second.

    Usage:
        agg = FeatureAggregator(config)
        for fp in frame_packets:
            result = agg.push(fp)
            if result is not None:
                # one TemporalFeature emitted for the just-completed second
                process(result)
        # Flush any trailing partial second at end-of-stream
        for tf in agg.flush():
            process(tf)
    """

    def __init__(self, config: AESEConfig) -> None:
        self.config = config
        self._window_start_ms: Optional[float] = None
        self._buffer: List[FramePacket] = []
        self._prev_feature: Optional[TemporalFeature] = None
        self._prev_audio_energy: Optional[float] = None
        self._prev_was_black: bool = False
        self._last_emitted_second: int = -1  # tracks which 1-second bucket was last emitted

    def push(self, fp: FramePacket) -> Optional[TemporalFeature]:
        """
        Push a FramePacket. Returns a TemporalFeature if a 1-second window just completed,
        otherwise returns None.

        NOTE: If the new packet's timestamp jumps by more than 1 second past the current
        window, this method fills in carry-forward TemporalFeatures for the gap and
        returns the most recent one. Callers should use push_all() to handle gaps correctly.
        """
        if self._window_start_ms is None:
            self._window_start_ms = float(int(fp.timestamp_ms // 1000) * 1000)

        # Check if this packet falls in the current window or a new one
        curr_second = int(fp.timestamp_ms // 1000)
        window_second = int(self._window_start_ms // 1000)

        if curr_second == window_second:
            # Still in current window — accumulate
            self._buffer.append(fp)
            return None
        else:
            # New second — close current window, emit TemporalFeature
            result = self._close_window()
            # Advance to the correct window for this packet
            self._window_start_ms = float(curr_second * 1000)
            self._buffer = [fp]
            return result

    def push_all(self, fp: FramePacket) -> List[TemporalFeature]:
        """
        Push a FramePacket and return all TemporalFeatures emitted (handles multi-second gaps).
        This is the preferred interface for the pipeline.
        """
        if self._window_start_ms is None:
            self._window_start_ms = float(int(fp.timestamp_ms // 1000) * 1000)

        curr_second = int(fp.timestamp_ms // 1000)
        window_second = int(self._window_start_ms // 1000)

        if curr_second == window_second:
            self._buffer.append(fp)
            return []

        results: List[TemporalFeature] = []
        # Close current window
        tf = self._close_window()
        results.append(tf)

        # Fill any gap seconds with carry-forward (streaming continuity rule)
        next_second = window_second + 1
        while next_second < curr_second:
            carry = self._carry_forward(next_second * 1000.0)
            results.append(carry)
            next_second += 1

        # Start the new window
        self._window_start_ms = float(curr_second * 1000)
        self._buffer = [fp]
        return results

    def flush(self) -> List[TemporalFeature]:
        """
        Flush any remaining buffered frames as a final partial-second TemporalFeature.
        Call at end-of-stream.
        """
        if not self._buffer:
            return []
        tf = self._close_window()
        self._buffer = []
        return [tf]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _close_window(self) -> TemporalFeature:
        """Aggregate buffered packets into a single TemporalFeature."""
        ts = self._window_start_ms or 0.0

        if not self._buffer:
            # Continuity rule: no packets in this second → carry forward
            return self._carry_forward(ts)

        packets = self._buffer

        # --- Determine whether any real pixel data exists for this second ---
        has_real_image = any(p.image is not None for p in packets)

        # --- Embedding: use real image only; never substitute a black frame ---
        image_for_embedding = next(
            (p.image for p in reversed(packets) if p.image is not None), None
        )
        subtitle_text = next(
            (p.subtitle_text for p in reversed(packets) if p.subtitle_text), None
        )
        if image_for_embedding is not None:
            embedding = compute_embedding(
                image_for_embedding,
                subtitle_text,
                model_name=self.config.clip_model,
                pretrained=self.config.clip_pretrained,
                fusion=self.config.embedding_fusion,
            )
        else:
            # No real image — propagate None; fusion will renormalize around this gap
            from .adapters.embedding import EMBEDDING_DIM
            embedding = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        # --- Numeric: mean pooling ---
        motion_mean = float(np.mean([p.motion_score for p in packets]))
        novelty_mean = float(np.mean([p.novelty_score for p in packets]))
        audio_mean = float(np.mean([p.audio_energy for p in packets]))

        # --- Spectral flux estimate from audio delta ---
        flux = estimate_spectral_flux(self._prev_audio_energy, audio_mean)
        self._prev_audio_energy = audio_mean

        # --- Categorical: majority vote ---
        # Only call image-dependent adapters on packets with real pixel data.
        # Do NOT substitute a black frame — that conflates "missing" with "confirmed zero."
        real_image_packets = [p for p in packets if p.image is not None]

        # Scene content rarely changes meaningfully within a single second —
        # label ONE representative frame per second, not every raw frame in the
        # window.  Using the middle frame avoids first/last-frame edge artefacts.
        #
        # DECISIONS.md §18 — call-frequency fix:
        #   Before this change, label_scene() was called once per raw FramePacket
        #   (N calls/second at 5–10 fps).  With the gemma4 backend, each call
        #   triggers a full generative forward pass, inflating call volume by up to
        #   10× per second.  Majority-voting N generative calls for the same second
        #   is pure waste; a single deterministic label is all the aggregator needs.
        if real_image_packets:
            _rep_packet = real_image_packets[len(real_image_packets) // 2]
            scene_label = (
                label_scene(_rep_packet.image)
                if _rep_packet.image is not None
                else "unknown"
            )
        else:
            scene_label = "unknown"

        action_labels = [label_action(p.motion_score) for p in packets]
        action_label = _majority_vote(action_labels)

        music_moods = [label_mood(p.audio_energy, flux) for p in packets]
        music_mood = _majority_vote(music_moods)

        # --- Boolean: OR-reduce ---
        dialogue_present = any(bool(p.subtitle_text) for p in packets)
        dialogue_texts = list(dict.fromkeys(
            p.subtitle_text for p in packets if p.subtitle_text
        ))
        dialogue_text_combined = " | ".join(dialogue_texts) if dialogue_texts else None

        # --- Camera cue: first non-None wins ---
        camera_cue: Optional[str] = None
        for p in packets:
            cue = detect_camera_cue(
                scene_change=p.scene_change,
                image=p.image,
                prev_was_black=self._prev_was_black,
            )
            if cue is not None:
                camera_cue = cue
                break

        # Update prev_was_black using last packet's image
        last_img = packets[-1].image
        self._prev_was_black = (last_img is not None and float(last_img.mean()) < 10.0)

        # --- Character count: max across window (real images only) ---
        # None means "not observed" (no real image). 0 means "observed zero faces."
        char_counts = [count_characters(p.image) for p in real_image_packets]
        character_count: Optional[int] = max(char_counts) if char_counts else None

        # --- Face embeddings: for Fix 4 anonymous clustering (real images only) ---
        # Use the representative image if available; extract face boxes then embed crops.
        # Returns [] gracefully when CLIP is unavailable or no faces found.
        face_embeddings = []
        if image_for_embedding is not None:
            boxes = detect_faces_with_boxes(image_for_embedding)
            face_embeddings = extract_face_embeddings(image_for_embedding, boxes)

        tf = TemporalFeature(
            timestamp_ms=ts,
            scene_label=scene_label,
            character_count=character_count,
            action_label=action_label,
            dialogue_present=dialogue_present,
            dialogue_text=dialogue_text_combined,
            camera_cue=camera_cue,
            music_mood=music_mood,
            multimodal_embedding=embedding,
            motion_score=motion_mean,
            novelty_score=novelty_mean,
            audio_energy=audio_mean,
            spectral_flux=flux,
            image_available=has_real_image,
            representative_image=image_for_embedding,  # raw RGB for VLM summary; None in manifest-replay mode
            face_embeddings=face_embeddings,
        )

        self._prev_feature = tf
        self._buffer = []
        return tf

    def _carry_forward(self, timestamp_ms: float) -> TemporalFeature:
        """
        Streaming continuity rule (§5.1): emit the previous second's feature
        for a gap second where no packets landed.
        """
        if self._prev_feature is None:
            # Cold-start: no previous feature — create a neutral default.
            # image_available=False because we have no real data yet.
            from .adapters.embedding import EMBEDDING_DIM
            return TemporalFeature(
                timestamp_ms=timestamp_ms,
                scene_label="unknown",
                character_count=None,    # None, not 0 — no real image data available
                action_label="static",
                dialogue_present=False,
                dialogue_text=None,
                camera_cue=None,
                music_mood="calm",
                multimodal_embedding=np.zeros(EMBEDDING_DIM, dtype=np.float32),
                motion_score=0.0,
                novelty_score=0.0,
                audio_energy=0.0,
                spectral_flux=0.0,
                image_available=False,
            )
        # Return a copy of the previous feature with updated timestamp
        prev = self._prev_feature
        return TemporalFeature(
            timestamp_ms=timestamp_ms,
            scene_label=prev.scene_label,
            character_count=prev.character_count,
            action_label=prev.action_label,
            dialogue_present=prev.dialogue_present,
            dialogue_text=prev.dialogue_text,
            camera_cue=None,  # camera cue is per-frame; don't carry over
            music_mood=prev.music_mood,
            multimodal_embedding=prev.multimodal_embedding.copy(),
            motion_score=prev.motion_score,
            novelty_score=prev.novelty_score,
            audio_energy=prev.audio_energy,
            spectral_flux=prev.spectral_flux,
            image_available=prev.image_available,  # carry forward availability flag
        )


def _majority_vote(labels: List[str]) -> str:
    """Return the most common label from a list. Returns the first if tie."""
    if not labels:
        return "unknown"
    counter = Counter(labels)
    return counter.most_common(1)[0][0]
