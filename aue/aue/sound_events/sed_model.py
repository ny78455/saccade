"""
aue/aue/sound_events/sed_model.py
Sound Event Detection (SED) using the shared PANNs model.

SHARED MODEL: The same PANNs CNN14 instance loaded in music/music_detection.py
is used here — no second large model load. This is the primary architectural
reason for choosing PANNs: one model, two uses (music detection + SED).

Ambiguity handling (per spec §3.21):
  When the top prediction confidence is below the threshold, the event is
  stored as "ambiguous" with top_predictions populated. A single forced label
  is NEVER returned for ambiguous inputs.

  Example: a "gunshot vs. firecracker" scenario where both labels score ~0.4.
  Output: SoundEvent(event_type="ambiguous", top_predictions=[("Gunshot", 0.42),
          ("Firecracker", 0.38), ("Explosion", 0.21)])

PANNs AudioSet labels:
  CNN14 covers 527 AudioSet classes. We filter out "Music" classes (already
  handled by music_detection.py) and focus on non-music events for SED.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..types import SoundEvent

logger = logging.getLogger(__name__)

# AudioSet class index ranges that are primarily music (handled by music_detection)
# We still report them but flag them as music-category events.
_MUSIC_CLASS_RANGE_START = 137  # "Music"
_MUSIC_CLASS_RANGE_END = 330    # end of music-related classes

_TOP_K = 3   # Number of top predictions to keep for ambiguous cases

# AudioSet label list (527 classes) — we load this from panns_inference if available.
# Fallback: numeric indices used as string labels if label list unavailable.
_AUDIOSET_LABELS: Optional[List[str]] = None


def _get_audioset_labels() -> Optional[List[str]]:
    """Load AudioSet label list from panns_inference package."""
    global _AUDIOSET_LABELS
    if _AUDIOSET_LABELS is not None:
        return _AUDIOSET_LABELS
    try:
        import panns_inference
        import os
        label_path = os.path.join(
            os.path.dirname(panns_inference.__file__), "assets", "labels.csv"
        )
        if os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            # CSV format: index,mid,display_name
            labels = []
            for line in lines[1:]:  # skip header
                parts = line.strip().split(",", 2)
                if len(parts) >= 3:
                    labels.append(parts[2].strip('"').strip())
                elif len(parts) == 2:
                    labels.append(parts[1].strip('"').strip())
            _AUDIOSET_LABELS = labels
        else:
            _AUDIOSET_LABELS = None
    except Exception:
        _AUDIOSET_LABELS = None
    return _AUDIOSET_LABELS


def _idx_to_label(idx: int) -> str:
    """Convert AudioSet class index to human-readable label."""
    labels = _get_audioset_labels()
    if labels and idx < len(labels):
        return labels[idx]
    return f"AudioSet_{idx}"


def detect_sound_events(
    waveform_window: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
    confidence_threshold: float = 0.5,
    checkpoint_dir: Optional[str] = None,
) -> List[SoundEvent]:
    """
    Detect sound events in a short waveform window using PANNs.

    Args:
        waveform_window:     Audio window slice from preloaded waveform.
        sr:                  Sample rate.
        start_ms:            Window start time (for SoundEvent timestamps).
        end_ms:              Window end time.
        confidence_threshold: Threshold below which top_predictions is populated
                             and event_type is set to "ambiguous".
        checkpoint_dir:      Optional PANNs checkpoint directory override.

    Returns:
        List of SoundEvent objects (typically 1 per window, potentially 0 on error).

    Honesty contract (spec §3.21):
        When top prediction confidence < threshold, event_type="ambiguous" and
        top_predictions contains the top-k candidates. A single forced label
        is NEVER returned when the signal is genuinely ambiguous.
    """
    if len(waveform_window) == 0:
        return []

    # Use shared PANNs model from music_detection
    from ..music.music_detection import get_panns_model, _panns_available

    panns_model = get_panns_model(checkpoint_dir)

    if panns_model is None or not _panns_available:
        return _heuristic_sed(waveform_window, sr, start_ms, end_ms, confidence_threshold)

    try:
        import librosa
        if sr != 32000:
            audio_32k = librosa.resample(waveform_window.astype(np.float32), orig_sr=sr, target_sr=32000)
        else:
            audio_32k = waveform_window.astype(np.float32)

        audio_input = audio_32k[np.newaxis, :]
        clipwise_output, _ = panns_model.inference(audio_input)
        probs = clipwise_output[0]  # shape (527,)

        # Get top-k predictions
        top_k_indices = np.argsort(probs)[::-1][:_TOP_K]
        top_k = [(str(_idx_to_label(int(i))), float(probs[i])) for i in top_k_indices]

        top_label, top_conf = top_k[0]

        if top_conf < confidence_threshold:
            # Ambiguous — store all top-k, do NOT force single label
            logger.debug(
                "AUE SED: ambiguous event at [%.0f–%.0f ms] "
                "(top=%s, conf=%.3f < %.2f) — storing top_predictions.",
                start_ms, end_ms, top_label, top_conf, confidence_threshold,
            )
            return [SoundEvent(
                event_type="ambiguous",
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=top_conf,
                top_predictions=top_k,
            )]

        # High-confidence event
        return [SoundEvent(
            event_type=top_label,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=top_conf,
            top_predictions=top_k,  # always store for transparency
        )]

    except Exception as exc:
        logger.warning(
            "AUE SED: PANNs inference failed at [%.0f–%.0f ms]: %s",
            start_ms, end_ms, exc,
        )
        return []


def _heuristic_sed(
    waveform_window: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
    confidence_threshold: float,
) -> List[SoundEvent]:
    """
    Heuristic sound-event detection fallback when PANNs is unavailable.
    Uses energy + ZCR + spectral features to classify into broad categories.
    """
    try:
        import librosa
        y = waveform_window.astype(np.float32)
        if len(y) < 64:
            return []

        rms = float(np.sqrt(np.mean(y ** 2)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

        candidates = []
        if rms > 0.5 and zcr > 0.15:
            candidates.append(("Explosion", 0.45))
        if rms > 0.3 and centroid > 3000:
            candidates.append(("Speech", 0.40))
        if rms > 0.1 and centroid > 1000:
            candidates.append(("Music", 0.35))
        if rms < 0.02:
            candidates.append(("Silence", 0.80))
        if not candidates:
            candidates.append(("Noise", 0.30))

        top_label, top_conf = candidates[0]
        top_k = candidates[:_TOP_K]

        if top_conf < confidence_threshold:
            return [SoundEvent(
                event_type="ambiguous",
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=top_conf,
                top_predictions=top_k,
            )]
        return [SoundEvent(
            event_type=top_label,
            start_ms=start_ms,
            end_ms=end_ms,
            confidence=top_conf,
            top_predictions=top_k,
        )]
    except Exception as exc:
        logger.debug("AUE SED: heuristic fallback failed: %s", exc)
        return []
