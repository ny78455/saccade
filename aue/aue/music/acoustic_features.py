"""
aue/aue/music/acoustic_features.py
Acoustic feature extraction for music segments using librosa.

All computations are:
  - Cheap: no neural network inference, pure signal processing.
  - Deterministic: same input always yields same output.
  - In-memory: operates on waveform slices from preload_audio().

Features extracted:
  - tempo_bpm:    estimated via librosa.beat.tempo()
  - rms_energy:   root mean square energy (0.0–1.0 normalized)
  - chroma:       chroma STFT (12-dim vector per window)
  - chroma_mode:  dominant chroma bin (0–11, proxy for musical key)
  - spectral_centroid: mean spectral centroid in Hz
  - spectral_rolloff:  mean spectral rolloff in Hz
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_music_acoustic_features(
    waveform_slice: np.ndarray,
    sr: int,
) -> dict:
    """
    Extract acoustic features from a music window.

    Args:
        waveform_slice: Audio slice from preloaded waveform.
        sr:             Sample rate.

    Returns:
        dict with keys: tempo_bpm, rms_energy, chroma (12-dim list),
        chroma_mode (int 0–11), spectral_centroid, spectral_rolloff.
        All values are 0.0 / empty if waveform_slice is too short.
    """
    result = {
        "tempo_bpm": None,
        "rms_energy": 0.0,
        "chroma": [0.0] * 12,
        "chroma_mode": 0,
        "spectral_centroid": 0.0,
        "spectral_rolloff": 0.0,
    }

    if len(waveform_slice) < 512:
        return result

    try:
        import librosa
        y = waveform_slice.astype(np.float32)

        # Tempo
        try:
            tempo_arr = librosa.beat.tempo(y=y, sr=sr)
            result["tempo_bpm"] = float(tempo_arr[0]) if len(tempo_arr) > 0 else None
        except Exception:
            result["tempo_bpm"] = None

        # RMS energy (normalized to peak)
        rms = float(np.sqrt(np.mean(y ** 2)))
        rms_global_peak = float(np.abs(y).max()) + 1e-9
        result["rms_energy"] = float(np.clip(rms / (rms_global_peak * 0.7), 0.0, 1.0))

        # Chroma STFT
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)   # shape (12,)
        result["chroma"] = chroma_mean.tolist()
        result["chroma_mode"] = int(np.argmax(chroma_mean))

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        result["spectral_centroid"] = float(np.mean(centroid))

        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        result["spectral_rolloff"] = float(np.mean(rolloff))

    except Exception as exc:
        logger.debug("AUE acoustic_features: extraction failed: %s", exc)

    return result
