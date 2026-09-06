"""
aue/aue/diarization/speaker_embedding.py
Lightweight speaker embedding extraction using resemblyzer.

Why resemblyzer over pyannote:
  - resemblyzer is ~17MB vs pyannote's multi-model stack (>500MB).
  - Sufficient for near-real-time diarization on movie-length audio.
  - Full pyannote pipeline (offline resegmentation) is too heavy for the
    latency budget. See DECISIONS.md §AUE-3 for DER accuracy tradeoff.

Fallback:
  If resemblyzer is unavailable (e.g. webrtcvad build failure on Windows),
  falls back to an MFCC mean vector computed with librosa. MFCC-based speaker
  embeddings are less discriminative than GE2E embeddings but non-zero and
  functional. This fallback is logged honestly as a degraded mode.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Embedding source flag
_EMBEDDING_SOURCE: Optional[str] = None   # "resemblyzer" | "mfcc_fallback" | "none"

# Resemblyzer encoder singleton
_encoder = None

# resemblyzer expects audio at this sample rate
_RESEMBLYZER_SR = 16000


def _probe_embedding_source() -> str:
    """Probe resemblyzer availability and cache result."""
    global _EMBEDDING_SOURCE, _encoder
    if _EMBEDDING_SOURCE is not None:
        return _EMBEDDING_SOURCE

    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        _encoder = VoiceEncoder()
        _EMBEDDING_SOURCE = "resemblyzer"
        logger.info(
            "AUE speaker_embedding: resemblyzer GE2E encoder ready "
            "(d-vector embeddings, 256-dim)."
        )
    except ImportError as exc:
        _EMBEDDING_SOURCE = "mfcc_fallback"
        logger.warning(
            "AUE speaker_embedding: resemblyzer unavailable (%s). "
            "Falling back to MFCC mean embeddings — less speaker-discriminative. "
            "Install: pip install resemblyzer>=0.1.4 (requires webrtcvad C build).",
            exc,
        )
    except Exception as exc:
        _EMBEDDING_SOURCE = "mfcc_fallback"
        logger.warning(
            "AUE speaker_embedding: resemblyzer init failed (%s); using MFCC fallback.",
            exc,
        )
    return _EMBEDDING_SOURCE


def _embed_resemblyzer(waveform_chunk: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """Embed a speech segment using resemblyzer GE2E encoder."""
    try:
        from resemblyzer import preprocess_wav
        # resemblyzer expects float64 at 16kHz
        if sr != _RESEMBLYZER_SR:
            import librosa
            waveform_chunk = librosa.resample(
                waveform_chunk.astype(np.float64),
                orig_sr=sr,
                target_sr=_RESEMBLYZER_SR,
            )
        else:
            waveform_chunk = waveform_chunk.astype(np.float64)

        wav_preprocessed = preprocess_wav(waveform_chunk, source_sr=_RESEMBLYZER_SR)
        if len(wav_preprocessed) < 160:  # too short for resemblyzer
            return None

        embedding = _encoder.embed_utterance(wav_preprocessed)
        return embedding.astype(np.float32)
    except Exception as exc:
        logger.debug("AUE speaker_embedding: resemblyzer embed failed: %s", exc)
        return None


def _embed_mfcc_fallback(waveform_chunk: np.ndarray, sr: int) -> Optional[np.ndarray]:
    """
    MFCC mean vector as a fallback speaker embedding.
    Less discriminative than GE2E but never raises an error.
    Produces a 40-dimensional vector (13 MFCC + 13 delta + 13 delta-delta + RMS).
    """
    try:
        import librosa
        if len(waveform_chunk) < 256:
            return None
        mfcc = librosa.feature.mfcc(y=waveform_chunk.astype(np.float32), sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        rms = librosa.feature.rms(y=waveform_chunk.astype(np.float32))
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.mean(mfcc_delta2, axis=1),
            [float(np.mean(rms))],
        ])
        # L2-normalize
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        return features.astype(np.float32)
    except Exception as exc:
        logger.debug("AUE speaker_embedding: MFCC fallback failed: %s", exc)
        return None


def embed_speaker(
    waveform_chunk: np.ndarray,
    sr: int,
) -> Optional[np.ndarray]:
    """
    Extract a speaker embedding from an audio chunk.

    Uses resemblyzer GE2E if available, MFCC mean if not.
    Returns None if the chunk is too short or embedding fails.

    Args:
        waveform_chunk: Mono float32 waveform slice (from preloaded audio).
        sr:             Sample rate.

    Returns:
        L2-normalized float32 embedding vector, or None on failure.
    """
    source = _probe_embedding_source()

    if len(waveform_chunk) == 0:
        return None

    if source == "resemblyzer":
        emb = _embed_resemblyzer(waveform_chunk, sr)
        if emb is not None:
            return emb
        # Fallback within resemblyzer path (e.g. short chunk)
        logger.debug("AUE speaker_embedding: resemblyzer returned None; trying MFCC.")
        return _embed_mfcc_fallback(waveform_chunk, sr)

    return _embed_mfcc_fallback(waveform_chunk, sr)


def get_embedding_source() -> str:
    """Return the active embedding source (for logging / test assertions)."""
    return _probe_embedding_source()
