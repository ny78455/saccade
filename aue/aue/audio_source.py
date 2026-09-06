"""
aue/aue/audio_source.py
Single in-memory audio decode for AUE — the most important latency rule.

preload_audio() decodes the FULL audio track exactly once per run.
Every downstream stage (VAD, Whisper, diarization, music, sound events)
slices from the returned numpy array in memory — never re-decodes from disk.

This mirrors ASVL's preload pattern (asvl/audio.py::_extract_audio_numpy),
which was specifically built to avoid the audio-reload-per-segment bug.
AUE replicates the same discipline: preload_audio() must NEVER be called
inside a per-segment or per-window loop.

WHY AUE DECODES AUDIO ITSELF (deviation from spec §1.1):
  The source spec assumes ASVL streams AudioPacket(waveform, ...) to AUE.
  ASVL does NOT do this — it only exposes scalar audio_energy per 250ms window.
  AUE therefore decodes the audio track directly from the video file,
  the same way ASVL does internally. See DECISIONS.md §AUE-1.

Sample rate: 16000 Hz (matches Whisper's expected input natively, avoiding
an extra resample step inside faster-whisper on every segment).
"""
from __future__ import annotations

import logging
from typing import Optional

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Target sample rate — 16kHz is Whisper's native rate.
# Resampling to 16kHz once here avoids repeated resampling inside faster-whisper.
_TARGET_SR = 16000

# Call counter — used by tests to assert preload_audio() is called exactly once.
_preload_call_count = 0


def reset_call_counter() -> None:
    """Reset the call counter (for test isolation between test runs)."""
    global _preload_call_count
    _preload_call_count = 0


def get_call_count() -> int:
    """Return the number of times preload_audio() has been called."""
    return _preload_call_count


def preload_audio(video_path: str) -> tuple:
    """
    Decode the FULL audio track from video_path exactly once.

    Returns:
        (waveform: np.ndarray float32 mono, sample_rate: int)
        waveform shape: (N,) at 16kHz.

    Raises:
        RuntimeError if audio cannot be extracted (graceful: returns silence
        rather than crashing the whole pipeline — caller checks for empty array).

    CRITICAL CONTRACT:
        This function must be called exactly once per AUE run.
        Every downstream component receives a slice of the returned waveform.
        Never call this inside a per-segment, per-window, or per-buffer loop.
        Doing so would reintroduce the ASVL audio-reload bug this entire module
        was designed to avoid.
    """
    global _preload_call_count
    _preload_call_count += 1

    if _preload_call_count > 1:
        logger.error(
            "AUE audio_source: preload_audio() called %d times — this is the bug! "
            "It must be called exactly once per run. Check pipeline.py for a loop "
            "that calls preload_audio() per segment or per buffer.",
            _preload_call_count,
        )

    logger.info("AUE audio_source: decoding audio from %s at %d Hz...", video_path, _TARGET_SR)

    # --- Primary path: librosa (wraps ffmpeg/audioread) ---
    try:
        y, sr = librosa.load(video_path, sr=_TARGET_SR, mono=True)
        if len(y) == 0:
            raise ValueError("librosa returned empty waveform")
        logger.info(
            "AUE audio_source: loaded %.1f s of audio (%d samples at %d Hz)",
            len(y) / sr, len(y), sr,
        )
        return y.astype(np.float32), int(sr)
    except Exception as exc:
        logger.warning("AUE audio_source: librosa load failed (%s); trying PyAV fallback.", exc)

    # --- Fallback: PyAV (mirrors ASVL's _extract_audio_numpy) ---
    try:
        import av as pyav

        container = pyav.open(video_path)
        if not container.streams.audio:
            container.close()
            raise ValueError("No audio streams found via PyAV")

        audio_stream = container.streams.audio[0]
        native_sr = audio_stream.sample_rate
        chunks = []

        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            if arr.ndim > 1:
                arr = arr.mean(axis=0)
            chunks.append(arr.astype(np.float32))

        container.close()

        if not chunks:
            raise ValueError("PyAV decoded zero chunks")

        audio = np.concatenate(chunks)
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val

        # Resample to target SR if needed
        if native_sr != _TARGET_SR:
            audio = librosa.resample(audio, orig_sr=native_sr, target_sr=_TARGET_SR)

        logger.info(
            "AUE audio_source: PyAV fallback — loaded %.1f s at %d Hz",
            len(audio) / _TARGET_SR, _TARGET_SR,
        )
        return audio.astype(np.float32), _TARGET_SR

    except Exception as exc:
        logger.error(
            "AUE audio_source: PyAV fallback also failed (%s). "
            "Returning empty waveform — audio pipeline will degrade gracefully.",
            exc,
        )
        return np.zeros(0, dtype=np.float32), _TARGET_SR


def slice_waveform(
    waveform: np.ndarray,
    sr: int,
    start_ms: float,
    end_ms: float,
) -> np.ndarray:
    """
    Return a slice of the preloaded waveform for a given time range.

    Args:
        waveform: Full preloaded waveform (from preload_audio).
        sr:       Sample rate.
        start_ms: Start time in milliseconds.
        end_ms:   End time in milliseconds.

    Returns:
        np.ndarray float32 slice. May be shorter than requested if near EOF.
        Returns empty array if waveform is empty or range is invalid.
    """
    if len(waveform) == 0:
        return np.zeros(0, dtype=np.float32)

    start_sample = max(0, int(start_ms / 1000.0 * sr))
    end_sample = min(len(waveform), int(end_ms / 1000.0 * sr))

    if start_sample >= end_sample:
        return np.zeros(0, dtype=np.float32)

    return waveform[start_sample:end_sample]


def audio_duration_ms(waveform: np.ndarray, sr: int) -> float:
    """Return the duration of the waveform in milliseconds."""
    if sr <= 0 or len(waveform) == 0:
        return 0.0
    return len(waveform) / sr * 1000.0
