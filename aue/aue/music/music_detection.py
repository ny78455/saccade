"""
aue/aue/music/music_detection.py
Music detection using the shared PANNs (Pretrained Audio Neural Networks) model.

The PANNs CNN14 (or CNN6 for speed) covers AudioSet's 527 classes including
"Music" and broad instrument tags. The SAME model instance is shared with
sound_events/sed_model.py — no second large model load for overlapping tasks.

Why PANNs over a dedicated music detector:
  A single efficient AudioSet CNN covers both music detection and sound-event
  detection (Section 5.6), avoiding two separate heavy model loads for overlapping
  functionality. See DECISIONS.md §AUE-5.

Music detection uses a 2-second sliding window. The "Music" tag probability
(AudioSet class 137) is used as music_probability. Tags for instruments and
broad musical content are also checked for corroboration.

Fallback: if panns-inference is unavailable, a heuristic based on spectral
centroid + ZCR (replicating ASVL's approach) is used as a lightweight substitute.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# AudioSet class indices for music-related tags
# (standard AudioSet 527-class label order)
_MUSIC_CLASS_IDX = 137     # "Music"
_MUSIC_RELATED_IDXS = [    # Instruments, music genres, etc.
    137,  # Music
    138,  # Musical instrument
    139,  # Plucked string instrument
    140,  # Guitar
    141,  # Electric guitar
    142,  # Bass guitar
    143,  # Acoustic guitar
    150,  # Piano
    151,  # Keyboard (musical)
    152,  # Organ
    153,  # Synthesizer
    154,  # Sampler
    155,  # Harpsichord
    156,  # Percussion
    157,  # Drum kit
    158,  # Drum machine
    159,  # Drum
    160,  # Snare drum
    161,  # Rimshot
    162,  # Drum roll
    163,  # Bass drum
    164,  # Timpani
    165,  # Tabla
    166,  # Cymbal
    167,  # Hi-hat
    168,  # Wood block
    169,  # Tambourine
    170,  # Rattle (instrument)
    171,  # Maraca
    172,  # Gong
    173,  # Tubular bells
    174,  # Mallet percussion
    175,  # Marimba, xylophone
    176,  # Glockenspiel
    177,  # Vibraphone
    178,  # Steelpan
    284,  # Music of Latin America
    285,  # Salsa music
    286,  # Flamenco
    287,  # Blues
    288,  # Music for children
    289,  # New-age music
    290,  # Vocal music
    291,  # A capella
    292,  # Music of Africa
    293,  # Afrobeat
    294,  # Christian music
    295,  # Gospel music
    296,  # Electronic music
    297,  # Music of Bollywood
    298,  # Ska
    299,  # Traditional music
    300,  # Independent music
    301,  # Pop music
    302,  # Hip hop music
    303,  # Rhythm and blues
    304,  # Soul music
    305,  # Reggae
    306,  # Country
    307,  # Swing music
    308,  # Bluegrass
    309,  # Funk
    310,  # Folk music
    311,  # Middle Eastern music
    312,  # Jazz
    313,  # Disco
    314,  # Classical music
    315,  # Opera
    316,  # Electronic dance music
    317,  # House music
    318,  # Techno
    319,  # Dubstep
    320,  # Drum and bass
    321,  # Electronica
    322,  # Electronic music
    323,  # Ambient music
    324,  # Trance music
    325,  # Music of Asia
    326,  # Music of Korea
    327,  # Carnatic music
    328,  # Music of Japan
    329,  # Mandopop
    330,  # Indonesion pop
]

# Shared PANNs model instance (initialized on first call)
_panns_model = None
_panns_available: Optional[bool] = None
_panns_labels: Optional[List[str]] = None


def _load_panns(checkpoint_dir: Optional[str] = None):
    """Load PANNs CNN14 model (or CNN6 for speed). Called once, result cached."""
    global _panns_model, _panns_available, _panns_labels

    if _panns_available is not None:
        return _panns_available

    try:
        from panns_inference import AudioTagging
        kwargs = {}
        if checkpoint_dir:
            kwargs["checkpoint_path"] = checkpoint_dir

        logger.info("AUE music_detection: loading PANNs AudioTagging model...")
        _panns_model = AudioTagging(model_type="CNN14", **kwargs)

        # Load label list
        try:
            from panns_inference import labels as panns_labels_module
            _panns_labels = panns_labels_module
        except Exception:
            _panns_labels = None

        _panns_available = True
        logger.info("AUE music_detection: PANNs CNN14 ready (AudioSet 527 classes).")
    except ImportError as exc:
        _panns_available = False
        logger.warning(
            "AUE music_detection: panns-inference unavailable (%s). "
            "Falling back to heuristic music detection. "
            "Install: pip install panns-inference>=0.1.1",
            exc,
        )
    except Exception as exc:
        _panns_available = False
        logger.warning("AUE music_detection: PANNs load failed (%s); using heuristic.", exc)

    return _panns_available


def get_panns_model(checkpoint_dir: Optional[str] = None):
    """Return the shared PANNs model instance (loads once)."""
    _load_panns(checkpoint_dir)
    return _panns_model


def _music_prob_from_panns(waveform_window: np.ndarray, sr: int) -> tuple:
    """Run PANNs and return (music_probability, top_tags_dict)."""
    try:
        # PANNs expects shape (batch, samples) at 32kHz
        import librosa
        if sr != 32000:
            audio_32k = librosa.resample(waveform_window.astype(np.float32), orig_sr=sr, target_sr=32000)
        else:
            audio_32k = waveform_window.astype(np.float32)

        audio_input = audio_32k[np.newaxis, :]  # (1, N)
        clipwise_output, _ = _panns_model.inference(audio_input)
        probs = clipwise_output[0]  # shape (527,)

        music_prob = float(probs[_MUSIC_CLASS_IDX])

        # Collect top music-related tags
        music_tag_probs = {
            str(i): float(probs[i])
            for i in _MUSIC_RELATED_IDXS
            if i < len(probs)
        }
        max_related = max(music_tag_probs.values()) if music_tag_probs else 0.0
        combined_prob = max(music_prob, max_related * 0.9)

        return combined_prob, probs
    except Exception as exc:
        logger.debug("AUE music_detection: PANNs inference failed: %s", exc)
        return 0.0, None


def _music_prob_heuristic(waveform_window: np.ndarray, sr: int) -> float:
    """Heuristic music detection using spectral centroid + ZCR (mirrors ASVL's approach)."""
    try:
        import librosa
        if len(waveform_window) < 64:
            return 0.0
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=waveform_window.astype(np.float32), sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(waveform_window.astype(np.float32))))
        # Music: high centroid, moderate ZCR (not pure noise, not pure speech)
        if centroid > 1500 and 0.02 < zcr < 0.25:
            return 0.70
        elif centroid > 1000:
            return 0.45
        return 0.15
    except Exception:
        return 0.0


def detect_music_regions(
    waveform: np.ndarray,
    sr: int,
    window_seconds: float = 2.0,
    threshold: float = 0.5,
    hop_seconds: float = 1.0,
    checkpoint_dir: Optional[str] = None,
) -> List[dict]:
    """
    Detect music regions across a waveform using a sliding window.

    Args:
        waveform:       Full preloaded waveform (from audio_source.preload_audio).
        sr:             Sample rate.
        window_seconds: Analysis window size in seconds.
        threshold:      music_probability threshold for marking a window as music.
        hop_seconds:    Window hop in seconds.
        checkpoint_dir: Optional PANNs checkpoint directory.

    Returns:
        List of dicts: {start_ms, end_ms, music_probability, raw_probs (optional)}
        These are merged into MusicSegment objects by the pipeline.
    """
    if len(waveform) == 0:
        return []

    _load_panns(checkpoint_dir)

    window_samples = int(window_seconds * sr)
    hop_samples = int(hop_seconds * sr)
    total_samples = len(waveform)
    regions = []

    for start in range(0, total_samples - window_samples + 1, hop_samples):
        window = waveform[start: start + window_samples]
        start_ms = start / sr * 1000.0
        end_ms = (start + window_samples) / sr * 1000.0

        if _panns_available:
            music_prob, raw_probs = _music_prob_from_panns(window, sr)
        else:
            music_prob = _music_prob_heuristic(window, sr)
            raw_probs = None

        regions.append({
            "start_ms": start_ms,
            "end_ms": end_ms,
            "music_probability": music_prob,
            "raw_probs": raw_probs,
        })

    return regions
