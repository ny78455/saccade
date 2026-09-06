"""
aue/aue/pipeline.py
AUE Pipeline Orchestrator — wires all modules together.

Two-phase architecture (mirrors AESE's online + retroactive batch pattern):

  Phase 1 — Online, per-buffer (streaming):
    - VAD → only speech regions go to ASR + diarization
    - Transcription (faster-whisper)
    - Speaker embedding (resemblyzer or MFCC fallback)
    - Online speaker cluster assignment (SpeakerClusterer)
    - Music detection (PANNs 2s windows)
    - Sound event detection (PANNs 1s windows)
    - Acoustic emotion (waveform features, transcript-independent)
    - Audio fusion (mean-pooled embedding)
    - AudioUnderstanding construction per buffer

  Phase 2 — Batch, after full-track processing:
    - Global speaker cluster consolidation (merge clusters that are the same
      person but appeared different online due to limited early context)
    - SpeakerCharacterBinder resolution applied retroactively across all segments
      (mirrors AESE's apply_resolved_names() pattern)

CRITICAL LATENCY RULE:
    preload_audio() is called EXACTLY ONCE at the top of run().
    No downstream function may call preload_audio() again.
    Violation of this rule reintroduces the ASVL audio-reload bug.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .acoustic_emotion import compute_acoustic_emotion
from .aese_association import associate_with_events
from .association.speaker_character_binder import (
    SpeakerCharacterBinder,
    apply_resolved_speakers,
)
from .audio_source import (
    audio_duration_ms,
    preload_audio,
    slice_waveform,
)
from .config import load_config
from .diarization.speaker_clusterer import SpeakerClusterer
from .diarization.speaker_embedding import embed_speaker
from .fusion import fuse_audio_segment
from .importance import (
    compute_audio_importance,
    compute_context_score,
    compute_dialogue_semantic_score,
)
from .music.acoustic_features import extract_music_acoustic_features
from .music.music_detection import detect_music_regions
from .music.music_emotion import compute_music_emotion
from .sound_events.sed_model import detect_sound_events
from .speech.transcribe import transcribe_segments
from .types import (
    AUEConfig,
    AudioUnderstanding,
    MusicSegment,
    SoundEvent,
    SpeechSegment,
)
from .vad import detect_speech_segments

logger = logging.getLogger(__name__)

_DECISION_LATENCY_WARN_MS = 200.0   # warn if single buffer takes > 200ms


def _get_rss_mb() -> float:
    """Return current RSS in MB (cross-platform via psutil)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def _build_music_segments(
    music_regions: List[dict],
    waveform: np.ndarray,
    sr: int,
    music_threshold: float,
) -> List[MusicSegment]:
    """Convert raw music detection dicts into MusicSegment objects."""
    segments = []
    for region in music_regions:
        mp = region.get("music_probability", 0.0)
        if mp < music_threshold:
            continue
        start_ms = region["start_ms"]
        end_ms = region["end_ms"]

        # Acoustic features for this region
        chunk = slice_waveform(waveform, sr, start_ms, end_ms)
        feats = extract_music_acoustic_features(chunk, sr)

        # Music emotion (multi-label, never single string)
        emotion = compute_music_emotion(
            tempo_bpm=feats.get("tempo_bpm"),
            rms_energy=feats.get("rms_energy", 0.0),
            chroma_mode=feats.get("chroma_mode", 0),
        )

        segments.append(MusicSegment(
            start_ms=start_ms,
            end_ms=end_ms,
            music_probability=mp,
            emotion=emotion,
            intensity=feats.get("rms_energy", 0.0),
            tension=emotion.get("tense", 0.0) * 0.5 + emotion.get("suspense", 0.0) * 0.5,
            tempo_bpm=feats.get("tempo_bpm"),
            confidence=mp,
        ))
    return segments


def _run_sed_on_buffer(
    waveform: np.ndarray,
    sr: int,
    buffer_start_ms: float,
    buffer_end_ms: float,
    window_seconds: float,
    confidence_threshold: float,
    checkpoint_dir: Optional[str],
) -> List[SoundEvent]:
    """Run SED on sliding windows over a buffer range."""
    events = []
    window_samples = int(window_seconds * sr)
    hop_samples = window_samples // 2  # 50% overlap for coverage

    start_sample = int(buffer_start_ms / 1000.0 * sr)
    end_sample = int(buffer_end_ms / 1000.0 * sr)

    for pos in range(start_sample, end_sample - window_samples + 1, hop_samples):
        chunk = waveform[pos: pos + window_samples]
        s_ms = pos / sr * 1000.0
        e_ms = (pos + window_samples) / sr * 1000.0

        detected = detect_sound_events(
            chunk, sr, s_ms, e_ms,
            confidence_threshold=confidence_threshold,
            checkpoint_dir=checkpoint_dir,
        )
        events.extend(detected)

    return events


def _process_buffer(
    waveform: np.ndarray,
    sr: int,
    buffer_start_ms: float,
    buffer_end_ms: float,
    config: AUEConfig,
    clusterer: SpeakerClusterer,
    binder: SpeakerCharacterBinder,
    aese_events: List,
    segment_counter: list,  # mutable counter [int]
) -> AudioUnderstanding:
    """
    Process a single rolling buffer window.
    Returns one AudioUnderstanding object for this buffer.
    """
    t0 = time.perf_counter()

    # Slice the buffer from preloaded waveform (no re-decode)
    buffer_waveform = slice_waveform(waveform, sr, buffer_start_ms, buffer_end_ms)

    # --- VAD: detect speech regions within this buffer ---
    t_vad = time.perf_counter()
    speech_regions_local = detect_speech_segments(
        buffer_waveform, sr, threshold=config.vad_speech_threshold
    )
    # Convert local-buffer timestamps to global timestamps
    speech_regions = [
        (buffer_start_ms + s, buffer_start_ms + e)
        for s, e in speech_regions_local
    ]
    _t_vad = (time.perf_counter() - t_vad) * 1000.0

    # --- ASR: transcribe only speech regions (gated by VAD) ---
    t_asr = time.perf_counter()
    speech_segments: List[SpeechSegment] = []
    if speech_regions:
        # Slice each speech region from global waveform
        local_regions = [
            (s - buffer_start_ms, e - buffer_start_ms)
            for s, e in speech_regions
        ]
        segs = transcribe_segments(
            buffer_waveform, sr, local_regions, model_size=config.whisper_model_size
        )
        # Fix timestamps to global timeline and assign IDs
        for seg, (global_start, global_end) in zip(segs, speech_regions):
            seg.start_ms = global_start
            seg.end_ms = global_end
            seg.segment_id = f"SP_{segment_counter[0]:05d}"
            segment_counter[0] += 1
            # Adjust word timestamps to global timeline
            for w in seg.words:
                w.start_ms += buffer_start_ms
                w.end_ms += buffer_start_ms
        speech_segments = segs
    _t_asr = (time.perf_counter() - t_asr) * 1000.0

    # --- Speaker embedding + online clustering (VAD-gated) ---
    t_diar = time.perf_counter()
    for seg in speech_segments:
        chunk = slice_waveform(waveform, sr, seg.start_ms, seg.end_ms)
        if len(chunk) > 0:
            emb = embed_speaker(chunk, sr)
            if emb is not None:
                seg.speaker_id = clusterer.assign(emb)
    _t_diar = (time.perf_counter() - t_diar) * 1000.0

    # --- Speaker↔Character binding evidence (online accumulation) ---
    for seg in speech_segments:
        for event in aese_events:
            ev_start = _get_ev_time(event, "start_time_ms")
            ev_end = _get_ev_time(event, "end_time_ms")
            if seg.start_ms < ev_end and ev_start < seg.end_ms:
                binder.observe(seg.speaker_id, event)

    # --- Music detection (sliding windows over this buffer) ---
    t_music = time.perf_counter()
    music_regions = detect_music_regions(
        waveform=buffer_waveform,
        sr=sr,
        window_seconds=config.music_window_seconds,
        threshold=config.music_detection_threshold,
        hop_seconds=config.music_window_seconds / 2.0,
        checkpoint_dir=config.panns_checkpoint_dir,
    )
    # Fix timestamps to global timeline
    for r in music_regions:
        r["start_ms"] += buffer_start_ms
        r["end_ms"] += buffer_start_ms

    music_segments = _build_music_segments(
        music_regions, waveform, sr, config.music_detection_threshold
    )
    _t_music = (time.perf_counter() - t_music) * 1000.0

    # --- Sound event detection (sliding windows) ---
    t_sed = time.perf_counter()
    sound_events = _run_sed_on_buffer(
        waveform, sr,
        buffer_start_ms, buffer_end_ms,
        config.sed_window_seconds,
        config.sound_event_confidence_threshold,
        config.panns_checkpoint_dir,
    )
    _t_sed = (time.perf_counter() - t_sed) * 1000.0

    # --- Acoustic emotion (whole buffer, transcript-independent) ---
    t_emo = time.perf_counter()
    acoustic_emotion = compute_acoustic_emotion(buffer_waveform, sr, transcript=None)
    _t_emo = (time.perf_counter() - t_emo) * 1000.0

    # --- Audio fusion ---
    t_fuse = time.perf_counter()
    audio_embedding = fuse_audio_segment(
        speech_segments, music_segments, sound_events, acoustic_emotion
    )
    _t_fuse = (time.perf_counter() - t_fuse) * 1000.0

    # --- Compute buffer-level confidence ---
    n_total = len(speech_segments) + len(music_segments) + len(sound_events)
    if n_total > 0:
        confidences = (
            [s.asr_confidence + 2.0 / 2.0 for s in speech_segments if not s.low_confidence] +
            [m.confidence for m in music_segments] +
            [e.confidence for e in sound_events if e.event_type != "ambiguous"]
        )
        confidence = float(np.mean(confidences)) if confidences else 0.0
    else:
        confidence = 0.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    if total_ms > _DECISION_LATENCY_WARN_MS:
        logger.warning(
            "AUE pipeline: buffer [%.0f–%.0f ms] took %.0f ms "
            "(VAD=%.0f ASR=%.0f diar=%.0f music=%.0f SED=%.0f emo=%.0f fuse=%.0f)",
            buffer_start_ms, buffer_end_ms, total_ms,
            _t_vad, _t_asr, _t_diar, _t_music, _t_sed, _t_emo, _t_fuse,
        )
    else:
        logger.debug(
            "AUE pipeline: buffer [%.0f–%.0f ms] | "
            "speech=%d music=%d SED=%d | %.0f ms total",
            buffer_start_ms, buffer_end_ms,
            len(speech_segments), len(music_segments), len(sound_events),
            total_ms,
        )

    return AudioUnderstanding(
        start_ms=buffer_start_ms,
        end_ms=buffer_end_ms,
        speech_segments=speech_segments,
        speakers=sorted(set(s.speaker_id for s in speech_segments)),
        music_segments=music_segments,
        sound_events=sound_events,
        acoustic_emotion=acoustic_emotion,
        audio_embedding=audio_embedding,
        confidence=confidence,
    )


def _get_ev_time(event: Any, field: str) -> float:
    if isinstance(event, dict):
        return float(event.get(field, 0.0))
    return float(getattr(event, field, 0.0))


def _consolidate_speakers_and_bind(
    audio_understandings: List[AudioUnderstanding],
    aese_events: List,
    clusterer: SpeakerClusterer,
    binder: SpeakerCharacterBinder,
) -> None:
    """
    Batch consolidation phase — mirrors AESE's retroactive naming pass.

    1. Global speaker cluster merge: find cluster pairs that are the same person
       but were kept separate online (limited early context).
    2. Apply resolved speaker↔character bindings retroactively across all segments.

    This runs AFTER full-track processing. Online decisions (anonymous SPK_XX) are
    never changed during the online phase — only the final display-time labels change.
    """
    # Step 1: Global cluster merge
    merge_candidates = clusterer.get_global_merge_candidates(global_threshold_factor=0.85)
    for idx_a, idx_b in merge_candidates:
        if idx_a >= clusterer.cluster_count() or idx_b >= clusterer.cluster_count():
            continue
        old_label = clusterer.cluster_labels[idx_b]
        new_label = clusterer.cluster_labels[idx_a]
        clusterer.merge_clusters(idx_a, idx_b)
        # Relabel all segments using old_label
        for au in audio_understandings:
            for seg in au.speech_segments:
                if seg.speaker_id == old_label:
                    seg.speaker_id = new_label
            au.speakers = sorted(set(s.speaker_id for s in au.speech_segments))

    # Step 2: Apply resolved bindings
    all_segments = [
        seg
        for au in audio_understandings
        for seg in au.speech_segments
    ]
    apply_resolved_speakers(all_segments, binder)


def run(
    video_path: str,
    aese_events: List,
    config: Optional[AUEConfig] = None,
) -> List[AudioUnderstanding]:
    """
    Main AUE pipeline: decode audio once, process in rolling buffers,
    then consolidate speakers and bind characters.

    Args:
        video_path:   Path to the input video file.
        aese_events:  List of AESE Event objects (dataclasses or dicts).
        config:       AUEConfig instance. Loads default config if None.

    Returns:
        List of AudioUnderstanding objects, one per rolling buffer window.

    Latency contract:
        preload_audio() is called EXACTLY ONCE at the top of this function.
        This function must never be called from within a loop downstream.
    """
    if config is None:
        config = load_config()

    run_start = time.perf_counter()

    # ─────────────────────────────────────────────────────────────────────
    # STEP 1: DECODE AUDIO EXACTLY ONCE
    # This is the most important latency rule in this module.
    # Every downstream stage slices from 'waveform' in memory.
    # ─────────────────────────────────────────────────────────────────────
    waveform, sr = preload_audio(video_path)
    total_audio_ms = audio_duration_ms(waveform, sr)

    if total_audio_ms == 0:
        logger.error("AUE pipeline: empty waveform from %s — aborting.", video_path)
        return []

    logger.info(
        "AUE pipeline: starting on %s | audio=%.1f s | "
        "buffer=%.0f s | model=%s",
        os.path.basename(video_path),
        total_audio_ms / 1000.0,
        config.rolling_buffer_seconds,
        config.whisper_model_size,
    )

    # ─────────────────────────────────────────────────────────────────────
    # STEP 2: ONLINE ROLLING-BUFFER PHASE
    # ─────────────────────────────────────────────────────────────────────
    clusterer = SpeakerClusterer(distance_threshold=config.speaker_cluster_threshold)
    binder = SpeakerCharacterBinder(min_votes_to_resolve=2)
    segment_counter = [0]  # mutable counter shared across buffer calls
    audio_understandings: List[AudioUnderstanding] = []

    buffer_ms = config.rolling_buffer_seconds * 1000.0
    buffer_start = 0.0
    peak_rss_mb = 0.0
    buffer_count = 0

    while buffer_start < total_audio_ms:
        buffer_end = min(buffer_start + buffer_ms, total_audio_ms)

        au = _process_buffer(
            waveform=waveform,
            sr=sr,
            buffer_start_ms=buffer_start,
            buffer_end_ms=buffer_end,
            config=config,
            clusterer=clusterer,
            binder=binder,
            aese_events=aese_events,
            segment_counter=segment_counter,
        )
        audio_understandings.append(au)

        buffer_count += 1
        buffer_start = buffer_end

        # RSS tracking every 5 buffers
        if buffer_count % 5 == 0:
            rss = _get_rss_mb()
            if rss > peak_rss_mb:
                peak_rss_mb = rss

    # ─────────────────────────────────────────────────────────────────────
    # STEP 3: BATCH CONSOLIDATION PHASE
    # Mirrors AESE's retroactive naming pass.
    # ─────────────────────────────────────────────────────────────────────
    _consolidate_speakers_and_bind(audio_understandings, aese_events, clusterer, binder)

    # ─────────────────────────────────────────────────────────────────────
    # END-OF-RUN STATS + RTF MEASUREMENT
    # Per spec §6: measure RTF honestly, don't assume.
    # ─────────────────────────────────────────────────────────────────────
    total_processing_s = time.perf_counter() - run_start
    audio_duration_s = total_audio_ms / 1000.0
    rtf = total_processing_s / max(audio_duration_s, 0.001)

    total_speech = sum(len(au.speech_segments) for au in audio_understandings)
    total_music = sum(len(au.music_segments) for au in audio_understandings)
    total_sed = sum(len(au.sound_events) for au in audio_understandings)

    logger.info(
        "AUE run complete: %.1f s audio processed in %.1f s | RTF=%.2f | "
        "%d buffers | speech=%d music=%d SED=%d | "
        "speakers=%d | peak_rss=%.0f MB",
        audio_duration_s,
        total_processing_s,
        rtf,
        buffer_count,
        total_speech,
        total_music,
        total_sed,
        clusterer.cluster_count(),
        peak_rss_mb,
    )

    if rtf >= config.rtf_warn_threshold:
        logger.warning(
            "AUE: RTF=%.2f >= %.2f — pipeline is NOT real-time capable on this hardware. "
            "Consider: smaller Whisper model, GPU inference, or reducing buffer size. "
            "See DECISIONS.md §AUE-8 for RTF benchmark details.",
            rtf, config.rtf_warn_threshold,
        )
    else:
        logger.info("AUE: RTF=%.2f < %.2f — real-time capable ✓", rtf, config.rtf_warn_threshold)

    return audio_understandings
