"""
aese/pipeline.py
Pipeline Orchestrator -- wires all AESE modules together.

Two-phase architecture (§22.1):
    PHASE 1 (online, CPU-bound, zero VLM calls):
        FramePackets -> FeatureAggregator -> ContextBuffer
                     -> CandidateDetector (signals + fusion + confidence)
                     -> EventConstructor (min/max duration)
                     -> OnlineMerger
                     -> EventClassifier
                     -> CharacterClusterer
                     -> [raw Event list]   ← no summaries yet

    PHASE 2 (batch enrichment, GPU-bound, all VLM calls together):
        raw Events -> describe_batch() [batched VLM, BATCH_SIZE=4]
                   -> generate_summary()  [per-event, with similarity_cache location skip]
                   -> secondary-keyframe addendum (gated)
                   -> caption-based count reconciliation
                   -> schema invariant guard
                   -> [enriched Event list]

Public API (unchanged):
    run(frame_packet_stream, config, binder) -> Iterator[Event]

Internal functions:
    run_online_phase(stream, config, binder)  -> (list[Event], dict, CharacterClusterer)
    run_enrichment_phase(events, features_map, config, binder, sim_cache) -> list[Event]
    run_with_overlap(stream, config, binder, chunk_seconds)  -> list[Event]

Non-functional requirements (§8):
    - Latency: <100ms per boundary decision (logged; Phase 1 only)
    - Memory: rolling buffer -- never holds full movie in RAM
    - Streaming: no future frames peeked in Phase 1
    - Batching: Phase 2 uses describe_batch() across all events (§22.2)
    - Caching: similarity_cache skips scene-classification for near-duplicate frames (§22.3)
    - Compile: warmup() fires at startup in a background thread (§22.4)
    - Overlap: run_with_overlap() overlaps Phase 1 CPU work with Phase 2 GPU work (§22.5)
"""
from __future__ import annotations

import concurrent.futures
import itertools
import logging
import os
import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from .adapters.character_cluster import CharacterClusterer, get_character_labels_for_event
from .adapters.character_naming import (
    CharacterNameBinder,
    apply_resolved_names,
    extract_name_mentions,
)
from .adapters.caption_person_count import estimate_person_count_from_caption
from .adapters.similarity_cache import KeyframeSimilarityCache
from .aggregator import FeatureAggregator
from .boundary.candidate_detector import CandidateDetector
from .context_buffer import ContextBuffer
from .event_classifier import classify_event
from .event_constructor import EventConstructor, build_template_summary
from .event_embedding import pool_event_embedding
from .event_graph import EventGraph
from .event_merge import OnlineMerger
from .keyframe import select_keyframe, select_keyframe_salient, needs_secondary_frame
from .summary import generate_summary, caption_frame_delta, _summary_call_counter
from .types import AESEConfig, Event, FramePacket

logger = logging.getLogger(__name__)

_DECISION_LATENCY_WARN_MS = 100.0
_MIN_FEATURES_FOR_BOUNDARY = 2  # need at least 2 seconds to make a boundary decision

# §22.2 — chunk size for batched VLM calls in Phase 2.
# Must match or be <= gemma4.BATCH_SIZE to avoid OOM.
# Imported lazily to avoid hard-coupling to gemma4 at module load time.
_ENRICHMENT_BATCH_SIZE = 4


def _get_rss_mb() -> float:
    """Return current RSS in MB (cross-platform via psutil)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return -1.0


def _chunked(iterable, n: int):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while True:
        chunk = list(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


# ===========================================================================
# Phase 1 helpers (online, zero VLM calls)
# ===========================================================================

def _finalize_event_online(
    event: Event,
    next_id: int,
    clusterer: Optional[CharacterClusterer] = None,
    event_features_map: Optional[dict] = None,
    binder: Optional[CharacterNameBinder] = None,
) -> None:
    """
    Phase 1 finalization: assign contiguous ID, run character clustering, feed
    name evidence to the binder.  NO VLM calls whatsoever.

    After this call the event has:
      - A stable event_id
      - character_labels populated from face embeddings (when clusterer is available)
      - Name evidence accumulated in binder (for retroactive relabeling in cli.py)
      - summary = "" (populated by Phase 2)
      - location_label = "" (populated by Phase 2)

    Mirrors the §21 schema invariant: character_labels are cleared here too if
    max_characters_seen == 0, so Phase 2 never sees stale labels.
    """
    event.event_id = next_id

    # --- Character clustering ---
    # Guard: skip when the event had no real image data (manifest-replay mode).
    if clusterer is not None and event_features_map is not None \
            and event.character_data_available:
        feats = event_features_map.get(event.event_id, [])
        face_embeddings_per_second = [getattr(tf, "face_embeddings", []) for tf in feats]
        event.character_labels = get_character_labels_for_event(face_embeddings_per_second, clusterer)

    # --- Name evidence accumulation ---
    # Only when exactly one cluster visible (unambiguous attribution, §4, §8).
    if binder is not None and event.character_labels:
        dialogue_text = getattr(event, "dialogue_text", None) or ""
        if dialogue_text and len(event.character_labels) == 1:
            for mention in extract_name_mentions(dialogue_text, event.start_time_ms):
                binder.observe(mention, event.character_labels)

    # --- Schema invariant (Phase 1 copy, mirrors Phase 2 invariant) ---
    if (event.max_characters_seen is None or event.max_characters_seen == 0) \
            and event.character_labels:
        logger.warning(
            "AESE: schema invariant violated for event %d (Phase 1) — "
            "clearing stale character_labels=%s.",
            event.event_id, event.character_labels,
        )
        event.character_labels = []


def run_online_phase(
    frame_packet_stream: Iterator[FramePacket],
    config: AESEConfig,
    binder: Optional[CharacterNameBinder] = None,
) -> Tuple[List[Event], Dict[int, list], CharacterClusterer]:
    """
    Phase 1: consume all FramePackets, produce raw (unenriched) Events.

    ZERO VLM calls are made here. The returned events have valid timestamps,
    boundaries, event types, character labels, and event embeddings, but
    summary == "" and location_label == "".

    Phase 2 (run_enrichment_phase) enriches the returned events in place.

    Returns:
        (raw_events, event_features_map, clusterer)
        raw_events:          list of Event objects in timeline order.
        event_features_map:  {event_id -> List[TemporalFeature]} for Phase 2's
                              secondary-keyframe and summary helpers.
        clusterer:           CharacterClusterer instance (retains gallery state).
    """
    # Kick off warmup in background so torch.compile finishes during Phase 1 CPU work.
    _start_warmup_background()

    aggregator = FeatureAggregator(config)
    buffer = ContextBuffer(buffer_seconds=config.buffer_seconds)
    detector = CandidateDetector(config, buffer)
    constructor = EventConstructor(
        config=config,
        event_embedding_fn=pool_event_embedding,
        keyframe_fn=lambda features: select_keyframe(features, strategy="most_salient"),
    )
    merger = OnlineMerger(config)
    event_graph = EventGraph()
    clusterer = CharacterClusterer()

    raw_events: List[Event] = []
    event_features: dict = {}
    current_event_features: List = []

    prev_feature = None
    decision_latencies_ms: List[float] = []
    peak_rss_mb = 0.0
    packets_processed = 0
    events_emitted = 0

    logger.info(
        "AESE Phase 1 (online): buffer=%.0fs, threshold=%.2f, max_event=%.0fs.",
        config.buffer_seconds, config.boundary_threshold, config.maximum_event_duration_s,
    )

    def _handle_completed(event: Event) -> None:
        nonlocal events_emitted
        event_features[event.event_id] = list(
            current_event_features[:-1] or current_event_features
        )
        ev_feats = event_features.get(event.event_id, [current_event_features[-1]] if current_event_features else [])
        event.event_type = classify_event(event, ev_feats)
        finalized = merger.process(event)
        if finalized is not None:
            _finalize_event_online(finalized, events_emitted, clusterer, event_features, binder)
            event_graph.add_event(finalized)
            buffer.record_boundary(finalized.end_time_ms)
            events_emitted += 1
            raw_events.append(finalized)

    # --- Main stream loop ---
    for fp in frame_packet_stream:
        loop_start = time.perf_counter()
        packets_processed += 1

        new_features = aggregator.push_all(fp)

        for tf in new_features:
            buffer.push(tf)
            current_event_features.append(tf)

            decision = detector.update(tf, prev_feature)
            completed_events = constructor.update(tf, decision)

            for event in completed_events:
                event_features[event.event_id] = list(
                    current_event_features[:-1] or current_event_features
                )
                current_event_features = [tf]
                ev_feats = event_features.get(event.event_id, [tf])
                event.event_type = classify_event(event, ev_feats)
                finalized = merger.process(event)
                if finalized is not None:
                    _finalize_event_online(finalized, events_emitted, clusterer, event_features, binder)
                    event_graph.add_event(finalized)
                    buffer.record_boundary(finalized.end_time_ms)
                    events_emitted += 1
                    raw_events.append(finalized)

            prev_feature = tf

        # Latency logging
        loop_ms = (time.perf_counter() - loop_start) * 1000.0
        decision_latencies_ms.append(loop_ms)
        if loop_ms > _DECISION_LATENCY_WARN_MS:
            logger.warning(
                "AESE Phase 1: decision loop %.1f ms > 100ms at ts=%.1f ms",
                loop_ms, fp.timestamp_ms,
            )

        if packets_processed % 50 == 0:
            rss = _get_rss_mb()
            if rss > peak_rss_mb:
                peak_rss_mb = rss

    # --- Flush aggregator trailing partial second ---
    for tf in aggregator.flush():
        buffer.push(tf)
        current_event_features.append(tf)
        decision = detector.update(tf, prev_feature)
        completed_events = constructor.update(tf, decision)
        for event in completed_events:
            ev_feats = event_features.get(event.event_id, current_event_features)
            event.event_type = classify_event(event, ev_feats)
            finalized = merger.process(event)
            if finalized is not None:
                _finalize_event_online(finalized, events_emitted, clusterer, event_features, binder)
                event_graph.add_event(finalized)
                buffer.record_boundary(finalized.end_time_ms)
                events_emitted += 1
                raw_events.append(finalized)
        prev_feature = tf

    # --- Close final open event ---
    final_event = constructor.close()
    if final_event is not None:
        ev_feats = current_event_features if current_event_features else []
        final_event.event_type = classify_event(final_event, ev_feats)
        finalized = merger.process(final_event)
        if finalized is not None:
            _finalize_event_online(finalized, events_emitted, clusterer, event_features, binder)
            event_graph.add_event(finalized)
            events_emitted += 1
            raw_events.append(finalized)

    # Flush merger's held event
    last_held = merger.finalize()
    if last_held is not None:
        last_held.event_type = classify_event(last_held, [])
        _finalize_event_online(last_held, events_emitted, clusterer, event_features, binder)
        event_graph.add_event(last_held)
        events_emitted += 1
        raw_events.append(last_held)

    # Phase 1 stats
    if decision_latencies_ms:
        sorted_lat = sorted(decision_latencies_ms)
        p95 = sorted_lat[min(int(len(sorted_lat) * 0.95), len(sorted_lat) - 1)]
        logger.info(
            "AESE Phase 1 complete: %d packets, %d raw events | p95=%.1f ms | RSS=%.1f MB",
            packets_processed, len(raw_events), p95, peak_rss_mb,
        )
        if p95 > _DECISION_LATENCY_WARN_MS:
            logger.warning("AESE Phase 1: p95 latency %.1f ms > 100ms target.", p95)

    return raw_events, event_features, clusterer


# ===========================================================================
# Phase 2 helpers (enrichment, GPU-bound, all VLM calls)
# ===========================================================================

def _build_context(event: Event) -> dict:
    """Build the context dict for describe_batch() / describe_scene_and_caption()."""
    return {
        "action_label": getattr(event, "action_label", "static") or "static",
        "dialogue_text": getattr(event, "dialogue_text", None),
        "scene_labels": None,  # uses SCENE_LABELS default
    }


def _enrich_one_event(
    event: Event,
    location_label: str,
    caption: str,
    event_features_map: dict,
) -> None:
    """
    Apply the enrichment result (location_label + caption) to an event and run
    the remaining post-enrichment steps:
      - Secondary keyframe addendum (gated, rare)
      - Caption-based character count reconciliation
      - Schema invariant guard
    """
    event.location_label = location_label

    # If the batch/cache path produced a non-empty caption, use it as the summary.
    # Fall back to generate_summary() (full system-prompt path) if empty.
    if caption and len(caption.strip()) >= 5:
        event.summary = caption.strip()
    else:
        # Full system-prompt summary as fallback (also handles VLM-unavailable case)
        keyframe_image = None
        if event.key_frame is not None and hasattr(event.key_frame, 'shape') and event.key_frame.ndim == 3:
            keyframe_image = event.key_frame
        event.summary = generate_summary(event, keyframe_image)
        if not event.summary:
            event.summary = build_template_summary(
                event_type=event.event_type,
                scene_label=event.location_label or "unknown",
                max_characters_seen=event.max_characters_seen,
            )

    # --- Secondary keyframe addendum (Fix §20.2) ---
    keyframe_image = None
    if event.key_frame is not None and hasattr(event.key_frame, 'shape') and event.key_frame.ndim == 3:
        keyframe_image = event.key_frame
    event_feats = event_features_map.get(event.event_id, [])
    if event_feats and keyframe_image is not None:
        duration_s = event.duration_ms / 1000.0
        sal = [f.motion_score + f.novelty_score for f in event_feats]
        primary_idx = int(np.argmax(sal)) if sal else 0
        secondary_idx = needs_secondary_frame(event_feats, primary_idx, duration_s)
        if secondary_idx is not None:
            sec_feat = event_feats[secondary_idx]
            secondary_frame = getattr(sec_feat, "representative_image", None)
            if secondary_frame is not None:
                logger.info(
                    "AESE Phase 2: secondary keyframe gate fired for event %d "
                    "(duration=%.1fs, secondary_idx=%d, salience=%.2f)",
                    event.event_id, duration_s, secondary_idx,
                    sal[secondary_idx] if sal else 0.0,
                )
                addendum = caption_frame_delta(keyframe_image, secondary_frame)
                if addendum:
                    event.summary = f"{event.summary} {addendum}".strip()

    # --- Caption-based count reconciliation ---
    caption_estimate = estimate_person_count_from_caption(event.summary)
    if caption_estimate is not None:
        current = event.max_characters_seen or 0
        if caption_estimate > current:
            logger.info(
                "AESE Phase 2: caption implies %d people but face detector found %d "
                "for event %d -- raising max_characters_seen.",
                caption_estimate, current, event.event_id,
            )
            event.max_characters_seen = caption_estimate

    # --- Schema invariant: stale labels when no characters seen ---
    if (event.max_characters_seen is None or event.max_characters_seen == 0) \
            and event.character_labels:
        logger.warning(
            "AESE Phase 2: schema invariant violated for event %d — "
            "character_labels=%s but max_characters_seen=%s. Clearing stale labels.",
            event.event_id, event.character_labels, event.max_characters_seen,
        )
        event.character_labels = []


def run_enrichment_phase(
    raw_events: List[Event],
    event_features_map: dict,
    config: AESEConfig,
    binder: Optional[CharacterNameBinder] = None,
    sim_cache: Optional[KeyframeSimilarityCache] = None,
) -> List[Event]:
    """
    Phase 2: enrich all raw events with VLM-generated summaries and scene labels.

    Batches VLM calls across events using describe_batch() (§22.2).
    Uses sim_cache to skip scene-classification for near-duplicate frames (§22.3).
    Falls back gracefully when VLM is unavailable (template summaries).

    Args:
        raw_events:       Output of run_online_phase().
        event_features_map: {event_id -> List[TemporalFeature]}.
        config:           AESEConfig.
        binder:           Optional CharacterNameBinder (already populated in Phase 1).
        sim_cache:        Optional KeyframeSimilarityCache for scene-label deduplication.

    Returns:
        The same list, mutated in place with summaries / location_labels added.
    """
    if not raw_events:
        return raw_events

    t0 = time.perf_counter()
    logger.info("AESE Phase 2: enriching %d events (batch_size=%d).", len(raw_events), _ENRICHMENT_BATCH_SIZE)

    try:
        from .adapters.vlm_router import vlm_available, describe_batch as router_describe_batch
        vlm_ok = vlm_available()
    except Exception:
        vlm_ok = False

    if not vlm_ok:
        # No VLM — apply template summaries directly
        for event in raw_events:
            event.location_label = event.location_label or "unknown"
            event.summary = build_template_summary(
                event_type=event.event_type,
                scene_label=event.location_label,
                max_characters_seen=event.max_characters_seen,
            )
            _enrich_one_event(event, event.location_label, event.summary, event_features_map)
        logger.info("AESE Phase 2: VLM unavailable — template summaries applied.")
        return raw_events

    # --- Batched VLM enrichment ---
    # For each event:
    #   1. Check sim_cache for location_label hit (§22.3)
    #   2. Cache HIT  → use cached location_label; generate caption via batch (no scene call)
    #   3. Cache MISS → include in full describe_batch() call (scene + caption together)
    #
    # We collect batch inputs, call describe_batch() per chunk, then apply results.

    # Annotate each event with whether it's a cache hit and the cached label
    cache_hits: List[Optional[str]] = []  # None = miss, str = hit with cached label
    if sim_cache is not None:
        for event in raw_events:
            if event.event_embedding is not None:
                hit = sim_cache.lookup(event.event_embedding)
            else:
                hit = None
            cache_hits.append(hit)
    else:
        cache_hits = [None] * len(raw_events)

    # Separate cache hits and misses for batch processing
    miss_indices = [i for i, h in enumerate(cache_hits) if h is None]
    miss_events  = [raw_events[i] for i in miss_indices]
    miss_images  = []
    miss_contexts = []

    for ev in miss_events:
        img = None
        if ev.key_frame is not None and hasattr(ev.key_frame, 'shape') and ev.key_frame.ndim == 3:
            if ev.key_frame.max() >= 5:
                img = ev.key_frame
        miss_images.append(img)
        miss_contexts.append(_build_context(ev))

    # Batch the misses in chunks of _ENRICHMENT_BATCH_SIZE
    miss_results: List[Tuple[str, str]] = []
    for chunk_imgs, chunk_ctxs in zip(
        _chunked(miss_images, _ENRICHMENT_BATCH_SIZE),
        _chunked(miss_contexts, _ENRICHMENT_BATCH_SIZE),
    ):
        try:
            chunk_results = router_describe_batch(chunk_imgs, chunk_ctxs)
        except Exception as exc:
            logger.warning("AESE Phase 2: describe_batch chunk failed (%s) -- template fallback for chunk.", exc)
            chunk_results = [("unknown", "")] * len(chunk_imgs)
        miss_results.extend(chunk_results)

    # Map results back to miss_indices
    miss_result_map: Dict[int, Tuple[str, str]] = {}
    for orig_idx, result in zip(miss_indices, miss_results):
        miss_result_map[orig_idx] = result

    # Apply all results
    for i, event in enumerate(raw_events):
        if cache_hits[i] is not None:
            # Cache hit: reuse location label only; caption still needs generation
            cached_location = cache_hits[i]
            logger.debug(
                "AESE Phase 2: sim_cache HIT for event %d -- location_label=%r from cache.",
                event.event_id, cached_location,
            )
            # Caption: use generate_summary() which uses the full system prompt.
            # This is intentional: the contract prohibits reusing stale caption text.
            keyframe_image = None
            if event.key_frame is not None and hasattr(event.key_frame, 'shape') and event.key_frame.ndim == 3:
                if event.key_frame.max() >= 5:
                    keyframe_image = event.key_frame
            fresh_caption = generate_summary(event, keyframe_image) if keyframe_image is not None else ""
            _enrich_one_event(event, cached_location, fresh_caption, event_features_map)
        else:
            # Cache miss: use batch result
            location, caption = miss_result_map.get(i, ("unknown", ""))
            _enrich_one_event(event, location or "unknown", caption, event_features_map)
            # Store in cache (only if we got a real location back)
            if sim_cache is not None and location and location != "unknown" \
                    and event.event_embedding is not None:
                sim_cache.store(event.event_embedding, location)

    elapsed = time.perf_counter() - t0
    cache_hit_count = sum(1 for h in cache_hits if h is not None)
    logger.info(
        "AESE Phase 2 complete: %d events enriched in %.2fs "
        "(cache hits: %d/%d, miss batches: %d).",
        len(raw_events), elapsed,
        cache_hit_count, len(raw_events),
        len(miss_indices),
    )
    if sim_cache is not None:
        logger.info("AESE Phase 2: sim_cache stats: %s", sim_cache.stats())

    return raw_events


# ===========================================================================
# Public API
# ===========================================================================

def _start_warmup_background() -> None:
    """
    Fire the Gemma-4 warmup (torch.compile JIT trigger) in a background daemon
    thread so it overlaps with Phase 1 CPU work.  No-op if not using gemma4.
    """
    try:
        from .adapters.vlm_router import get_backend
        if get_backend() != "gemma4":
            return
        from .adapters import gemma4
        t = threading.Thread(target=gemma4.warmup, daemon=True, name="gemma4-warmup")
        t.start()
        logger.debug("AESE: Gemma-4 warmup started in background thread.")
    except Exception as exc:
        logger.debug("AESE: warmup thread start failed (non-fatal): %s", exc)


def run(
    frame_packet_stream: Iterator[FramePacket],
    config: AESEConfig,
    binder: Optional[CharacterNameBinder] = None,
) -> Iterator[Event]:
    """
    Main AESE pipeline: consume FramePackets, yield completed enriched Events.

    Public API — unchanged from the single-phase version.  Internally now
    calls run_online_phase() followed by run_enrichment_phase() so all VLM
    calls are batched together in Phase 2 instead of interleaved one-at-a-time
    as events close during Phase 1.

    Args:
        frame_packet_stream: Iterator of FramePacket objects.
        config:              AESEConfig instance.
        binder:              Optional CharacterNameBinder for retroactive naming.

    Yields:
        Enriched Event objects in timeline order.
    """
    sim_cache = KeyframeSimilarityCache(
        similarity_threshold=getattr(config, "similarity_cache_threshold", 0.92),
    )

    raw_events, features_map, clusterer = run_online_phase(
        frame_packet_stream, config, binder
    )

    enriched = run_enrichment_phase(
        raw_events, features_map, config, binder, sim_cache
    )

    yield from enriched


def run_with_overlap(
    frame_packet_stream: Iterator[FramePacket],
    config: AESEConfig,
    binder: Optional[CharacterNameBinder] = None,
    chunk_seconds: float = 60.0,
) -> List[Event]:
    """
    Fix 5 / §22.5 — CPU/GPU work overlap via chunked processing.

    Processes the FramePacket stream in time-chunks of ~chunk_seconds each.
    After Phase 1 completes for chunk N, enrichment (Phase 2, GPU-bound) is
    submitted to a background thread while Phase 1 continues on chunk N+1.
    This overlaps CPU-bound Phase 1 work with GPU-bound Phase 2 work.

    Only beneficial when CUDA is available. When CUDA is absent, this degrades
    to simple sequential processing (same result, slightly more overhead).
    The benchmark harness (eval/benchmark_speedup.py) measures whether the
    overlap actually helps on this hardware — it may not, if Phase 2 is very
    fast or if there is only one event per chunk.

    Args:
        frame_packet_stream: Iterator of FramePacket objects.
        config:              AESEConfig instance.
        binder:              Optional CharacterNameBinder.
        chunk_seconds:       Approximate time-slice length per Phase 1 chunk.
                             Smaller = more overlap opportunities; larger = more
                             events per batch (better GPU utilization).

    Returns:
        Fully enriched list[Event] in timeline order (same as run()).
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except ImportError:
        cuda_available = False

    if not cuda_available:
        logger.info(
            "AESE run_with_overlap: CUDA not available -- running sequential two-phase pipeline."
        )
        return list(run(frame_packet_stream, config, binder))

    logger.info(
        "AESE run_with_overlap: CUDA available, chunk_seconds=%.0f, "
        "using CPU/GPU overlap.",
        chunk_seconds,
    )

    # Bucket incoming FramePackets into time-chunks
    def _chunk_stream(stream: Iterator[FramePacket], chunk_s: float):
        """Yield lists of FramePackets, one list per chunk_s time window."""
        chunk: List[FramePacket] = []
        chunk_start_ms: Optional[float] = None
        for fp in stream:
            if chunk_start_ms is None:
                chunk_start_ms = fp.timestamp_ms
            if fp.timestamp_ms - chunk_start_ms >= chunk_s * 1000.0 and chunk:
                yield chunk
                chunk = []
                chunk_start_ms = fp.timestamp_ms
            chunk.append(fp)
        if chunk:
            yield chunk

    sim_cache = KeyframeSimilarityCache(
        similarity_threshold=getattr(config, "similarity_cache_threshold", 0.92),
    )

    # State that persists across chunks (Phase 1 must be contiguous, but we
    # process chunks sequentially in Phase 1 and overlap Phase 2 in a thread).
    all_raw_events: List[Event] = []
    all_features_map: dict = {}

    enrichment_futures: List[concurrent.futures.Future] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for packet_chunk in _chunk_stream(frame_packet_stream, chunk_seconds):
            # Phase 1 for this chunk (runs in main thread — CPU work)
            chunk_raw, chunk_features, _ = run_online_phase(
                iter(packet_chunk), config, binder
            )
            all_raw_events.extend(chunk_raw)
            all_features_map.update(chunk_features)

            # Submit Phase 2 for this chunk to the thread pool (GPU work)
            if chunk_raw:
                future = executor.submit(
                    run_enrichment_phase,
                    chunk_raw, chunk_features, config, binder, sim_cache,
                )
                enrichment_futures.append(future)

        # Collect all enrichment results (in order)
        all_enriched: List[Event] = []
        for future in enrichment_futures:
            try:
                all_enriched.extend(future.result())
            except Exception as exc:
                logger.warning("AESE run_with_overlap: enrichment chunk failed: %s", exc)

    logger.info(
        "AESE run_with_overlap: %d events, %d chunk(s).",
        len(all_enriched), len(enrichment_futures),
    )
    return all_enriched


# ===========================================================================
# Legacy single-phase _finalize_event() — kept for backward compatibility
# with any external callers and for tests that mock it directly.
# Internally, pipeline.run() now uses the two-phase path exclusively.
# ===========================================================================

def _finalize_event(
    event: Event,
    next_id: int,
    clusterer: Optional[CharacterClusterer] = None,
    event_features_map: Optional[dict] = None,
    binder: Optional[CharacterNameBinder] = None,
) -> None:
    """
    Legacy single-phase finalization. Preserved for backward compatibility.
    New code should use _finalize_event_online() + run_enrichment_phase() instead.

    Internally calls _finalize_event_online() then _enrich_one_event() sequentially,
    reproducing the pre-refactor behaviour (one VLM call per event, in-line).
    """
    _finalize_event_online(event, next_id, clusterer, event_features_map, binder)

    # Immediate VLM call (single-event, no batching) for backward compatibility
    keyframe_image = None
    if event.key_frame is not None and hasattr(event.key_frame, 'shape') and event.key_frame.ndim == 3:
        keyframe_image = event.key_frame

    event.summary = generate_summary(event, keyframe_image)
    if not event.summary:
        event.summary = build_template_summary(
            event_type=event.event_type,
            scene_label=event.location_label or "unknown",
            max_characters_seen=event.max_characters_seen,
        )

    _enrich_one_event(event, event.location_label or "unknown", event.summary, event_features_map or {})
