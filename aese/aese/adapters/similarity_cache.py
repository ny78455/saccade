"""
aese/adapters/similarity_cache.py
Visual-similarity cache for keyframe scene-label deduplication (Fix 3 — §22.3).

Contract:
  - Cache ONLY the location_label (scene classification). Never the caption text.
  - Caption text is ALWAYS generated fresh per event with its own character labels,
    dialogue context, and action label. Reusing stale captions is the root cause
    of hallucination bugs fixed in prior rounds; this cache does NOT reintroduce it.
  - A cache HIT means: this event's keyframe is visually near-identical to one
    already classified → the physical location has not meaningfully changed → we can
    skip the expensive VLM scene-classification call and reuse the label.
  - A cache MISS means: first time seeing this kind of frame → run full VLM call.

Embedding source:
  Uses the event's pre-computed `event_embedding` (pooled CLIP multimodal embedding
  from Phase 1). This is free — Phase 1 already computed it. No second CLIP call
  is needed for cache lookup, which is the key efficiency gain here.

Thread safety:
  Not thread-safe. Call only from the enrichment phase, which runs sequentially
  by default. If overlap (Fix 5) runs enrichment in a thread pool, each chunk
  gets its own cache instance (no shared state).

Threshold:
  Default 0.92 cosine similarity. This is deliberately tight:
    - 0.95+ would be "almost identical" (same shot, different frame)
    - 0.92-0.95 catches "same room, same people, similar framing" across
      consecutive events in a dialogue-heavy scene (the Andhadhun clip pattern)
    - Below 0.92, the scenes may have drifted (different room, outdoors/indoors,
      lighting change) and a fresh classification is warranted
  Tune via AESEConfig.similarity_cache_threshold if needed.

See DECISIONS.md §22.3 for threshold calibration notes.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KeyframeSimilarityCache:
    """
    Cache scene-classification labels for visually near-duplicate event keyframes.

    Only caches `location_label` (str). Captions are always generated fresh.

    Args:
        similarity_threshold: Cosine similarity floor for a cache hit. Default 0.92.
        max_size: Maximum number of cached entries. Oldest entries are evicted when
            the cache is full (FIFO). Default 64 (more than enough for a feature film).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 64,
    ) -> None:
        self.threshold = similarity_threshold
        self.max_size = max_size
        self._embeddings: List[np.ndarray] = []
        self._labels: List[str] = []
        self._hits: int = 0
        self._misses: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(self, event_embedding: np.ndarray) -> Optional[str]:
        """
        Look up a location_label for an event whose embedding is given.

        Returns the cached label if any stored embedding has cosine similarity
        >= self.threshold, otherwise returns None.

        Args:
            event_embedding: Pre-computed event embedding from Phase 1 (float32, L2-normalised).

        Returns:
            Cached location_label str, or None on a cache miss.
        """
        if not self._embeddings or event_embedding is None:
            self._misses += 1
            return None

        try:
            norm = float(np.linalg.norm(event_embedding))
            if norm < 1e-8:
                self._misses += 1
                return None
            q = event_embedding / norm

            best_sim = -1.0
            best_label: Optional[str] = None
            for emb, label in zip(self._embeddings, self._labels):
                sim = float(np.dot(q, emb))
                if sim > best_sim:
                    best_sim = sim
                    best_label = label

            if best_sim >= self.threshold:
                self._hits += 1
                logger.debug(
                    "AESE similarity_cache: HIT (sim=%.4f >= %.3f) → label=%r",
                    best_sim, self.threshold, best_label,
                )
                return best_label

            self._misses += 1
            logger.debug(
                "AESE similarity_cache: MISS (best_sim=%.4f < %.3f)",
                best_sim, self.threshold,
            )
            return None
        except Exception as exc:
            logger.debug("AESE similarity_cache: lookup error: %s", exc)
            self._misses += 1
            return None

    def store(self, event_embedding: np.ndarray, location_label: str) -> None:
        """
        Store a (embedding, location_label) pair.

        The embedding is L2-normalised before storage so dot-product in
        lookup() equals cosine similarity. If the cache is full, the oldest
        entry is evicted (FIFO).

        Args:
            event_embedding: Pre-computed event embedding from Phase 1.
            location_label:  Scene classification label to cache.
        """
        if event_embedding is None:
            return
        try:
            norm = float(np.linalg.norm(event_embedding))
            if norm < 1e-8:
                return
            normalised = event_embedding / norm

            # FIFO eviction when full
            if len(self._embeddings) >= self.max_size:
                self._embeddings.pop(0)
                self._labels.pop(0)

            self._embeddings.append(normalised.astype(np.float32))
            self._labels.append(location_label)
        except Exception as exc:
            logger.debug("AESE similarity_cache: store error: %s", exc)

    # ------------------------------------------------------------------
    # Stats / diagnostics
    # ------------------------------------------------------------------

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction [0, 1]. Returns 0.0 if no lookups yet."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        """Number of cached entries."""
        return len(self._embeddings)

    def stats(self) -> dict:
        """Return a dict of hit/miss stats for benchmarking and logging."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 4),
            "size": self.size,
        }

    def reset(self) -> None:
        """Clear all cached data and reset stats. Used between test runs."""
        self._embeddings = []
        self._labels = []
        self._hits = 0
        self._misses = 0
