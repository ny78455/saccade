"""
aue/aue/diarization/speaker_clusterer.py
Online exemplar-gallery speaker clustering for AUE.

DIRECTLY MIRRORS CharacterClusterer from AESE (aese/adapters/character_cluster.py §21.3).
Same design rationale — prevents identity drift through the same mechanism:
  - Exemplar gallery: stores up to max_exemplars raw embeddings per cluster.
  - Assignment matches against the BEST (minimum-distance) exemplar, not a drifting mean.
  - A single drifting centroid (0.9 old + 0.1 new) slowly merges embeddings of two
    different speakers across alternating shots. The gallery approach is mathematically
    immune to this: the original exemplars are never averaged.

Why not full pyannote diarization:
  pyannote.audio's offline resegmentation pipeline is >500MB and requires several
  seconds of global context before producing speaker labels. For a streaming, rolling-
  buffer architecture this latency budget is not acceptable for V1. See DECISIONS.md §AUE-3.
  Expected DER: ~25-35% with resemblyzer+gallery vs ~15-20% with full pyannote —
  this tradeoff is documented honestly rather than hidden.

Speaker labels: "SPK_01", "SPK_02", ... (anonymous, like "Person A" in AESE).
Real character names are bound later by SpeakerCharacterBinder.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Maximum clusters before switching to numeric overflow labels
_MAX_CLUSTERS = 99

# Distance threshold — same as AESE's DISTANCE_THRESHOLD_CLIP (0.45)
DISTANCE_THRESHOLD: float = 0.45

# Maximum exemplars stored per cluster — same as AESE's _MAX_EXEMPLARS_PER_CLUSTER (5)
MAX_EXEMPLARS_PER_CLUSTER: int = 5


def _label_for_index(idx: int) -> str:
    """Generate an anonymous speaker label."""
    return f"SPK_{idx + 1:02d}"


class SpeakerClusterer:
    """
    Online nearest-exemplar speaker clusterer.

    Directly mirrors AESE's CharacterClusterer (§21.3) for the same reason:
    exemplar galleries are immune to identity drift across alternating shots.

    Usage:
        clusterer = SpeakerClusterer()
        label = clusterer.assign(speaker_embedding)   # "SPK_01", "SPK_02", ...
        clusterer.reset()   # between video runs
    """

    def __init__(
        self,
        distance_threshold: float = DISTANCE_THRESHOLD,
        max_exemplars_per_cluster: int = MAX_EXEMPLARS_PER_CLUSTER,
    ) -> None:
        self.threshold = distance_threshold
        self.max_exemplars = max_exemplars_per_cluster
        # Per-cluster exemplar galleries — same structure as AESE's CharacterClusterer
        self.cluster_exemplars: List[List[np.ndarray]] = []
        self.cluster_labels: List[str] = []

    def assign(self, speaker_embedding: np.ndarray) -> str:
        """
        Assign a speaker embedding to an existing cluster or create a new one.

        Args:
            speaker_embedding: L2-normalized float32 embedding from speaker_embedding.py.

        Returns:
            str: Anonymous speaker label, e.g. "SPK_01".
        """
        if len(speaker_embedding) == 0:
            return _label_for_index(0)

        if not self.cluster_exemplars:
            return self._new_cluster(speaker_embedding)

        # Find best (minimum) distance across ALL exemplars in ALL clusters
        # — same loop structure as AESE CharacterClusterer.assign()
        best_dist = float("inf")
        best_idx = -1
        for i, gallery in enumerate(self.cluster_exemplars):
            for ex in gallery:
                d = float(np.linalg.norm(speaker_embedding - ex))
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_dist >= self.threshold:
            return self._new_cluster(speaker_embedding)

        # Accept match — update exemplar gallery
        gallery = self.cluster_exemplars[best_idx]
        if len(gallery) < self.max_exemplars:
            gallery.append(speaker_embedding.copy())

        return self.cluster_labels[best_idx]

    def _new_cluster(self, speaker_embedding: np.ndarray) -> str:
        """Create a new speaker cluster."""
        label = _label_for_index(len(self.cluster_labels))
        self.cluster_exemplars.append([speaker_embedding.copy()])
        self.cluster_labels.append(label)
        logger.debug(
            "AUE speaker_clusterer: new cluster %s (total=%d)",
            label, len(self.cluster_labels),
        )
        return label

    def reset(self) -> None:
        """Reset all cluster state (used between video runs / test isolation)."""
        self.cluster_exemplars = []
        self.cluster_labels = []

    def cluster_count(self) -> int:
        """Return the number of distinct speaker clusters found so far."""
        return len(self.cluster_labels)

    def merge_clusters(self, idx_a: int, idx_b: int) -> None:
        """
        Merge two clusters (batch consolidation phase).
        Cluster idx_b's exemplars are absorbed into idx_a.
        idx_b is removed. Called by consolidate_speakers() in pipeline.py.
        """
        if idx_a == idx_b:
            return
        if idx_a >= len(self.cluster_exemplars) or idx_b >= len(self.cluster_exemplars):
            return

        # Absorb exemplars from idx_b into idx_a (up to max_exemplars)
        for ex in self.cluster_exemplars[idx_b]:
            if len(self.cluster_exemplars[idx_a]) < self.max_exemplars:
                self.cluster_exemplars[idx_a].append(ex)

        merged_label = self.cluster_labels[idx_a]
        removed_label = self.cluster_labels[idx_b]

        self.cluster_exemplars.pop(idx_b)
        self.cluster_labels.pop(idx_b)

        logger.info(
            "AUE speaker_clusterer: merged cluster %s into %s (batch consolidation).",
            removed_label, merged_label,
        )

    def get_global_merge_candidates(
        self, global_threshold_factor: float = 0.85
    ) -> list:
        """
        Find pairs of clusters whose BEST cross-exemplar distance is below
        (threshold * global_threshold_factor). Used in batch consolidation
        after full-track processing to merge clusters that online processing
        kept separate due to limited early context.

        Returns list of (idx_a, idx_b) pairs to merge (smaller idx first).
        """
        candidates = []
        n = len(self.cluster_exemplars)
        merge_threshold = self.threshold * global_threshold_factor

        for i in range(n):
            for j in range(i + 1, n):
                best_dist = float("inf")
                for ex_i in self.cluster_exemplars[i]:
                    for ex_j in self.cluster_exemplars[j]:
                        d = float(np.linalg.norm(ex_i - ex_j))
                        if d < best_dist:
                            best_dist = d
                if best_dist < merge_threshold:
                    candidates.append((i, j, best_dist))

        # Sort by distance (closest pairs first)
        candidates.sort(key=lambda x: x[2])
        return [(i, j) for i, j, _ in candidates]
