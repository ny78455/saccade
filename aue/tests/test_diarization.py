"""
tests/test_diarization.py
Acceptance tests for diarization/speaker_embedding.py + speaker_clusterer.py.

Key test (mirrors AESE §21.3 identity-flip regression):
  Two distinct voice-like signals alternated across many segments must
  produce two separate speaker clusters, not one merged cluster.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


def _make_voice_a(n_samples: int = 16000) -> np.ndarray:
    """Synthetic 'voice A' — low-frequency tone with noise."""
    t = np.linspace(0, 1.0, n_samples)
    rng = np.random.default_rng(1)
    return (np.sin(2 * np.pi * 120 * t) * 0.3 + rng.standard_normal(n_samples) * 0.05).astype(np.float32)


def _make_voice_b(n_samples: int = 16000) -> np.ndarray:
    """Synthetic 'voice B' — high-frequency tone with noise (distinct from A)."""
    t = np.linspace(0, 1.0, n_samples)
    rng = np.random.default_rng(2)
    return (np.sin(2 * np.pi * 280 * t) * 0.3 + rng.standard_normal(n_samples) * 0.05).astype(np.float32)


class TestSpeakerClusterer:
    def setup_method(self):
        from aue.diarization.speaker_clusterer import SpeakerClusterer
        self.clusterer = SpeakerClusterer(distance_threshold=0.45)

    def _make_orthogonal_embedding(self, seed: int, dim: int = 40) -> np.ndarray:
        """Generate a random unit vector — seeds produce clearly distinct embeddings."""
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_single_speaker_one_cluster(self):
        """Same embedding repeated → one cluster."""
        emb = self._make_orthogonal_embedding(seed=1)
        labels = set()
        for _ in range(5):
            labels.add(self.clusterer.assign(emb + np.random.default_rng(99).standard_normal(len(emb)) * 0.01))
        assert len(labels) == 1, f"Expected 1 cluster, got {len(labels)}: {labels}"

    def test_two_distinct_speakers_two_clusters(self):
        """
        Two clearly orthogonal embeddings → two clusters.
        This is the identity-flip regression test (mirrors AESE §21.3).
        """
        emb_a = self._make_orthogonal_embedding(seed=1)
        emb_b = self._make_orthogonal_embedding(seed=99)  # far from seed=1

        labels_a = set()
        labels_b = set()
        for _ in range(5):
            labels_a.add(self.clusterer.assign(emb_a.copy()))
            labels_b.add(self.clusterer.assign(emb_b.copy()))

        assert labels_a == labels_b == {labels_a.pop()}, False  # not this
        # They should be different labels
        label_a = self.clusterer.assign(emb_a.copy())
        label_b = self.clusterer.assign(emb_b.copy())
        assert label_a != label_b, (
            f"Both speakers got label {label_a!r} — identity flip! "
            "Exemplar gallery should prevent this."
        )

    def test_two_speakers_alternated_no_drift(self):
        """
        Alternating A and B many times → 2 clusters (not 1 merged cluster).
        This is the exact drift failure mode that EMA centroid had.
        """
        emb_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0] * 8, dtype=np.float32)  # orthogonal
        emb_b = np.array([0.0, 0.0, 0.0, 0.0, 1.0] * 8, dtype=np.float32)  # orthogonal
        # Normalize
        emb_a = emb_a / np.linalg.norm(emb_a)
        emb_b = emb_b / np.linalg.norm(emb_b)

        for _ in range(20):
            self.clusterer.assign(emb_a.copy())
            self.clusterer.assign(emb_b.copy())

        assert self.clusterer.cluster_count() == 2, (
            f"Expected 2 clusters after 40 alternating assignments, "
            f"got {self.clusterer.cluster_count()}. Centroid drift detected!"
        )

    def test_reset_clears_all_state(self):
        """reset() clears all clusters."""
        emb = self._make_orthogonal_embedding(seed=1)
        self.clusterer.assign(emb)
        assert self.clusterer.cluster_count() == 1
        self.clusterer.reset()
        assert self.clusterer.cluster_count() == 0

    def test_label_format(self):
        """Labels follow SPK_XX format."""
        emb = self._make_orthogonal_embedding(seed=1)
        label = self.clusterer.assign(emb)
        assert label.startswith("SPK_"), f"Label {label!r} doesn't start with SPK_"

    def test_three_distinct_speakers_three_clusters(self):
        """Three distinct embeddings → 3 clusters."""
        e1 = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        e2 = np.array([0, 1, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        e3 = np.array([0, 0, 1, 0, 0, 0, 0, 0], dtype=np.float32)
        self.clusterer.assign(e1)
        self.clusterer.assign(e2)
        self.clusterer.assign(e3)
        assert self.clusterer.cluster_count() == 3


class TestSpeakerEmbedding:
    def test_embed_speaker_returns_none_or_array(self):
        """embed_speaker returns None or a float32 array."""
        from aue.diarization.speaker_embedding import embed_speaker
        y = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
        result = embed_speaker(y, sr=16000)
        assert result is None or (isinstance(result, np.ndarray) and result.dtype == np.float32)

    def test_embed_speaker_empty_returns_none(self):
        """Empty waveform → None."""
        from aue.diarization.speaker_embedding import embed_speaker
        result = embed_speaker(np.array([], dtype=np.float32), sr=16000)
        assert result is None

    def test_embed_speaker_l2_normalized(self):
        """If embedding is returned, it should be approximately L2-normalized."""
        from aue.diarization.speaker_embedding import embed_speaker
        rng = np.random.default_rng(42)
        y = rng.standard_normal(32000).astype(np.float32)
        result = embed_speaker(y, sr=16000)
        if result is not None:
            norm = np.linalg.norm(result)
            assert abs(norm - 1.0) < 0.1, f"Embedding not unit-normalized: norm={norm:.3f}"
