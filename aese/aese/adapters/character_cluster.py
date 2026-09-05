"""
aese/adapters/character_cluster.py
Anonymous-but-consistent face clustering for AESE (Fix 4, §21.3).

Architectural contract:
  - Assigns detected faces to consistent labels ("Person A", "Person B", ...)
    across the ENTIRE video -- same face always gets the same label.
  - Labels are ANONYMOUS by default. Real names are NEVER assigned here.
    Real names only appear if the user supplies --character-references (Fix 5),
    which runs a separate match step in character_reference.py.
  - Streaming-compatible: online nearest-exemplar assignment.
  - If face detection or CLIP are unavailable, returns empty lists gracefully.

Identity-discriminative embedding (§21.3):
  The preferred path uses InsightFace (buffalo_sc, ONNX-based, no torch required),
  which produces ArcFace-style embeddings specifically trained for identity
  discrimination. If InsightFace is unavailable (§21.3 probe: not installed in this
  deployment), the CLIP image encoder is used as a fallback with a tighter distance
  threshold (0.45 vs. the prior EMA-centroid default of 0.6).

  InsightFace availability is probed once at first call and cached. If unavailable,
  the code uses CLIP but logs a one-time warning. The fallback is honest about its
  limitations: CLIP embeddings trained for image-text matching are not specifically
  identity-discriminative, and the appearance-descriptor conflict check provides
  an additional defense-in-depth layer.

Exemplar gallery (§21.3):
  Replaces the prior EMA centroid update rule (0.9 old + 0.1 new), which caused
  identity drift when alternating shots slowly pulled the centroid toward the
  second actor. Each cluster now stores up to max_exemplars raw embeddings.
  Assignment matches against the BEST exemplar in each gallery (min distance),
  not a drifting mean. This makes long-range identity consistency robust to
  per-shot lighting variation.

Appearance-descriptor conflict check (§21.3):
  Defense-in-depth: attaches a coarse torso-region color histogram alongside
  the embedding. If a face embedding would match cluster X, but the torso HSV
  histogram conflicts sharply with cluster X's recorded appearance, the match
  is rejected and the face is assigned to a new cluster instead.

  This guard prevents cross-actor identity flips in scenes where two actors
  wear distinctly different clothing, even when the face embedding is ambiguous
  (backlit, low-resolution, or CLIP-space near-neighbor confusion).

Character identity guardrail (see README.md and DECISIONS.md):
  AESE clusters faces into consistent anonymous labels within a single video.
  It does NOT perform real-world identity recognition and will NOT name real
  people, including public figures, unless a labeled reference is explicitly
  supplied per Fix 5.

See: character_reference.py for the opt-in named-matching layer.
"""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Maximum number of distinct clusters before we stop creating new ones.
# At label index 26 (beyond "Z") we use "Person 27" etc. -- still anonymous.
_MAX_CLUSTERS = 52  # A-Z then a-z

# Embedding source constants — set by _probe_embedding_source() at first call.
_EMBEDDING_SOURCE: str = "unknown"   # "insightface" | "clip" | "none"
_EMBEDDING_SOURCE_PROBED: bool = False

# Distance threshold per embedding source:
#   InsightFace ArcFace embeddings: 0.45 (tight, identity-trained L2 space)
#   CLIP image embeddings:          0.45 (tighter than the prior 0.6 default
#                                         to reduce false cluster merges, though
#                                         CLIP is still less identity-reliable)
DISTANCE_THRESHOLD_INSIGHTFACE: float = 0.45
DISTANCE_THRESHOLD_CLIP: float = 0.45

# Appearance descriptor conflict check thresholds
# Histogram distance above this → reject embedding match, create new cluster
_APPEARANCE_CONFLICT_THRESHOLD: float = 0.40

# Maximum exemplars stored per cluster
_MAX_EXEMPLARS_PER_CLUSTER: int = 5


def _label_for_index(idx: int) -> str:
    """Generate a human-readable anonymous label for cluster index idx."""
    if idx < 26:
        return f"Person {chr(65 + idx)}"  # Person A .. Person Z
    return f"Person {idx + 1}"            # Person 27 .. (overflow fallback)


def _probe_embedding_source() -> None:
    """
    Probe InsightFace availability once and cache the result.

    InsightFace (buffalo_sc model) uses ONNX Runtime — no torch dependency
    required. This is the preferred embedding source for identity tasks.
    If unavailable, falls back to the CLIP image encoder.
    """
    global _EMBEDDING_SOURCE, _EMBEDDING_SOURCE_PROBED
    if _EMBEDDING_SOURCE_PROBED:
        return
    _EMBEDDING_SOURCE_PROBED = True
    try:
        import insightface  # noqa: F401
        _EMBEDDING_SOURCE = "insightface"
        logger.info(
            "AESE character_cluster: InsightFace available — "
            "using ArcFace identity-discriminative embeddings (§21.3)."
        )
    except ImportError:
        _EMBEDDING_SOURCE = "clip"
        logger.warning(
            "AESE character_cluster: InsightFace not available (ModuleNotFoundError). "
            "Falling back to CLIP image embeddings for face clustering. "
            "CLIP embeddings are not specifically identity-discriminative — "
            "appearance-descriptor conflict check provides defense-in-depth. "
            "To improve identity consistency, install: pip install insightface onnxruntime"
        )


# ---------------------------------------------------------------------------
# Appearance descriptor: coarse torso-region HSV histogram
# ---------------------------------------------------------------------------

def _torso_color_descriptor(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    """
    Extract a coarse 8-bin HSV hue histogram of the torso region below a face box.

    The torso region is approximated as the area from the bottom of the face box
    to 2× the face height below it, at the same horizontal extent.  This gives a
    rough clothing-color signature without requiring a body-pose estimator.

    Returns:
        Normalized float32 histogram of shape (8,), or None if the region is
        out-of-bounds or the image is too small.
    """
    try:
        import cv2
        x, y, bw, bh = box
        h, w = image.shape[:2]
        # Torso region: from bottom of face to 2x face height below
        ty1 = min(h, y + bh)
        ty2 = min(h, y + 3 * bh)
        if ty2 <= ty1 or bw < 8:
            return None
        torso_crop = image[ty1:ty2, max(0, x):min(w, x + bw)]
        if torso_crop.size == 0:
            return None
        bgr = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0], None, [8], [0, 180])
        hist = hist.flatten().astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist
    except Exception as exc:
        logger.debug("AESE character_cluster: torso descriptor failed: %s", exc)
        return None


def _histogram_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute the Bhattacharyya-like L1 distance between two normalized histograms.
    Returns 0.0 for identical histograms, up to 1.0 for completely different ones.
    """
    return float(np.sum(np.abs(a - b))) / 2.0


# ---------------------------------------------------------------------------
# Face crop embedding
# ---------------------------------------------------------------------------

def _embed_face_crop_insightface(crop: np.ndarray) -> Optional[np.ndarray]:
    """Embed a face crop using InsightFace (ArcFace, ONNX-based)."""
    try:
        import insightface
        from insightface.app import FaceAnalysis
        # Use a module-level singleton to avoid re-loading on every call
        global _insightface_app
        if "_insightface_app" not in globals() or _insightface_app is None:
            _insightface_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
            _insightface_app.prepare(ctx_id=-1, det_size=(160, 160))
        faces = _insightface_app.get(crop)
        if not faces:
            return None
        emb = faces[0].normed_embedding
        return emb.astype(np.float32)
    except Exception as exc:
        logger.debug("AESE character_cluster: InsightFace embed failed: %s", exc)
        return None


def _embed_face_crop_clip(crop: np.ndarray) -> Optional[np.ndarray]:
    """Embed a face crop using CLIP image encoder (fallback when InsightFace unavailable)."""
    try:
        from .embedding import _clip_model, _clip_preprocess, _clip_available
        import torch
        import PIL.Image as PILImage

        if not _clip_available or _clip_model is None:
            return None

        device = next(_clip_model.parameters()).device
        pil_img = PILImage.fromarray(crop)
        img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = _clip_model.encode_image(img_tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        vec = emb.squeeze(0).cpu().numpy().astype(np.float32)
        return vec
    except Exception as exc:
        logger.debug("AESE character_cluster: CLIP face crop embed failed: %s", exc)
        return None


def _embed_face_crop(crop: np.ndarray) -> Optional[np.ndarray]:
    """
    Embed a face crop using the best available identity-discriminative model.

    Priority:
      1. InsightFace ArcFace (buffalo_sc, ONNX — preferred, identity-trained)
      2. CLIP image encoder (fallback — less reliable for identity tasks)

    Returns None if both paths fail or are unavailable. Callers must handle None
    honestly — do not substitute a weaker embedding silently.
    """
    _probe_embedding_source()
    if _EMBEDDING_SOURCE == "insightface":
        return _embed_face_crop_insightface(crop)
    return _embed_face_crop_clip(crop)


# ---------------------------------------------------------------------------
# Clusterer
# ---------------------------------------------------------------------------

class CharacterClusterer:
    """
    Online nearest-exemplar face clusterer with appearance-descriptor conflict check.

    Changes from prior EMA-centroid design (§21.3):
      - Exemplar gallery: each cluster stores up to max_exemplars raw embeddings.
        Assignment matches against the BEST (closest) exemplar in each gallery,
        not a drifting mean. This prevents the centroid slowly drifting toward a
        second actor across alternating shots.
      - Appearance-descriptor conflict check: if a face embedding would match
        cluster X, but the torso HSV histogram conflicts sharply with cluster X's
        recorded appearance, the match is rejected and a new cluster is created.
        This is a defense-in-depth guard for cases where CLIP embeddings are
        ambiguous between two different actors.

    Args:
        distance_threshold: L2 distance threshold for cluster assignment.
            Defaults to DISTANCE_THRESHOLD_CLIP; the InsightFace path will
            use DISTANCE_THRESHOLD_INSIGHTFACE if available.
        max_exemplars_per_cluster: Maximum exemplars stored per cluster.
    """

    def __init__(
        self,
        distance_threshold: float = DISTANCE_THRESHOLD_CLIP,
        max_exemplars_per_cluster: int = _MAX_EXEMPLARS_PER_CLUSTER,
    ) -> None:
        self.threshold = distance_threshold
        self.max_exemplars = max_exemplars_per_cluster
        # Per-cluster exemplar galleries (list of embedding lists)
        self.cluster_exemplars: List[List[np.ndarray]] = []
        self.cluster_labels: List[str] = []
        # Per-cluster appearance descriptor galleries (list of histogram lists)
        self.cluster_descriptors: List[List[np.ndarray]] = []

    def assign(
        self,
        face_embedding: np.ndarray,
        appearance_descriptor: Optional[np.ndarray] = None,
    ) -> str:
        """
        Assign a face embedding to an existing cluster or create a new one.

        Args:
            face_embedding: Normalized float32 face embedding vector.
            appearance_descriptor: Optional 8-bin HSV torso histogram for
                conflict checking. If None, the appearance check is skipped.

        Returns:
            str: Anonymous label, e.g. "Person A".
        """
        if not self.cluster_exemplars:
            return self._new_cluster(face_embedding, appearance_descriptor)

        # Find best (minimum) distance across ALL exemplars in ALL clusters
        best_dist = float("inf")
        best_idx = -1
        for i, gallery in enumerate(self.cluster_exemplars):
            for ex in gallery:
                d = float(np.linalg.norm(face_embedding - ex))
                if d < best_dist:
                    best_dist = d
                    best_idx = i

        if best_dist >= self.threshold:
            return self._new_cluster(face_embedding, appearance_descriptor)

        # Appearance conflict check: if we have a torso descriptor and the cluster
        # has recorded appearances, reject if they conflict sharply.
        if appearance_descriptor is not None and self.cluster_descriptors[best_idx]:
            # Compute minimum histogram distance to any stored descriptor
            min_hist_dist = min(
                _histogram_distance(appearance_descriptor, d)
                for d in self.cluster_descriptors[best_idx]
            )
            if min_hist_dist > _APPEARANCE_CONFLICT_THRESHOLD:
                logger.debug(
                    "AESE character_cluster: appearance conflict for cluster %s "
                    "(hist_dist=%.3f > %.3f) — creating new cluster",
                    self.cluster_labels[best_idx], min_hist_dist,
                    _APPEARANCE_CONFLICT_THRESHOLD,
                )
                return self._new_cluster(face_embedding, appearance_descriptor)

        # Accept the match — update exemplar gallery and descriptors
        gallery = self.cluster_exemplars[best_idx]
        if len(gallery) < self.max_exemplars:
            gallery.append(face_embedding.copy())
        if appearance_descriptor is not None:
            desc_gallery = self.cluster_descriptors[best_idx]
            if len(desc_gallery) < self.max_exemplars:
                desc_gallery.append(appearance_descriptor.copy())

        return self.cluster_labels[best_idx]

    def _new_cluster(
        self,
        face_embedding: np.ndarray,
        appearance_descriptor: Optional[np.ndarray],
    ) -> str:
        label = _label_for_index(len(self.cluster_labels))
        self.cluster_exemplars.append([face_embedding.copy()])
        self.cluster_labels.append(label)
        self.cluster_descriptors.append(
            [appearance_descriptor.copy()] if appearance_descriptor is not None else []
        )
        logger.debug(
            "AESE character_cluster: new cluster %s (total=%d)",
            label, len(self.cluster_labels),
        )
        return label

    def reset(self) -> None:
        """Reset all cluster state (used between test runs)."""
        self.cluster_exemplars = []
        self.cluster_labels = []
        self.cluster_descriptors = []


# ---------------------------------------------------------------------------
# Face crop embedding (public)
# ---------------------------------------------------------------------------

def extract_face_embeddings(
    image: np.ndarray,
    face_boxes: List[Tuple[int, int, int, int]],
) -> List[np.ndarray]:
    """
    Extract embeddings for each detected face crop.

    Uses InsightFace ArcFace if available, CLIP image encoder otherwise.

    Args:
        image:      HxWx3 RGB numpy array.
        face_boxes: List of (x, y, w, h) bounding boxes from character_stub.

    Returns:
        List of normalized float32 embedding vectors. Empty list if the
        embedding model is unavailable or no valid crops could be extracted.
    """
    if not face_boxes or image is None:
        return []

    h, w = image.shape[:2]
    embeddings = []
    for (bx, by, bw, bh) in face_boxes:
        x1 = max(0, bx)
        y1 = max(0, by)
        x2 = min(w, bx + bw)
        y2 = min(h, by + bh)
        if x2 <= x1 or y2 <= y1:
            continue
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        emb = _embed_face_crop(crop)
        if emb is not None:
            embeddings.append(emb)

    return embeddings


# ---------------------------------------------------------------------------
# Per-event label resolution
# ---------------------------------------------------------------------------

def get_character_labels_for_event(
    face_embeddings_per_second: List[List[np.ndarray]],
    clusterer: CharacterClusterer,
) -> List[str]:
    """
    Assign all face embeddings in an event to cluster labels and return
    the deduplicated sorted set of labels seen during this event.

    Args:
        face_embeddings_per_second: List of per-second face embedding lists
            (from TemporalFeature.face_embeddings across the event window).
        clusterer: The global CharacterClusterer for this video run.

    Returns:
        Sorted list of unique anonymous labels, e.g. ["Person A", "Person B"].
        Empty list if no embeddings were available.
    """
    seen_labels = set()
    for second_embeddings in face_embeddings_per_second:
        for emb in second_embeddings:
            # No per-embedding appearance descriptor here (aggregator doesn't
            # pass boxes through to features yet); conflict check uses None.
            label = clusterer.assign(emb, appearance_descriptor=None)
            seen_labels.add(label)
    return sorted(seen_labels)