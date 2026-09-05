"""
aese/adapters/scene_label.py
Scene labeling adapter.

V3: Primary path routes through vlm_router so the active backend
(fastvlm / gemma4 / yunet) is respected without hard-coding fastvlm here.

Fallback chain (applied in order when image is unavailable or model fails):
  1. Active VLM backend via vlm_router (fastvlm / gemma4)
  2. CLIP zero-shot against the same label set -- if open_clip is available
  3. Color-temperature heuristic -- last resort

Falls back to "unknown" if all methods fail or if the image is None/black.
See DECISIONS.md S3.
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Fixed label set -- must stay in sync with fastvlm.py._SCENE_LABELS
# Exported as SCENE_LABELS (public) for tests and CLI banner.
# See Fix 3 for expanded vocabulary.
SCENE_LABELS = [
    "kitchen", "living room", "bedroom", "office", "hallway",
    "street", "village", "forest", "beach", "outdoor field",
    "vehicle interior", "rooftop", "restaurant", "stage/studio",
    "unknown",
]
# Backward-compatible private alias used internally
_SCENE_LABELS = SCENE_LABELS

# CLIP text features cached at first call (fallback path)
_clip_text_features = None
_clip_labels_loaded = False


# ---------------------------------------------------------------------------
# Fix 3 — Deterministic graphics/end-card pre-check (DECISIONS.md §20.3)
# ---------------------------------------------------------------------------

def is_graphics_or_endcard(image: np.ndarray) -> bool:
    """
    Detect flat graphics/logo/end-card frames BEFORE any VLM or CLIP call.

    Two-path check (DECISIONS.md §21.4) — covers both classes of end-card:

    Path 1 — Flat near-monochrome card (grey title card, watermark on white):
      color_std < 18.0    — near-solid background, very low variance
      edge_density < 0.015 — minimal fine structure

    Path 2 — Dark background with logo (Netflix-style: colored letter on black):
      dark_fraction > 0.70 — >70% of pixels are near-black (gray < 15)
      color_std < 30.0     — low overall variance despite the colored logo
                             (the logo occupies only a small fraction of the frame)

    Why Path 2 is needed: a red 'N' on a black background has dark_fraction ≈ 0.85
    but color_std ≈ 25 (the red pixels are a small fraction of the total).  A real
    dark dramatic scene retains color variation from actors, props, and practical
    light sources, giving color_std > 30 in practice.

    Pure black frames (image.max() < 5): dark_fraction=1.0, color_std=0.0 → Path 2
    fires. These return "graphics/end card" (a fade-to-black is not a real scene
    location). The previous "unknown" return for black frames is replaced by
    "graphics/end card" — see DECISIONS.md §21.4 for rationale.

    REQUIRED before production: validate Path 2 thresholds against real night scenes
    and silhouette shots where dark_fraction may be > 0.70 but color_std should
    stay above 30 due to actor outlines and practical light sources.

    Args:
        image: HxWx3 RGB numpy array. Caller guarantees image is not None.

    Returns:
        True if the frame looks like a graphics/logo/end-card; False otherwise.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        color_std = float(np.std(image))

        # Path 1: flat near-monochrome card
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / edges.size
        if color_std < 18.0 and edge_density < 0.015:
            return True

        # Path 2: dark background with small colored logo (or pure black)
        dark_fraction = float(np.mean(gray < 15))
        if dark_fraction > 0.70 and color_std < 30.0:
            return True

        return False
    except Exception as exc:
        logger.debug("AESE scene_label: is_graphics_or_endcard() failed: %s", exc)
        return False  # fail-safe: don't misclassify on error


def _clip_available() -> bool:
    """
    Return True if the CLIP model is loaded and available for scene classification.

    Does NOT trigger a load attempt -- this is a safe read of the current state.
    Used by cli.py for the SCENE CLASSIFICATION MODE startup banner.
    """
    try:
        from .embedding import _clip_available as _emb_clip_available
        return bool(_emb_clip_available)
    except Exception:
        return False


def _load_clip_text_features() -> bool:
    """Cache CLIP text encodings for all scene labels. Returns True on success."""
    global _clip_text_features, _clip_labels_loaded
    if _clip_labels_loaded:
        return _clip_text_features is not None

    try:
        from .embedding import _clip_model, _clip_tokenizer, _clip_available
        import torch

        if not _clip_available or _clip_model is None:
            _clip_labels_loaded = True
            return False

        device = next(_clip_model.parameters()).device
        prompts = [f"a photo of {lbl}" for lbl in _SCENE_LABELS]
        tokens = _clip_tokenizer(prompts).to(device)
        with torch.no_grad():
            feats = _clip_model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        _clip_text_features = feats.cpu().numpy().astype(np.float32)
        _clip_labels_loaded = True
        logger.debug("AESE scene_label: cached CLIP text features for %d labels", len(_SCENE_LABELS))
        return True
    except Exception as exc:
        logger.debug("AESE scene_label: CLIP text feature cache failed: %s", exc)
        _clip_labels_loaded = True
        return False


def label_scene(image: np.ndarray) -> str:
    """
    Classify the scene label of a single frame.

    Pre-check:    Deterministic graphics/end-card detection (no model call).
    Primary path: Active VLM backend via vlm_router (fastvlm / gemma4).
    Fallback 1:   CLIP zero-shot (if open_clip is available).
    Fallback 2:   Color-temperature heuristic (last resort).

    Returns:
        str: One of the labels in SCENE_LABELS, or "graphics/end card" for
             flat title/logo/end-card frames detected by the pre-check.
             ALWAYS returns a str, never None or raises.
             Returns "unknown" on any failure.

    Note:
        "graphics/end card" is NOT in SCENE_LABELS — it is returned by the
        pre-check only, never by the VLM or CLIP paths.

        Pure black frames (image.max() < 5) also return "graphics/end card"
        via the pre-check Path 2 (dark_fraction=1.0 > 0.70, color_std=0 < 30).
        This replaces the previous "unknown" return for black frames — a
        fade-to-black is not a real scene location (DECISIONS.md §21.4).
    """
    if image is None:
        return "unknown"

    # --- Pre-check: deterministic graphics/end-card detection (§21.4) ---
    # Path 1 catches flat near-monochrome cards. Path 2 catches dark-bg
    # colored logos and pure black frames. Both short-circuit before any
    # VLM or CLIP call.
    if is_graphics_or_endcard(image):
        logger.debug("AESE scene_label: graphics/end-card pre-check fired -- skipping VLM/CLIP")
        return "graphics/end card"

    # --- Path 1: Active VLM backend (fastvlm / gemma4 / yunet via router) ---
    try:
        from .vlm_router import describe_scene as _vlm_describe_scene, vlm_available
        if vlm_available():
            result = _vlm_describe_scene(image)
            if result and result != "unknown":
                return result
            # VLM returned "unknown" -- trust it and skip CLIP
            return "unknown"
    except Exception as exc:
        logger.debug("AESE scene_label: VLM path failed: %s -- trying CLIP", exc)

    # --- Path 2: CLIP zero-shot ---
    try:
        from .embedding import _clip_model, _clip_preprocess, _clip_available
        import torch

        if _clip_available and _clip_model is not None and _load_clip_text_features():
            device = next(_clip_model.parameters()).device
            import PIL.Image as PILImage
            pil_img = PILImage.fromarray(image)
            img_tensor = _clip_preprocess(pil_img).unsqueeze(0).to(device)

            with torch.no_grad():
                img_feat = _clip_model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                img_vec = img_feat.squeeze(0).cpu().numpy().astype(np.float32)

            sims = _clip_text_features @ img_vec
            best_idx = int(np.argmax(sims))
            return _SCENE_LABELS[best_idx]

    except Exception as exc:
        logger.debug("AESE scene_label CLIP inference failed: %s -- using heuristic", exc)

    # --- Path 3: heuristic ---
    return _heuristic_scene_label(image)


def _heuristic_scene_label(image: np.ndarray) -> str:
    """
    Bare-minimum heuristic for scene label when VLM and CLIP are unavailable.
    Uses color temperature (blue channel ratio for sky/outdoor, warm tones for indoor).
    Not reliable -- only a last-resort fallback. Always returns a label in SCENE_LABELS.
    """
    try:
        h, w = image.shape[:2]
        top = image[:h // 3, :, :]
        mean_r = float(top[:, :, 0].mean())
        mean_b = float(top[:, :, 2].mean())
        if mean_b > mean_r + 15 and mean_b > 80:
            return "outdoor field"  # blue top third => sky visible
        brightness = float(image.mean())
        if brightness < 30:
            return "unknown"        # too dark to classify safely
        # Generic warm/neutral interior -- use "office" as the nearest non-specific label
        return "office"
    except Exception:
        return "unknown"
