"""
aese/adapters/gemma4.py
Singleton adapter for google/gemma-4-E2B-it (Gemma 4 2B instruction-tuned).

Provides the same public surface as fastvlm.py so either adapter can be
used as a drop-in via vlm_router.py:

  describe_scene(image)       -> str   (one of SCENE_LABELS or "unknown")
  count_people(image)         -> int   (NOT USED -- see note below)
  caption_event(image, ...)   -> str   (rich one-sentence event description)

NOTE on character counting:
  count_people() is provided for API symmetry but MUST NOT be called by
  the character-counting hot path. Character counting is handled exclusively
  by the deterministic OpenCV chain in character_stub.py (see DECISIONS.md
  Section 16). This function returns 0 and logs a warning if called.

Dependencies (add to requirements.txt):
  transformers>=4.52.0
  accelerate>=0.26.0
  torch

Usage (via vlm_router, not directly):
  python cli.py --input manifest.jsonl --output events.jsonl --vlm gemma4
"""
from __future__ import annotations

import logging
import re
import threading
from typing import List, Optional, Tuple

import numpy as np

from ._dtype_utils import _select_dtype  # shared fp16/fp32 selection logic

logger = logging.getLogger(__name__)

_MODEL_ID = "google/gemma-4-E2B-it"

# Fix 3 — maximum image dimension before feeding to the vision tower.
# Gemma-4's image encoder works well at 512px; larger inputs increase patch
# token count (and therefore prefill compute) with no meaningful quality gain
# for typical movie-frame content.  Tune if downstream quality degrades.
_MAX_DIM = 512

# Module-level singletons
_model = None
_processor = None
_gemma4_available: bool | None = None   # None = not yet attempted
_load_lock = threading.Lock()

# Fix 2 / §22.2 — batched generation
# Conservative starting batch size; tune upward with benchmark_speedup.py.
# OOM risk: each image + prompt pair occupies VRAM proportional to token count.
# At BATCH_SIZE=4 with 512px images and ~120 output tokens, peak VRAM usage
# is roughly 4x single-event usage — safe on 16GB+ GPUs, marginal on 8GB.
BATCH_SIZE: int = 4

# Fix 4 / §22.4 — torch.compile probe results (set by _ensure_loaded)
_compiled: bool = False        # True if torch.compile was applied successfully
_warmup_done: bool = False     # True if warmup() has completed

# _supports_batching() probe result (None = not yet tested)
_batching_supported: bool | None = None
_batching_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal: load model once
# ---------------------------------------------------------------------------

def _ensure_loaded() -> bool:
    """
    Attempt to load Gemma-4 processor + model.
    Returns True if successfully loaded, False if unavailable.
    Thread-safe via double-checked locking.
    """
    global _model, _processor, _gemma4_available

    if _gemma4_available is not None:
        return _gemma4_available

    with _load_lock:
        if _gemma4_available is not None:
            return _gemma4_available
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForMultimodalLM

            # §19 — explicit dtype via shared _select_dtype():
            #   float16 on CUDA (GPU fp16 is well-optimised; ~1.5-2x over fp32)
            #   float32 on CPU/MPS (CPU fp16 kernels are poorly optimised or absent)
            # Previously used bfloat16 (§18.4); changed to float16 per contract §19.
            # See DECISIONS.md §19 for the bfloat16 vs float16 tradeoff rationale.
            dtype = _select_dtype()

            if not torch.cuda.is_available():
                logger.warning(
                    "AESE Gemma-4: no CUDA GPU detected — running a 2B multimodal "
                    "model on CPU will be significantly slower than GPU regardless "
                    "of efficiency fixes applied here. Consider passing "
                    "'--vlm fastvlm' (apple/FastVLM-0.5B) as a CPU-friendly "
                    "alternative if real-time or near-real-time performance is needed."
                )

            # §18.4 — explicit attention implementation: SDPA is widely supported
            # (PyTorch ≥ 2.0) and meaningfully faster than eager attention.
            # Flash-Attention-2 is even faster but requires a separate install.
            # Fall back gracefully if the installed transformers version does not
            # accept attn_implementation (pre-4.36 builds).
            attn_impl = "sdpa"
            logger.info(
                "AESE Gemma-4: loading %s (dtype=%s, attn=%s, device_map=auto) ...",
                _MODEL_ID, dtype, attn_impl,
            )
            _processor = AutoProcessor.from_pretrained(_MODEL_ID)
            try:
                _model = AutoModelForMultimodalLM.from_pretrained(
                    _MODEL_ID,
                    torch_dtype=dtype,
                    device_map="auto",
                    attn_implementation=attn_impl,
                )
            except TypeError:
                # Older transformers versions don't accept attn_implementation.
                logger.warning(
                    "AESE Gemma-4: attn_implementation='sdpa' not accepted by "
                    "this transformers version — falling back to eager attention. "
                    "Upgrade to transformers>=4.36 for SDPA support."
                )
                _model = AutoModelForMultimodalLM.from_pretrained(
                    _MODEL_ID,
                    torch_dtype=dtype,
                    device_map="auto",
                )
            _model.eval()

            # Fix 4 / §22.4 — torch.compile for repeated same-shape inference.
            # mode="reduce-overhead" trades ~30-60s one-time compile cost for
            # 15-30% per-call throughput gain on repeated generate() calls.
            # Only applied on CUDA (CPU torch.compile has no meaningful benefit
            # for transformer inference and can be slower due to overhead).
            # warmup() must be called after load to amortize compile cost at
            # startup rather than on the first real event's timing window.
            global _compiled
            if torch.cuda.is_available():
                try:
                    _model = torch.compile(_model, mode="reduce-overhead")
                    _compiled = True
                    logger.info(
                        "AESE Gemma-4: torch.compile applied (mode=reduce-overhead). "
                        "Call warmup() to amortize compile cost before first real event."
                    )
                except Exception as compile_exc:
                    logger.warning(
                        "AESE Gemma-4: torch.compile failed (%s) — running uncompiled. "
                        "This is non-fatal; inference still works.",
                        compile_exc,
                    )
                    _compiled = False

            _gemma4_available = True
            logger.info("AESE Gemma-4: model loaded successfully (dtype=%s, compiled=%s).", dtype, _compiled)
        except ImportError as exc:
            logger.warning(
                "AESE Gemma-4: load failed (missing dependency: %s). "
                "Install transformers>=4.52, accelerate, and torch to enable.",
                exc,
            )
            _gemma4_available = False
        except Exception as exc:
            logger.warning(
                "AESE Gemma-4: load failed (%s). "
                "Check model ID, disk space, and GPU memory.",
                exc,
            )
            _gemma4_available = False

    return _gemma4_available


def get_active_detector_mode() -> str:
    """Return 'gemma4' if the model loaded, 'unavailable' otherwise."""
    if _gemma4_available is None:
        _ensure_loaded()
    return "gemma4" if _gemma4_available else "unavailable"


def warmup() -> None:
    """
    Fire one dummy generate() pass to trigger torch.compile's JIT compilation.

    Must be called AFTER _ensure_loaded() returns True and BEFORE the first
    real event's timing window starts. Calling it in a background thread during
    Phase 1 (CPU-bound) ensures compile finishes before Phase 2 (GPU-bound).

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _warmup_done
    if _warmup_done or not _ensure_loaded():
        return
    logger.info("AESE Gemma-4: running warmup pass (triggers torch.compile JIT)...")
    _ask(np.zeros((64, 64, 3), dtype=np.uint8), "warmup", max_new_tokens=5)
    _warmup_done = True
    logger.info("AESE Gemma-4: warmup complete.")



# ---------------------------------------------------------------------------
# Internal: run a prompt through the model
# ---------------------------------------------------------------------------

def _ask(
    image_rgb: Optional[np.ndarray],
    prompt: str,
    max_new_tokens: int = 256,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Run a text-only or multimodal prompt through Gemma-4.

    Args:
        image_rgb:      HxWx3 RGB numpy array, or None for text-only inference.
        prompt:         User-turn text prompt.
        max_new_tokens: Maximum tokens to generate.
        system_prompt:  Optional system instruction placed before the user turn.

    Returns:
        Decoded response string, or "" on any failure.
    """
    if not _ensure_loaded():
        return ""

    try:
        import PIL.Image as PILImage
        import torch

        # Build the message list
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})

        if image_rgb is not None:
            pil_img = _prepare_image(image_rgb)  # Fix 3: downscale before encoding
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text",  "text": prompt},
                    ],
                }
            )
        else:
            messages.append(
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            )

        # Use apply_chat_template to build tokenized inputs
        inputs = _processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(_model.device)

        input_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        raw = _processor.decode(
            output_ids[0][input_len:],
            skip_special_tokens=True,
        ).strip()

        # parse_response is available on some processors -- skip if not present
        if hasattr(_processor, "parse_response"):
            try:
                raw = _processor.parse_response(raw) or raw
            except Exception:
                pass  # parse_response is optional; raw string is fine

        # §19 — fp16 degenerate-output guard: fp16's reduced exponent range can
        # occasionally produce NaN/inf in logits, which decodes to empty or
        # single-character garbage.  Treat that identically to any other VLM
        # failure: return "" so the caller's existing template-summary fallback
        # engages.  No new failure mode; no special-casing needed elsewhere.
        if not raw or len(raw) < 2:
            logger.debug(
                "AESE Gemma-4: suspiciously short/empty output (len=%d) — "
                "possible fp16 degenerate decode; treating as failure.",
                len(raw),
            )
            return ""

        return raw

    except Exception as exc:
        logger.debug("AESE Gemma-4: inference error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Fix 3 — image preparation (downscale to <= _MAX_DIM before vision-tower encoding)
# ---------------------------------------------------------------------------

def _prepare_image(image_rgb: np.ndarray) -> "PILImage.Image":  # type: ignore[name-defined]
    """
    Convert an RGB numpy array to a PIL Image, downscaling so the longest
    dimension is at most _MAX_DIM pixels.

    Vision-language models internally tile/patch high-resolution images;
    feeding a 1080p frame directly increases the number of image tokens (and
    therefore prefill compute) without improving caption quality once the
    resolution exceeds the model's effective working resolution (~512px for
    Gemma-4-E2B).  Aspect ratio is preserved; LANCZOS filter minimises
    aliasing on edges.

    DECISIONS.md §18 — quality sign-off required before production:
        Run a manual comparison on 5 representative keyframes at original vs
        512px resolution and record results in DECISIONS.md §18 before shipping.
    """
    import PIL.Image as PILImage
    pil_img = PILImage.fromarray(image_rgb)
    if max(pil_img.size) > _MAX_DIM:
        scale = _MAX_DIM / max(pil_img.size)
        new_w = max(1, int(pil_img.width * scale))
        new_h = max(1, int(pil_img.height * scale))
        pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
    return pil_img


# ---------------------------------------------------------------------------
# Public API -- same surface as fastvlm.py
# ---------------------------------------------------------------------------

def describe_scene(image: np.ndarray) -> str:
    """
    Use Gemma-4 to classify the scene into one of the fixed SCENE_LABELS.

    Returns a label from SCENE_LABELS or "unknown" on failure.
    """
    from .scene_label import SCENE_LABELS
    label_list = ", ".join(f'"{l}"' for l in SCENE_LABELS if l != "unknown")
    prompt = (
        f"Look at this image and choose the single best matching scene label "
        f"from this list: {label_list}. "
        f"Reply with only the label, nothing else."
    )
    response = _ask(image, prompt, max_new_tokens=16)
    if not response:
        return "unknown"

    response_lower = response.lower().strip().strip('"').strip("'")
    for label in SCENE_LABELS:
        if label in response_lower or response_lower in label:
            return label

    logger.debug("AESE Gemma-4: scene label %r not in vocabulary -- using 'unknown'", response)
    return "unknown"


def count_people(image: np.ndarray) -> int:
    """
    API-symmetry stub. Character counting MUST use character_stub.py (DECISIONS.md §16).
    This function is NOT called by the pipeline and returns 0 if invoked directly.
    """
    logger.warning(
        "AESE Gemma-4: count_people() called -- this should not happen. "
        "Character counting must use the deterministic OpenCV chain in character_stub.py."
    )
    return 0


def caption_event(
    image: np.ndarray,
    scene_label: str,
    action_label: str,
    dialogue_text: Optional[str],
) -> str:
    """
    Use Gemma-4 to generate a rich one-sentence caption for an event.
    Returns "" on failure -- caller falls back to template summary.
    """
    ctx_parts = [f"scene: {scene_label}", f"action: {action_label}"]
    if dialogue_text:
        short_dialogue = dialogue_text[:80] + ("..." if len(dialogue_text) > 80 else "")
        ctx_parts.append(f'dialogue: "{short_dialogue}"')
    context = "; ".join(ctx_parts)

    prompt = (
        f"Describe what is happening in this movie scene in one concise sentence. "
        f"Context -- {context}. "
        f"Be specific about actions and setting. Reply with one sentence only."
    )
    return _ask(image, prompt, max_new_tokens=100)


# ---------------------------------------------------------------------------
# Fix 2 — merged scene-label + caption in a single forward pass
# ---------------------------------------------------------------------------

def _parse_scene_and_caption(
    raw: str,
    fallback_labels: List[str],
) -> Tuple[str, str]:
    """
    Parse a combined SCENE/CAPTION response produced by describe_scene_and_caption().

    Expected format (case-insensitive, both fields present):
        SCENE: <label>
        CAPTION: <one sentence>

    Returns:
        (scene_label, caption): both are strings; scene_label falls back to
        "unknown" if not in fallback_labels or missing; caption falls back to ""
        if missing.  Never raises.
    """
    scene_match = re.search(r"SCENE:\s*(.+)", raw, re.IGNORECASE)
    caption_match = re.search(r"CAPTION:\s*(.+)", raw, re.IGNORECASE)

    raw_scene = scene_match.group(1).strip().strip('"').strip("'") if scene_match else ""
    scene = raw_scene if raw_scene in fallback_labels else "unknown"

    caption = caption_match.group(1).strip() if caption_match else ""
    return scene, caption


def describe_scene_and_caption(
    image: np.ndarray,
    action_label: str,
    dialogue_text: Optional[str],
    scene_labels: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Single Gemma-4 generation call that produces BOTH the scene label AND the
    event caption, avoiding a second full image-encoding pass for the same frame.

    This halves the number of vision-tower forward passes per event compared to
    calling describe_scene() + caption_event() separately.

    Args:
        image:        HxWx3 RGB numpy array (the representative keyframe).
        action_label: Coarse action bucket ("static" / "walking" / "fast_action").
        dialogue_text: Optional subtitle text (truncated to 80 chars in the prompt).
        scene_labels: Label vocabulary to constrain the SCENE field.  Defaults to
                      SCENE_LABELS from scene_label.py.

    Returns:
        (scene_label, caption) — both strings.  scene_label is one of scene_labels
        or "unknown"; caption is the generated sentence or "" on failure.
        Never raises; falls back to ("unknown", "") on any error.
    """
    from .scene_label import SCENE_LABELS
    labels = scene_labels if scene_labels is not None else SCENE_LABELS
    label_list = ", ".join(f'"{l}"' for l in labels if l != "unknown")
    dialogue_part = (
        f' Dialogue spoken: "{dialogue_text[:80]}{"..." if len(dialogue_text) > 80 else ""}"'
        if dialogue_text
        else ""
    )
    prompt = (
        f"Look at this image and respond in EXACTLY this format (two lines, no extras):\n"
        f"SCENE: <one label from: {label_list}>\n"
        f"CAPTION: <one concise sentence describing the action and setting>\n"
        f"Context — action: {action_label}.{dialogue_part}"
    )
    raw = _ask(image, prompt, max_new_tokens=120)
    if not raw:
        return "unknown", ""
    return _parse_scene_and_caption(raw, fallback_labels=labels)

# ---------------------------------------------------------------------------
# Fix 2 / \u00a722.2 \u2014 Batched generation across multiple events
# ---------------------------------------------------------------------------


def _supports_batching() -> bool:
    """
    Probe whether the loaded processor + model support batched multimodal generation.

    Gemma-4 with Transformers >= 4.52 supports batched text, but batching multiple
    images in a single apply_chat_template call depends on the processor
    implementation.  Runs a 2-item dummy probe and caches the result.

    Returns True if batching is clean; False (sequential fallback) otherwise.
    """
    global _batching_supported
    if _batching_supported is not None:
        return _batching_supported
    with _batching_lock:
        if _batching_supported is not None:
            return _batching_supported
        if not _ensure_loaded():
            _batching_supported = False
            return False
        try:
            import torch
            import PIL.Image as PILImage
            import numpy as _np
            dummy = PILImage.fromarray(_np.zeros((64, 64, 3), dtype=_np.uint8))
            probe = [
                [{"role": "user", "content": [
                    {"type": "image", "image": dummy},
                    {"type": "text",  "text": "Say OK."},
                ]}],
                [{"role": "user", "content": [
                    {"type": "image", "image": dummy},
                    {"type": "text",  "text": "Say OK."},
                ]}],
            ]
            inputs = _processor.apply_chat_template(
                probe, tokenize=True, return_dict=True, return_tensors="pt",
                padding=True, add_generation_prompt=True, enable_thinking=False,
            ).to(_model.device)
            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=5, do_sample=False)
            if out.shape[0] != 2:
                raise ValueError(f"Expected batch dim 2, got {out.shape[0]}")
            _batching_supported = True
            logger.info("AESE Gemma-4: batched multimodal generation probe PASSED.")
        except Exception as exc:
            _batching_supported = False
            logger.warning(
                "AESE Gemma-4: batch probe FAILED (%s) -- using sequential.", exc
            )
    return _batching_supported


def describe_batch(
    images: List[Optional[np.ndarray]],
    contexts: List[dict],
) -> List[Tuple[str, str]]:
    """
    Process N event keyframes in ONE generate() call (batch dimension).

    Context dicts: {action_label, dialogue_text, scene_labels}.
    Falls back to sequential describe_scene_and_caption() on any failure.
    Never raises; returns ('unknown', '') for events without a keyframe or on error.
    """
    if not images:
        return []

    def _seq() -> List[Tuple[str, str]]:
        out = []
        for img, ctx in zip(images, contexts):
            if img is None:
                out.append(("unknown", ""))
                continue
            try:
                out.append(describe_scene_and_caption(
                    img,
                    action_label=ctx.get("action_label", "static"),
                    dialogue_text=ctx.get("dialogue_text"),
                    scene_labels=ctx.get("scene_labels"),
                ))
            except Exception:
                out.append(("unknown", ""))
        return out

    if not _ensure_loaded() or not _supports_batching():
        return _seq()

    try:
        import torch
        from .scene_label import SCENE_LABELS

        n = len(images)
        all_msgs: list = []
        fb_masks: list = []
        lbl_lists: list = []

        for img, ctx in zip(images, contexts):
            labels = ctx.get("scene_labels") or SCENE_LABELS
            lbl_lists.append(labels)
            if img is None:
                all_msgs.append(None)
                fb_masks.append(True)
                continue
            lbl_str = ", ".join('"' + l + '"' for l in labels if l != "unknown")
            dlg_text  = ctx.get("dialogue_text")
            act_label = ctx.get("action_label", "static")
            if dlg_text:
                dlg_part = (
                    ' Dialogue spoken: "'
                    + dlg_text[:80]
                    + ("..." if len(dlg_text) > 80 else "")
                    + '"'
                )
            else:
                dlg_part = ""
            prompt = (
                "Look at this image and respond in EXACTLY this format (two lines):\n"
                "SCENE: <one label from: " + lbl_str + ">\n"
                "CAPTION: <one concise sentence describing the action and setting>\n"
                "Context -- action: " + act_label + "." + dlg_part
            )
            all_msgs.append([{"role": "user", "content": [
                {"type": "image", "image": _prepare_image(img)},
                {"type": "text",  "text": prompt},
            ]}])
            fb_masks.append(False)

        batch_msgs = [m for m, fm in zip(all_msgs, fb_masks) if not fm]
        batch_idx  = [i for i, fm in enumerate(fb_masks) if not fm]
        b_labels   = [lbl_lists[i] for i in batch_idx]

        if not batch_msgs:
            return _seq()

        inputs = _processor.apply_chat_template(
            batch_msgs,
            tokenize=True, return_dict=True, return_tensors="pt",
            padding=True, add_generation_prompt=True, enable_thinking=False,
        ).to(_model.device)
        in_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            out_ids = _model.generate(**inputs, max_new_tokens=120, do_sample=False)

        batch_res: List[Tuple[str, str]] = []
        for i, (out, labels) in enumerate(zip(out_ids, b_labels)):
            raw = _processor.decode(out[in_len:], skip_special_tokens=True).strip()
            if not raw or len(raw) < 2:
                logger.debug(
                    "AESE Gemma-4: describe_batch empty output item %d "
                    "-- sequential fallback for this item.", i,
                )
                orig_ctx = contexts[batch_idx[i]]
                batch_res.append(describe_scene_and_caption(
                    images[batch_idx[i]],
                    action_label=orig_ctx.get("action_label", "static"),
                    dialogue_text=orig_ctx.get("dialogue_text"),
                    scene_labels=orig_ctx.get("scene_labels"),
                ))
            else:
                batch_res.append(_parse_scene_and_caption(raw, fallback_labels=labels))

        output: List[Tuple[str, str]] = [("unknown", "")] * n
        for orig_i, res in zip(batch_idx, batch_res):
            output[orig_i] = res

        logger.debug(
            "AESE Gemma-4: describe_batch: %d items (%d batched, %d fallback).",
            n, len(batch_idx), sum(fb_masks),
        )
        return output

    except Exception as exc:
        logger.warning(
            "AESE Gemma-4: describe_batch failed (%s) -- sequential fallback.", exc
        )
        return _seq()
