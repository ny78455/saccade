"""
aese/adapters/fastvlm.py
Singleton wrapper for the riddhimanrana/fastvlm-0.5b-captions VLM.

Provides three helpers that replace the static stub methods used by
scene_label.py, character_stub.py, and event_constructor.py:

  describe_scene(image)       -> str   (one of _SCENE_LABELS or "unknown")
  count_people(image)         -> int   (≥0, same contract as count_characters)
  caption_event(image, ...)   -> str   (rich one-sentence event description)

All three:
  - Return the original stub fallback value if the model is not installed,
    not loadable, or the image is None / black (image_available=False guard
    is applied by callers).
  - Load the model ONCE on first call (lazy, thread-safe via a module lock).
  - Use `_select_dtype()` from `_dtype_utils` (float16 on CUDA, float32 on CPU/MPS).

Dependencies (added to requirements.txt):
  transformers>=4.52.0   (FastVlmForConditionalGeneration first appeared here)
  accelerate>=0.26.0     (needed by device_map="auto")
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Optional

import numpy as np

from ._dtype_utils import _select_dtype  # shared fp16/fp32 selection logic

logger = logging.getLogger(__name__)

_MODEL_ID = "apple/FastVLM-0.5B"

# Module-level singletons — set on first successful load
_model = None
_tok = None
_fastvlm_available: bool | None = None   # None = not yet attempted
_load_lock = threading.Lock()

# Scene label vocabulary — must stay in sync with scene_label.py
_SCENE_LABELS = [
    "indoor", "outdoor", "vehicle interior", "street", "nature",
    "building exterior", "office", "restaurant", "kitchen", "bedroom",
    "nighttime", "unknown",
]


# ---------------------------------------------------------------------------
# Internal: load model once
# ---------------------------------------------------------------------------

def _ensure_loaded() -> bool:
    """
    Attempt to load FastVLM model + processor.
    Returns True if successfully loaded, False if unavailable.
    Thread-safe — safe to call from multiple aggregator calls concurrently.
    """
    global _model, _tok, _fastvlm_available

    if _fastvlm_available is not None:
        return _fastvlm_available

    with _load_lock:
        if _fastvlm_available is not None:   # double-checked locking
            return _fastvlm_available
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            # §19 — use shared _select_dtype(): float16 on CUDA, float32 on CPU/MPS.
            # Previously inlined here; extracted to _dtype_utils for DRY sharing with gemma4.
            dtype = _select_dtype()
            logger.info(
                "AESE FastVLM: loading %s (dtype=%s, device_map=auto) ...",
                _MODEL_ID, dtype,
            )
            _tok = AutoTokenizer.from_pretrained(
                _MODEL_ID, trust_remote_code=True
            )
            _model = AutoModelForCausalLM.from_pretrained(
                _MODEL_ID,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )
            _model.eval()
            _fastvlm_available = True
            logger.info("AESE FastVLM: model loaded successfully.")
        except Exception as exc:
            logger.warning(
                "AESE FastVLM: load failed (%s) — "
                "scene_label, character_count, and event summary will use "
                "static fallbacks. Install transformers>=4.52 and accelerate "
                "to enable VLM-powered event generation.",
                exc,
            )
            _fastvlm_available = False

    return _fastvlm_available


def get_active_detector_mode() -> str:
    """
    Return the active detector mode as a short string.

    Returns 'fastvlm' if the model loaded successfully, 'unavailable' otherwise.
    Always call _ensure_loaded() first (or call count_people() once) to trigger
    the load attempt before reading this value.
    """
    if _fastvlm_available is None:
        # Load not yet attempted — trigger it so the result is meaningful
        _ensure_loaded()
    return "fastvlm" if _fastvlm_available else "unavailable"


def _ask(image_rgb: np.ndarray, prompt: str, max_new_tokens: int = 60,
         system_prompt: Optional[str] = None) -> str:
    """
    Run a single image + text prompt through FastVLM.
    Returns the model's response string, or "" on any failure.

    FastVLM has no system-role support; system_prompt is prepended
    to the user turn when provided.

    Output extraction note:
        LLaVA-style models expand the single -200 image-placeholder token into
        N visual patch tokens during generate(). This makes the returned
        output_ids sequence longer than input_ids.shape[1] by (N-1) tokens,
        so slicing at input_len cuts mid-sentence. Instead we decode the full
        output with special tokens preserved, then split on the Qwen assistant
        turn marker to isolate exactly the generated reply.
    """
    if not _ensure_loaded():
        return ""

    try:
        import PIL.Image as PILImage
        import torch

        pil_img = PILImage.fromarray(image_rgb)

        # Prepend system instruction to the user prompt when supplied
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt

        # Build chat -> render to string (not tokens) so we can place <image> exactly
        messages = [
            {"role": "user", "content": f"<image>\n{full_prompt}"}
        ]
        rendered = _tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        pre, post = rendered.split("<image>", 1)

        # Tokenize the text *around* the image token (no extra specials!)
        pre_ids  = _tok(pre,  return_tensors="pt", add_special_tokens=False).input_ids
        post_ids = _tok(post, return_tensors="pt", add_special_tokens=False).input_ids

        IMAGE_TOKEN_INDEX = -200
        # Splice in the IMAGE token id (-200) at the placeholder position
        img_tok = torch.tensor([[IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
        input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(_model.device)
        attention_mask = torch.ones_like(input_ids, device=_model.device)

        # Preprocess image via the model's own processor
        px = _model.get_vision_tower().image_processor(images=pil_img, return_tensors="pt")["pixel_values"]
        px = px.to(_model.device, dtype=_model.dtype)

        # Get the EOS token id for the Qwen <|im_end|> end-of-turn marker so
        # generation stops cleanly rather than emitting textual stop-tags.
        try:
            eot_id = _tok.convert_tokens_to_ids("<|im_end|>")
        except Exception:
            eot_id = _tok.eos_token_id

        with torch.no_grad():
            output_ids = _model.generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                images=px,
                max_new_tokens=max_new_tokens,
                do_sample=False,   # greedy -- deterministic, fast
                eos_token_id=eot_id,
            )

        # --- Extract assistant reply via turn-marker split ---
        # Decoding the full sequence (with special tokens) lets us find the
        # Qwen assistant turn marker even when the image expands the sequence.
        full_output = _tok.decode(output_ids[0], skip_special_tokens=False)
        logger.debug("AESE FastVLM: raw full output: %r", full_output[-300:])

        # FastVLM uses the Qwen chat template: ...<|im_start|>assistant\n{reply}<|im_end|>
        ASSISTANT_MARKER = "<|im_start|>assistant"
        if ASSISTANT_MARKER in full_output:
            assistant_text = full_output.split(ASSISTANT_MARKER)[-1]
            # Strip trailing end-of-turn markers
            for end_marker in ("<|im_end|>", "<|endoftext|>"):
                assistant_text = assistant_text.split(end_marker)[0]
            response = assistant_text.strip()
        else:
            # Fallback: slice by input token count (may be slightly wrong for
            # multi-patch image models, but better than nothing)
            input_len = input_ids.shape[1]
            response = _tok.decode(
                output_ids[0][input_len:], skip_special_tokens=True
            ).strip()

        return response

    except Exception as exc:
        logger.warning("AESE FastVLM: inference error: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API — drop-in replacements for static stub methods
# ---------------------------------------------------------------------------

def describe_scene(image: np.ndarray) -> str:
    """
    Use FastVLM to classify the scene into one of the fixed _SCENE_LABELS.

    Prompt engineering: ask VLM to choose from the label vocabulary so the
    output stays compatible with the existing TemporalFeature.scene_label contract.

    Returns "unknown" on model failure or unavailability.
    """
    label_list = ", ".join(f'"{l}"' for l in _SCENE_LABELS if l != "unknown")
    prompt = (
        f"Look at this image and choose the single best matching scene label "
        f"from this list: {label_list}. "
        f"Reply with only the label, nothing else."
    )
    response = _ask(image, prompt, max_new_tokens=10)
    if not response:
        return "unknown"

    # Match response against the label vocabulary (case-insensitive)
    response_lower = response.lower().strip().strip('"').strip("'")
    for label in _SCENE_LABELS:
        if label in response_lower or response_lower in label:
            return label

    logger.debug("AESE FastVLM: scene label %r not in vocabulary — using 'unknown'", response)
    return "unknown"


def count_people(image: np.ndarray) -> int:
    """
    Use FastVLM to count the number of people visible in the frame.

    Returns an int ≥ 0. Returns 0 on model failure or unavailability.
    Same contract as count_characters() — no identity, just a count.
    """
    prompt = (
        "How many people are visible in this image? "
        "Reply with only a single integer (e.g. 0, 1, 2, 3)."
    )
    response = _ask(image, prompt, max_new_tokens=8)
    if not response:
        return 0

    # Extract first integer from the response
    match = re.search(r"\d+", response)
    if match:
        return int(match.group())

    logger.debug("AESE FastVLM: count_people response %r had no integer — returning 0", response)
    return 0


def caption_event(
    image: np.ndarray,
    scene_label: str,
    action_label: str,
    dialogue_text: Optional[str],
) -> str:
    """
    Use FastVLM to generate a rich one-sentence caption for an event.

    Falls back gracefully to None (caller uses template summary).

    Args:
        image:        Representative frame from the event.
        scene_label:  Aggregated scene label (used as context hint).
        action_label: Aggregated action label (used as context hint).
        dialogue_text: Most recent dialogue text in this event (or None).

    Returns:
        str: A one-sentence description of the event, or "" on failure.
    """
    ctx_parts = [f"scene: {scene_label}", f"action: {action_label}"]
    if dialogue_text:
        short_dialogue = dialogue_text[:80] + ("…" if len(dialogue_text) > 80 else "")
        ctx_parts.append(f'dialogue: "{short_dialogue}"')
    context = "; ".join(ctx_parts)

    prompt = (
        f"Describe what is happening in this movie scene in one concise sentence. "
        f"Context — {context}. "
        f"Be specific about actions and setting. Reply with one sentence only."
    )
    response = _ask(image, prompt, max_new_tokens=80)
    return response  # "" on failure — caller falls back to template
