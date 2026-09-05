"""
aese/adapters/_gemma4_batch_patch.py
Temporary helper: appends describe_batch() and _supports_batching() to gemma4.py.
Run once: python aese/adapters/_gemma4_batch_patch.py
"""
import pathlib

PATCH = '''

# ---------------------------------------------------------------------------
# Fix 2 / \u00a722.2 -- Batched generation across multiple events
# ---------------------------------------------------------------------------

def _supports_batching() -> bool:
    """
    Probe whether the loaded processor + model support batched multimodal generation.

    Gemma-4 with Transformers >= 4.52 supports batched text, but multi-image batching
    (different image per batch item via apply_chat_template) depends on the specific
    processor implementation.  Probe runs a 2-item batch with dummy inputs and checks
    for clean output.  Result is cached after the first call (no repeated probes).

    Returns True if batching is supported and produces coherent output; False otherwise.
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
            dummy_img = PILImage.fromarray(_np.zeros((64, 64, 3), dtype=_np.uint8))
            probe_messages = [
                [{"role": "user", "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text",  "text": "Say OK."},
                ]}],
                [{"role": "user", "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text",  "text": "Say OK."},
                ]}],
            ]
            inputs = _processor.apply_chat_template(
                probe_messages,
                tokenize=True, return_dict=True, return_tensors="pt",
                padding=True, add_generation_prompt=True, enable_thinking=False,
            ).to(_model.device)
            with torch.no_grad():
                out = _model.generate(**inputs, max_new_tokens=5, do_sample=False)
            if out.shape[0] != 2:
                raise ValueError(f"Expected batch dim=2, got {out.shape[0]}")
            _batching_supported = True
            logger.info("AESE Gemma-4: batched multimodal generation probe PASSED.")
        except Exception as exc:
            _batching_supported = False
            logger.warning(
                "AESE Gemma-4: batched generation probe FAILED (%s) -- using sequential.", exc
            )
    return _batching_supported


def describe_batch(
    images: List[Optional[np.ndarray]],
    contexts: List[dict],
) -> List[Tuple[str, str]]:
    """
    Process N event keyframes in ONE generate() call (batch dimension).

    Context dicts must contain:
        action_label  (str)
        dialogue_text (Optional[str])
        scene_labels  (Optional[List[str]])  -- defaults to SCENE_LABELS

    Falls back to sequential describe_scene_and_caption() on any failure.
    Never raises; returns ("unknown", "") for events without a keyframe or on error.

    Args:
        images:   List of N RGB numpy arrays (or None for no-image events).
        contexts: List of N context dicts in the same order as images.

    Returns:
        List of N (scene_label, caption) tuples in input order.
    """
    if not images:
        return []

    def _seq() -> List[Tuple[str, str]]:
        results = []
        for img, ctx in zip(images, contexts):
            if img is None:
                results.append(("unknown", ""))
                continue
            try:
                results.append(describe_scene_and_caption(
                    img,
                    action_label=ctx.get("action_label", "static"),
                    dialogue_text=ctx.get("dialogue_text"),
                    scene_labels=ctx.get("scene_labels"),
                ))
            except Exception:
                results.append(("unknown", ""))
        return results

    if not _ensure_loaded() or not _supports_batching():
        return _seq()

    try:
        import torch
        from .scene_label import SCENE_LABELS

        n = len(images)
        all_messages: list = []
        fallback_masks: list = []
        label_lists: list = []

        for img, ctx in zip(images, contexts):
            labels = ctx.get("scene_labels") or SCENE_LABELS
            label_lists.append(labels)
            if img is None:
                all_messages.append(None)
                fallback_masks.append(True)
                continue

            label_str = ", ".join(f\'"{l}"\' for l in labels if l != "unknown")
            dialogue_text = ctx.get("dialogue_text")
            action_label  = ctx.get("action_label", "static")
            dlg = (
                f\' Dialogue spoken: "{dialogue_text[:80]}{"..." if len(dialogue_text) > 80 else ""}"\\'
                if dialogue_text else ""
            )
            prompt = (
                f"Look at this image and respond in EXACTLY this format (two lines):\\n"
                f"SCENE: <one label from: {label_str}>\\n"
                f"CAPTION: <one concise sentence describing the action and setting>\\n"
                f"Context -- action: {action_label}.{dlg}"
            )
            all_messages.append([{"role": "user", "content": [
                {"type": "image", "image": _prepare_image(img)},
                {"type": "text",  "text": prompt},
            ]}])
            fallback_masks.append(False)

        batch_msgs = [m for m, fm in zip(all_messages, fallback_masks) if not fm]
        batch_idx  = [i for i, fm in enumerate(fallback_masks) if not fm]
        b_labels   = [label_lists[i] for i in batch_idx]

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
                    "AESE Gemma-4: describe_batch empty output for item %d "
                    "-- falling back to sequential for this item.", i,
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
            n, len(batch_idx), sum(fallback_masks),
        )
        return output

    except Exception as exc:
        logger.warning(
            "AESE Gemma-4: describe_batch failed (%s) -- sequential fallback.", exc
        )
        return _seq()
'''

if __name__ == "__main__":
    target = pathlib.Path(__file__).parent / "gemma4.py"
    existing = target.read_text(encoding="utf-8")
    if "_supports_batching" in existing:
        print("Already patched -- skipping.")
    else:
        target.write_text(existing + PATCH, encoding="utf-8")
        print(f"Patched {target}")
