"""
Appends _supports_batching() and describe_batch() to gemma4.py.
Run: python _patch_gemma4_batch.py
"""
import pathlib

TARGET = pathlib.Path(__file__).parent / "aese" / "adapters" / "gemma4.py"
existing = TARGET.read_text(encoding="utf-8")

if "def _supports_batching" in existing:
    print("Already patched -- skipping.")
    raise SystemExit(0)

BATCH = """

# ---------------------------------------------------------------------------
# Fix 2 / \\u00a722.2 \\u2014 Batched generation across multiple events
# ---------------------------------------------------------------------------


def _supports_batching() -> bool:
    \"\"\"
    Probe whether the loaded processor + model support batched multimodal generation.

    Gemma-4 with Transformers >= 4.52 supports batched text, but batching multiple
    images in a single apply_chat_template call depends on the processor
    implementation.  Runs a 2-item dummy probe and caches the result.

    Returns True if batching is clean; False (sequential fallback) otherwise.
    \"\"\"
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
    \"\"\"
    Process N event keyframes in ONE generate() call (batch dimension).

    Context dicts: {action_label, dialogue_text, scene_labels}.
    Falls back to sequential describe_scene_and_caption() on any failure.
    Never raises; returns ('unknown', '') for events without a keyframe or on error.
    \"\"\"
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
                "Look at this image and respond in EXACTLY this format (two lines):\\n"
                "SCENE: <one label from: " + lbl_str + ">\\n"
                "CAPTION: <one concise sentence describing the action and setting>\\n"
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
"""

TARGET.write_text(existing + BATCH, encoding="utf-8")
print(f"Patched: {TARGET} ({TARGET.stat().st_size} bytes)")
