"""
eval/benchmark_fp16.py
Before/after benchmark for Gemma-4 fp32 vs fp16 weight loading.

REQUIRED by AESE contract §19 — do NOT skip this.
A precision change needs measured evidence, not an assumption that "fp16 = faster".

Usage
-----
    python eval/benchmark_fp16.py [--iterations N] [--output PATH]

Requirements
------------
  - CUDA GPU is required for a meaningful fp16 vs fp32 comparison.
    This script exits cleanly with a message if no GPU is found; the
    _select_dtype() logic is still verified by mock-based unit tests regardless.
  - transformers >= 4.52, accelerate, torch, Pillow

Output
------
  - Console: speedup ratio + side-by-side captions for manual quality review
  - JSON file (default: eval/benchmark_fp16_results.json) for DECISIONS.md reference

After running, copy the measured speedup and caption comparison into DECISIONS.md §19.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.WARNING)  # suppress per-token noise during benchmark
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synthetic test frames (representative of real fight-clip content)
# ---------------------------------------------------------------------------
# These are generated synthetically so the script runs without a real video.
# Replace with real keyframes from your fight clip for production sign-off.

def _make_synthetic_frames(n: int = 5) -> List[np.ndarray]:
    """
    Generate N synthetic 512×288 RGB frames with varied content.
    Each frame has a different dominant colour + noise pattern to exercise
    different image-token paths in the vision tower.
    """
    rng = np.random.default_rng(seed=42)
    frames = []
    dominant_colours = [
        (200, 80,  50),   # warm / fight scene
        (50,  80, 200),   # cool / outdoor
        (30,  30,  30),   # dark / night
        (180, 160, 140),  # neutral / indoor
        (100, 180,  80),  # green / nature
    ]
    for i in range(n):
        base = np.array(dominant_colours[i % len(dominant_colours)], dtype=np.uint8)
        noise = rng.integers(-30, 30, (288, 512, 3), dtype=np.int16)
        frame = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


# ---------------------------------------------------------------------------
# Single-dtype benchmark run
# ---------------------------------------------------------------------------

def _run_for_dtype(
    dtype_label: str,
    frames: List[np.ndarray],
    torch_dtype,  # torch.float16 or torch.float32
) -> Dict:
    """
    Load Gemma-4 with the given dtype, run caption_event() on every frame,
    record wall-clock time and captions.  Returns a result dict.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    MODEL_ID = "google/gemma-4-E2B-it"

    print(f"\n{'='*60}")
    print(f"Loading Gemma-4 with dtype={dtype_label} ({torch_dtype}) ...")

    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    try:
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="sdpa",
        )
    except TypeError:
        model = AutoModelForMultimodalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
    model.eval()
    load_elapsed = time.perf_counter() - load_start
    print(f"  Model loaded in {load_elapsed:.1f}s")

    # Warm-up pass (excluded from timing)
    _caption_once(model, processor, frames[0])

    # Timed inference passes
    infer_start = time.perf_counter()
    captions = [_caption_once(model, processor, frame) for frame in frames]
    infer_elapsed = time.perf_counter() - infer_start

    print(f"  Inference ({len(frames)} frames): {infer_elapsed:.2f}s "
          f"({infer_elapsed/len(frames)*1000:.0f}ms/frame)")

    # Free GPU memory before loading next variant
    del model
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return {
        "dtype_label": dtype_label,
        "torch_dtype": str(torch_dtype),
        "load_elapsed_s": round(load_elapsed, 3),
        "infer_elapsed_s": round(infer_elapsed, 3),
        "ms_per_frame": round(infer_elapsed / len(frames) * 1000, 1),
        "captions": captions,
    }


def _caption_once(model, processor, frame: np.ndarray) -> str:
    """Run a single caption_event-style prompt through the loaded model."""
    import PIL.Image as PILImage
    import torch

    pil_img = PILImage.fromarray(frame)
    # Downscale to 512px (matching _prepare_image logic)
    if max(pil_img.size) > 512:
        scale = 512 / max(pil_img.size)
        pil_img = pil_img.resize(
            (max(1, int(pil_img.width * scale)), max(1, int(pil_img.height * scale))),
            PILImage.LANCZOS,
        )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {
                    "type": "text",
                    "text": (
                        "Describe what is happening in this movie scene in one concise sentence. "
                        "Context -- scene: outdoor; action: fast_action. "
                        "Be specific about actions and setting. Reply with one sentence only."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
        )

    raw = processor.decode(
        output_ids[0][input_len:], skip_special_tokens=True
    ).strip()
    return raw if len(raw) >= 2 else ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Gemma-4 fp32 vs fp16 and record results for DECISIONS.md §19."
    )
    parser.add_argument(
        "--iterations", type=int, default=5,
        help="Number of frames to caption per dtype (default: 5)."
    )
    parser.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent / "benchmark_fp16_results.json"),
        help="Path to write the JSON results file.",
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # GPU check — exit cleanly if no GPU
    # -----------------------------------------------------------------------
    try:
        import torch
    except ImportError:
        print("ERROR: torch is not installed. Cannot run benchmark.", file=sys.stderr)
        sys.exit(1)

    if not torch.cuda.is_available():
        print(
            "\n" + "!" * 60 + "\n"
            "NO CUDA GPU DETECTED — benchmark cannot run.\n"
            "\n"
            "fp16 loading has no benefit and no measurement value on CPU.\n"
            "The _select_dtype() logic is still verified by the mock-based unit\n"
            "tests in tests/test_gemma4_speed_regression.py.\n"
            "\n"
            "If CPU speed is the real constraint, consider int4/int8 quantization\n"
            "via bitsandbytes — that is a different, larger technique not in scope\n"
            "for this contract.  See DECISIONS.md §19.\n"
            "!" * 60
        )
        sys.exit(0)

    device_name = torch.cuda.get_device_name(0)
    print(f"\nGPU detected: {device_name}")
    print(f"Benchmark: fp32 baseline vs fp16, {args.iterations} frames each.\n")

    frames = _make_synthetic_frames(args.iterations)

    # -----------------------------------------------------------------------
    # Run fp32 baseline first, then fp16
    # -----------------------------------------------------------------------
    fp32_result = _run_for_dtype("fp32_baseline", frames, torch.float32)
    fp16_result = _run_for_dtype("fp16", frames, torch.float16)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    speedup = fp32_result["infer_elapsed_s"] / fp16_result["infer_elapsed_s"]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"fp32 inference: {fp32_result['infer_elapsed_s']:.2f}s  "
          f"({fp32_result['ms_per_frame']:.0f}ms/frame)")
    print(f"fp16 inference: {fp16_result['infer_elapsed_s']:.2f}s  "
          f"({fp16_result['ms_per_frame']:.0f}ms/frame)")
    print(f"Speedup:        {speedup:.2f}x")
    print()
    print("SIDE-BY-SIDE CAPTION COMPARISON (manual quality review required)")
    print("-" * 60)
    for i, (c32, c16) in enumerate(
        zip(fp32_result["captions"], fp16_result["captions"])
    ):
        print(f"\nFrame {i+1}:")
        print(f"  fp32: {c32 or '[EMPTY]'}")
        print(f"  fp16: {c16 or '[EMPTY]'}")

    print()
    print("ACTION REQUIRED: Review the captions above for quality regression.")
    print("If fp16 captions are meaningfully degraded, revert gemma4.py to bfloat16")
    print("and document the decision in DECISIONS.md §19.")

    # -----------------------------------------------------------------------
    # Write JSON results
    # -----------------------------------------------------------------------
    results = {
        "gpu": device_name,
        "n_frames": args.iterations,
        "speedup_x": round(speedup, 3),
        "fp32_baseline": fp32_result,
        "fp16": fp16_result,
        "notes": (
            "Run eval/benchmark_fp16.py on your GPU to populate this file. "
            "Copy speedup and caption comparison into DECISIONS.md §19."
        ),
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to: {out_path}")
    print("Copy speedup + caption comparison into DECISIONS.md §19 before shipping.")


if __name__ == "__main__":
    main()
