"""
aese/adapters/_dtype_utils.py
Shared dtype selection utility for AESE VLM adapters.

Centralises the single rule used by both gemma4.py and fastvlm.py:
  - CUDA available  → torch.float16 (GPU fp16 is well-optimised; ~1.5-2x over fp32)
  - CPU / MPS / other → torch.float32

Why float16 and not bfloat16?
  bfloat16 has the same exponent range as fp32 and is therefore more numerically stable
  (no overflow risk on activations with large magnitudes).  However, fp16 is the dtype
  requested by AESE contract §19 for a measured GPU speedup, and the degenerate-output
  guard in gemma4._ask() catches the rare NaN/inf case rather than silently passing it
  through.  If Ampere+ GPU numerical instability becomes a real problem in production,
  revert gemma4.py to bfloat16 — the _select_dtype() abstraction makes that a one-line
  change in this file (or via a per-adapter override).

Why NOT force float16 on CPU?
  Most CPU kernels either lack optimised fp16 implementations or internally upcast to
  fp32 anyway, paying the conversion cost for no benefit.  Apple Silicon (MPS) has partial
  fp16 support but has not been verified for this model — treated conservatively as fp32.
  See DECISIONS.md §19.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _select_dtype() -> "torch.dtype":  # type: ignore[name-defined]
    """
    Return the appropriate weight dtype for the current hardware:
      - torch.float16  if a CUDA GPU is available
      - torch.float32  otherwise (CPU, MPS, or no torch)

    Import torch lazily so this module can be imported without torch installed
    (it will raise ImportError at call time, not at import time of the adapter).
    """
    import torch  # noqa: PLC0415 — intentional lazy import

    if torch.cuda.is_available():
        return torch.float16

    # CPU: fp16 is poorly supported by most kernels (slow or unimplemented ops).
    # MPS (Apple Silicon): partial fp16 support; treated conservatively as fp32
    # until verified for this model class.
    return torch.float32
