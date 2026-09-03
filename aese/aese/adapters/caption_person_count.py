"""
aese/adapters/caption_person_count.py
Caption-text person-count estimator — Fix 5 (DECISIONS.md §20.5).

Reconciles face-detector-based character counts with what the VLM's own
caption text describes.  Face detectors cannot see people who are partially
visible, facing away, or off to the side — a common pattern in medium-shot
alternating-angle dialogue.  The VLM's own caption frequently describes a
second person even when the face detector counted only one.

IMPORTANT rules (see DECISIONS.md §20.5):
  - This estimate is used ONLY to RAISE max_characters_seen, NEVER to lower it.
  - A face-detector-confirmed count is a hard, real observation (a face was
    actually seen). The caption estimate is a softer signal that could be a VLM
    error. Treating it as a floor rather than a replacement keeps the system
    honest in both directions.
  - Every reconciliation is logged so discrepancies stay visible.
"""
from __future__ import annotations

import re
from typing import Optional

# Explicit count-word patterns — checked first (highest confidence)
_COUNT_WORD_PATTERN = re.compile(
    r"\b(two|three|four)\s+(people|persons|individuals|men|women|characters|figures)\b",
    re.IGNORECASE,
)
_COUNT_WORDS = {"two": 2, "three": 3, "four": 4}

# Individual person-phrase patterns — counted for a heuristic estimate
_PERSON_PHRASES = re.compile(
    r"\b(a\s+man|a\s+woman|another\s+man|another\s+woman|a\s+person|"
    r"someone|a\s+figure|a\s+character|a\s+individual|"
    r"the\s+man|the\s+woman|the\s+person)\b",
    re.IGNORECASE,
)


def estimate_person_count_from_caption(summary_text: str) -> Optional[int]:
    """
    Best-effort estimate of how many distinct people the caption text describes.

    Strategy (in order of confidence):
      1. Explicit count-word match: "two people", "three individuals" → return
         the numeric value directly.
      2. Person-phrase count: count distinct singular person-phrase mentions.
         Two mentions → 2, one mention → 1, zero → None.

    Returns:
        int ≥ 1 if any people are described, or None if the caption contains
        no identifiable person references.  NEVER returns 0 — an absent signal
        (None) is not the same as a confirmed observation of zero people.

    This is a heuristic cross-check, not a precise count.  The VLM's own text
    can also be wrong.  Use only to raise (never lower) the face-detector count.
    """
    if not summary_text or not summary_text.strip():
        return None

    # Pass 1: explicit count words ("two people", "three individuals", …)
    m = _COUNT_WORD_PATTERN.search(summary_text)
    if m:
        word = m.group(1).lower()
        return _COUNT_WORDS.get(word)

    # Pass 2: count distinct person-phrase mentions
    mentions = _PERSON_PHRASES.findall(summary_text)
    if mentions:
        # Deduplicate loosely: "the man" + "a man" in same sentence = 2 people;
        # two "a man" mentions could be the same man described twice.
        # Use raw count as an upper-bound heuristic — prefer over-counting to
        # under-counting since we only use this to raise, not lower.
        return max(1, len(mentions))

    return None
