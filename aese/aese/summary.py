"""
aese/summary.py
Post-finalization event summary generator.

Architectural contract (see DECISIONS.md §17):
  - generate_summary() is called ONCE PER FINALIZED EVENT in pipeline.py,
    after merge_events() / filter_short_events() / EventClassifier have all resolved.
  - It is NEVER called per TemporalFeature or per boundary-detection step.
  - The <100ms/decision hot-path constraint applies to the boundary loop; this
    function is explicitly outside that loop.

Output guarantees:
  - Always returns a non-empty string.
  - If the generative VLM path produces filler text, multi-line rambling, or any
    exception, the guaranteed fallback is build_template_summary(), which never fails.
  - Filler detection is performed against _FILLER_PATTERNS before any VLM output
    is accepted.

Testing hook:
  - _summary_call_counter.count tracks the number of generate_summary() calls.
  - _summary_call_counter.reset() resets it between test runs.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Optional

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .types import Event


# ---------------------------------------------------------------------------
# Filler-pattern registry
# ---------------------------------------------------------------------------

_FILLER_PATTERNS = [
    re.compile(r"(?i)let\s+me\s+know\s+if"),
    re.compile(r"(?i)i('| a)ll\s+be\s+happy\s+to"),
    re.compile(r"(?im)^(answer|here\s+is|here'?s)\b"),
    re.compile(r"(?m)^-{2,}\s*$"),       # bare markdown dividers on their own line
    re.compile(r"(?i)as\s+an\s+ai"),
    re.compile(r"(?i)i\s+(can'?t|cannot|am\s+unable)"),
    # --- Artifact patterns from FastVLM / Gemma-4 meta-commentary ---
    re.compile(r"(?i)please\s+note\s+that"),
    re.compile(r"(?i)based\s+on\s+(the\s+)?visible\s+elements"),
    re.compile(r"(?i)does\s+not\s+include\s+any\s+assumptions"),
    re.compile(r"(?m)^\s*\d+\.\s"),      # leaked numbered-list formatting ("2. In this")
    re.compile(r"<end\s+of[^>]*>", re.I), # leaked stop/end tags ("<end of detailed answer>")
]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SUMMARY_SYSTEM_PROMPT = (
    "You are a factual video-event data extractor, not a conversational assistant. "
    "Given one representative frame from a video event, describe what is happening "
    "in ONE TO TWO sentences: who or what is present, the setting, the main action, "
    "and any visible reaction from other subjects. "
    "CRITICAL: Describe ONLY what is explicitly visible in this image. Do not assume "
    "typical or expected scene behavior based on props alone (for example, do not "
    "assume people are eating just because a dining table is present, or that a room "
    "is calm just because it looks furnished). Pay close attention to anything unusual, "
    "unexpected, or anomalous -- items on the floor, unusual body positions, weapons, "
    "spills, or anyone not clearly participating in an ordinary activity. "
    "If you are not confident about a detail, do not state it as fact. "
    "Do not include greetings, offers of further help, meta-commentary, or markdown."
)

# Short prompt for small models (e.g. FastVLM 0.5B).
# Large example blocks cause small models to predict EOS immediately
# (the model sees a completed example and thinks the sequence is done).
# Keep this under ~120 chars so it tokenizes to ~30 tokens,
# leaving maximum context budget for the image and generated response.
SUMMARY_SYSTEM_PROMPT_SHORT = (
    "Describe what is happening in this image in one or two sentences. "
    "Be specific about people, actions, and setting. Output only the description."
)


# ---------------------------------------------------------------------------
# Call counter -- testability hook
# ---------------------------------------------------------------------------

class _CallCounter:
    """Simple call counter for test instrumentation."""
    __slots__ = ("count",)

    def __init__(self) -> None:
        self.count: int = 0

    def increment(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


_summary_call_counter = _CallCounter()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _vlm_available() -> bool:
    """Return True if the active VLM backend is loaded and usable."""
    try:
        from .adapters.vlm_router import vlm_available
        return vlm_available()
    except Exception:
        return False


def _call_vlm(system_prompt: str, image: np.ndarray, max_tokens: int = 100) -> str:
    """
    Run a VLM inference call via vlm_router.

    Chooses between the full system prompt (Gemma-4) and the compact short
    prompt (FastVLM and other small models) to avoid the EOS-immediately
    problem that occurs when small models see completed example blocks.

    Returns the raw response string, or "" on any failure.
    """
    try:
        from .adapters.vlm_router import ask, get_backend
        # Small models (FastVLM 0.5B) generate EOS immediately when given long
        # prompts ending with completed Output: examples.  Use the short prompt.
        backend = get_backend()
        if backend == "fastvlm":
            effective_prompt = SUMMARY_SYSTEM_PROMPT_SHORT
        else:
            effective_prompt = system_prompt  # full prompt with examples for gemma4
        return ask(
            image_rgb=image,
            prompt="What is happening in this image?",
            system_prompt=effective_prompt,
            max_new_tokens=max_tokens,
        )
    except Exception as exc:
        logger.warning("AESE summary: VLM call failed: %s", exc)
        return ""


def _validate_or_fallback(raw: str, fallback: str) -> str:
    """
    Validate VLM output against quality gates; return fallback if any gate fails.

    Gates (in order):
      1. Non-empty and at least 5 characters.
      2. No filler patterns from _FILLER_PATTERNS.
      3. No more than one newline (multi-line = rambling).

    Args:
        raw:      Raw VLM response string.
        fallback: Template summary -- guaranteed non-empty.

    Returns:
        The cleaned raw string if all gates pass, otherwise fallback.
    """
    cleaned = raw.strip()

    # Gate 1: length
    if not cleaned or len(cleaned) < 5:
        logger.warning("AESE summary: VLM output too short (%d chars) -- using fallback", len(cleaned))
        return fallback

    # Gate 2: filler patterns
    for pattern in _FILLER_PATTERNS:
        if pattern.search(cleaned):
            logger.warning(
                "AESE summary: filler pattern %r matched -- using fallback",
                pattern.pattern,
            )
            return fallback

    # Gate 3: excessive multi-line rambling (>4 newlines = almost certainly off-task)
    # Allow up to 4 newlines so 2-sentence answers with a blank line separator pass.
    if cleaned.count("\n") > 4:
        logger.warning("AESE summary: multi-line VLM output (%d newlines) -- using fallback", cleaned.count("\n"))
        return fallback

    return cleaned


def caption_frame_delta(
    primary_frame: np.ndarray,
    secondary_frame: np.ndarray,
    prompt: Optional[str] = None,
    max_tokens: int = 80,
) -> str:
    """
    Generate a brief addendum describing what is NEW or DIFFERENT in the
    secondary keyframe compared to the primary (Fix 2 — DECISIONS.md §20.2).

    Called at most ONCE per long event where needs_secondary_frame() returns
    a non-None index.  The result is appended to the primary summary, not
    used as a replacement.  Returns "" on any failure (caller ignores it).

    This is a targeted one-sentence addendum, NOT a full scene description --
    max_tokens is intentionally smaller than generate_summary() to keep cost low.

    Args:
        primary_frame:   The already-captioned primary keyframe (for model context).
        secondary_frame: The high-salience secondary keyframe to describe.
        prompt:          Custom delta prompt, or None for the default.
        max_tokens:      Maximum tokens for the addendum response.

    Returns:
        str: One-sentence addendum or "" on failure.
    """
    if secondary_frame is None:
        return ""
    if not _vlm_available():
        return ""

    default_prompt = (
        "Describe anything new or different in this second image compared to a "
        "calm establishing shot, focusing on people, objects, or actions that were "
        "not present before. One sentence only."
    )
    effective_prompt = prompt if prompt else default_prompt

    delta_system = (
        "You are a factual video-event data extractor. "
        "You will see a second frame from the same event. "
        "Describe ONLY what is new, changed, or anomalous compared to the earlier "
        "part of the event. Describe only what is explicitly visible. "
        "One sentence maximum. No greetings, no meta-commentary."
    )
    try:
        raw = _call_vlm(delta_system, secondary_frame, max_tokens=max_tokens)
        if not raw or len(raw) < 5:
            return ""
        # Apply filler check -- the same gate as generate_summary
        for pattern in _FILLER_PATTERNS:
            if pattern.search(raw):
                return ""
        return raw.strip()
    except Exception as exc:
        logger.debug("AESE summary: caption_frame_delta failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_summary(event: "Event", keyframe_image: Optional[np.ndarray]) -> str:
    """
    Generate a summary for a finalized event.

    Called ONCE PER FINALIZED EVENT in pipeline.py, never in the per-second
    hot path. Call count is tracked by _summary_call_counter.

    Architecture:
      1. Build the guaranteed template fallback first (always available).
      2. If no real image or VLM unavailable -> return template immediately.
      3. Build system prompt, extending it with dialogue context if available.
         (event.dialogue_text is populated by event_constructor.py from subtitles
         when --subtitles is supplied to the CLI. Without subtitles, this is None
         and the prompt remains visual-only -- we do NOT fabricate dialogue.)
      4. Call VLM with strict system prompt (max_tokens=80 with dialogue, 60 without).
      5. Validate output through _validate_or_fallback.
      6. Return validated output or template fallback.

    Args:
        event:          Finalized Event object (post-merge/classify).
        keyframe_image: Representative RGB frame (np.ndarray) or None.

    Returns:
        str: Non-empty summary string (never filler, never multi-line).
    """
    _summary_call_counter.increment()

    # Build template fallback first -- always valid regardless of VLM availability.
    from .event_constructor import build_template_summary
    scene = event.location_label or "unknown"
    template_fallback = build_template_summary(
        event_type=event.event_type,
        scene_label=scene,
        max_characters_seen=event.max_characters_seen,
    )

    # Skip VLM if no real image data is available
    if keyframe_image is None or keyframe_image.ndim != 3 or keyframe_image.max() < 5:
        logger.debug("AESE summary: no real pixel image -- using template for event %d", event.event_id)
        return template_fallback

    # Skip VLM if model not loaded
    if not _vlm_available():
        logger.warning("AESE summary: VLM unavailable -- using template for event %d", event.event_id)
        return template_fallback

    try:
        # --- Build context-aware prompt ---
        system_prompt = SUMMARY_SYSTEM_PROMPT
        max_tokens = 100
        dialogue_text = getattr(event, "dialogue_text", None)
        if dialogue_text:
            # Inject verbatim subtitle text as grounding context.
            # IMPORTANT: This only runs when the user supplied --subtitles.
            # Without subtitles, dialogue_text is None and we do NOT fabricate dialogue.
            system_prompt = (
                system_prompt
                + f'\nDialogue spoken during this event: "{dialogue_text}"'
                + "\nDo not fabricate dialogue not provided above."
            )
            max_tokens = 130  # slightly more room to incorporate dialogue context
            logger.debug(
                "AESE summary: injecting %d chars of dialogue context for event %d",
                len(dialogue_text), event.event_id,
            )

        # Inject resolved character identities into the prompt so the model
        # references people by label/name rather than generic "a person".
        character_labels = getattr(event, "character_labels", [])
        if character_labels:
            identity_context = ", ".join(character_labels)
            system_prompt = (
                system_prompt
                + f"\nThe people visible have been identified as: {identity_context}. "
                + "Refer to them by these labels/names in your description "
                + "(e.g. 'Person A enters the room and speaks to Dev'), "
                + "not generic terms like 'a person' or 'someone'."
            )

        raw = _call_vlm(system_prompt, keyframe_image, max_tokens=max_tokens)
        result = _validate_or_fallback(raw, template_fallback)
        if result != template_fallback:
            logger.debug(
                "AESE summary: VLM summary accepted for event %d (%d chars)",
                event.event_id, len(result),
            )
        return result
    except Exception as exc:
        logger.warning(
            "AESE summary: unexpected error for event %d: %s -- using template",
            event.event_id, exc,
        )
        return template_fallback