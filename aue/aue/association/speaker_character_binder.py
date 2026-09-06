"""
aue/aue/association/speaker_character_binder.py
Evidence-based speaker cluster → visual character binding.

DIRECTLY PORTS CharacterNameBinder from AESE (aese/adapters/character_naming.py).
Same evidence discipline, same design rationale:
  - Bind only when EXACTLY ONE character (cluster) is visible in the concurrent
    AESE Event. Multiple visible characters → ambiguous → silently discard.
  - Conflicting votes → leave unresolved (an unresolved SPK_01 is correct output;
    a wrong binding is a worse failure than no binding).
  - Retroactive: the batch pass applies resolved bindings across the entire track
    after full processing. Online decisions remain anonymous SPK_XX.

Why reuse this pattern rather than building a parallel mechanism:
  The underlying problem is identical — bind an anonymous cluster ID (SPK_01)
  to a named entity (visual character "Person A") using single-subject temporal
  co-occurrence evidence. Solving the same problem twice with slightly different
  code creates inconsistency and doubles the maintenance burden. See DECISIONS.md §AUE-4.

V1 scope (per spec §1.2):
  Lip-movement association is OUT OF SCOPE. V1 uses temporal overlap +
  single-visible-character evidence only. This is documented, not hidden.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence accumulator — mirrors ClusterNameEvidence in AESE
# ---------------------------------------------------------------------------

@dataclass
class SpeakerCharacterEvidence:
    """Accumulated character votes for one speaker cluster."""
    votes: Dict[str, int] = field(default_factory=dict)
    resolved_character: Optional[str] = None


# ---------------------------------------------------------------------------
# SpeakerCharacterBinder — mirrors CharacterNameBinder from AESE
# ---------------------------------------------------------------------------

class SpeakerCharacterBinder:
    """
    Accumulates temporal co-occurrence evidence and resolves speaker cluster →
    visual character bindings only when evidence is strong and unambiguous.

    Rules (mirrors CharacterNameBinder §1–4):
      1. Only accumulate when EXACTLY ONE character is visible (character_labels).
      2. Resolve only when the top character beats ALL rivals by vote count.
      3. Tied / conflicting evidence → leave unresolved.
      4. Retroactive relabeling only — never feeds into online pipeline logic.

    Args:
        min_votes_to_resolve: Co-occurrence observations needed before binding.
            Default 2 (stricter than AESE's 1, because audio-visual temporal
            overlap is noisier than subtitle vocative naming).
    """

    def __init__(self, min_votes_to_resolve: int = 2) -> None:
        self.evidence: Dict[str, SpeakerCharacterEvidence] = {}
        self.min_votes = min_votes_to_resolve

    def observe(
        self,
        speaker_id: str,
        concurrent_aese_event,  # aese.types.Event (type-checked at runtime)
    ) -> None:
        """
        Record a temporal co-occurrence between a speaker and an AESE event.

        Fires only when EXACTLY ONE character is visible in the event.
        Multiple visible characters → ambiguous → silently discard.
        This is the same single-subject discipline as CharacterNameBinder.observe().

        Args:
            speaker_id:            Anonymous speaker ID, e.g. "SPK_01".
            concurrent_aese_event: AESE Event object whose time range overlaps
                                   this speech segment.
        """
        character_labels = getattr(concurrent_aese_event, "character_labels", [])

        if len(character_labels) != 1:
            # Ambiguous: multiple people visible (or none confirmed).
            # Cannot attribute speech to a specific character.
            return

        character = character_labels[0]
        ev = self.evidence.setdefault(speaker_id, SpeakerCharacterEvidence())
        ev.votes[character] = ev.votes.get(character, 0) + 1

        top_char = max(ev.votes, key=ev.votes.get)
        top_count = ev.votes[top_char]
        rival_total = sum(v for c, v in ev.votes.items() if c != top_char)

        if top_count >= self.min_votes and top_count > rival_total:
            if ev.resolved_character != top_char:
                ev.resolved_character = top_char
                logger.info(
                    "AUE speaker_binder: speaker %r resolved to character %r (votes=%d)",
                    speaker_id, top_char, top_count,
                )
        else:
            # Conflicting evidence — revert rather than commit a wrong binding
            if ev.resolved_character is not None:
                logger.info(
                    "AUE speaker_binder: speaker %r became conflicted — unresolved.",
                    speaker_id,
                )
            ev.resolved_character = None

    def resolved_speakers(self) -> Dict[str, str]:
        """
        Return speaker clusters with confirmed character bindings only.
        Unresolved speakers are omitted — they remain as SPK_XX.
        Mirrors CharacterNameBinder.resolved_names().
        """
        return {
            spk: ev.resolved_character
            for spk, ev in self.evidence.items()
            if ev.resolved_character is not None
        }

    def reset(self) -> None:
        """Reset all evidence (for test isolation)."""
        self.evidence = {}


def apply_resolved_speakers(
    speech_segments: List,
    binder: "SpeakerCharacterBinder",
) -> None:
    """
    Replace anonymous speaker IDs with resolved character IDs across all segments.

    BATCH POST-PROCESS only — runs after the full clip is processed.
    Applies retroactively: a segment at t=2s gets the character evidenced at t=15s
    because they are the same physical person.

    Mirrors AESE's apply_resolved_names() — same pattern, same justification.

    Args:
        speech_segments: List of SpeechSegment objects (mutated in place).
        binder:          SpeakerCharacterBinder after full-track processing.
    """
    resolved = binder.resolved_speakers()
    if not resolved:
        return

    logger.info(
        "AUE speaker_binder: applying resolved speakers to %d segments: %s",
        len(speech_segments), resolved,
    )
    for seg in speech_segments:
        if seg.speaker_id in resolved:
            seg.character_id = resolved[seg.speaker_id]
