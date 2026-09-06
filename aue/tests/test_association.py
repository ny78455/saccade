"""
tests/test_association.py
Acceptance tests for association/speaker_character_binder.py.

Key tests (mirrors CharacterNameBinder tests in AESE):
  1. Single visible character → binding accumulates and resolves.
  2. Multiple visible characters → no binding (ambiguous, correctly left unresolved).
  3. Conflicting evidence → unresolved.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock

from aue.association.speaker_character_binder import SpeakerCharacterBinder, apply_resolved_speakers
from aue.types import SpeechSegment


def _make_event(character_labels: list, start_ms: float = 0.0, end_ms: float = 5000.0):
    """Create a mock AESE Event with given character_labels."""
    event = MagicMock()
    event.character_labels = character_labels
    event.start_time_ms = start_ms
    event.end_time_ms = end_ms
    return event


class TestSpeakerCharacterBinder:
    def setup_method(self):
        self.binder = SpeakerCharacterBinder(min_votes_to_resolve=2)

    def test_single_character_resolves_after_min_votes(self):
        """
        SPK_01 speaking while only 'Person A' visible → resolves to 'Person A'
        after min_votes_to_resolve observations.
        """
        event_one_char = _make_event(["Person A"])
        for _ in range(2):
            self.binder.observe("SPK_01", event_one_char)

        resolved = self.binder.resolved_speakers()
        assert "SPK_01" in resolved, "SPK_01 should be resolved"
        assert resolved["SPK_01"] == "Person A"

    def test_multiple_characters_no_binding(self):
        """
        Two characters visible → ambiguous → no binding.
        This is the core single-subject evidence discipline.
        """
        event_two_chars = _make_event(["Person A", "Person B"])
        for _ in range(10):
            self.binder.observe("SPK_01", event_two_chars)

        resolved = self.binder.resolved_speakers()
        assert "SPK_01" not in resolved, (
            "SPK_01 must NOT be resolved when 2 characters visible — ambiguous!"
        )

    def test_empty_character_labels_no_binding(self):
        """Zero characters visible → no binding."""
        event_no_chars = _make_event([])
        for _ in range(5):
            self.binder.observe("SPK_01", event_no_chars)
        assert "SPK_01" not in self.binder.resolved_speakers()

    def test_conflicting_evidence_stays_unresolved(self):
        """
        Two different characters observed equally → conflict → unresolved.
        Mirrors CharacterNameBinder's conflict-rejection rule.
        """
        event_a = _make_event(["Person A"])
        event_b = _make_event(["Person B"])

        # Equal votes — tied
        for _ in range(3):
            self.binder.observe("SPK_01", event_a)
        for _ in range(3):
            self.binder.observe("SPK_01", event_b)

        resolved = self.binder.resolved_speakers()
        assert "SPK_01" not in resolved, (
            "SPK_01 must NOT be resolved with conflicting evidence."
        )

    def test_strong_evidence_wins_over_minority_conflict(self):
        """
        3 votes for Person A, 1 for Person B → resolves to Person A.
        """
        event_a = _make_event(["Person A"])
        event_b = _make_event(["Person B"])

        for _ in range(3):
            self.binder.observe("SPK_01", event_a)
        self.binder.observe("SPK_01", event_b)

        resolved = self.binder.resolved_speakers()
        assert resolved.get("SPK_01") == "Person A"

    def test_reset_clears_evidence(self):
        """reset() clears all evidence."""
        event_a = _make_event(["Person A"])
        for _ in range(3):
            self.binder.observe("SPK_01", event_a)
        self.binder.reset()
        assert self.binder.resolved_speakers() == {}

    def test_multiple_speakers_independent(self):
        """Multiple speaker bindings are tracked independently."""
        event_a = _make_event(["Person A"])
        event_b = _make_event(["Person B"])

        for _ in range(2):
            self.binder.observe("SPK_01", event_a)
            self.binder.observe("SPK_02", event_b)

        resolved = self.binder.resolved_speakers()
        assert resolved.get("SPK_01") == "Person A"
        assert resolved.get("SPK_02") == "Person B"


def test_apply_resolved_speakers():
    """apply_resolved_speakers sets character_id on matching segments."""
    binder = SpeakerCharacterBinder(min_votes_to_resolve=1)
    event_a = _make_event(["Person A"])
    binder.observe("SPK_01", event_a)

    segs = [
        SpeechSegment(
            segment_id="SP_00000", start_ms=0.0, end_ms=1000.0,
            speaker_id="SPK_01", transcript="hello",
        ),
        SpeechSegment(
            segment_id="SP_00001", start_ms=1000.0, end_ms=2000.0,
            speaker_id="SPK_02", transcript="world",
        ),
    ]
    apply_resolved_speakers(segs, binder)

    assert segs[0].character_id == "Person A"
    assert segs[1].character_id is None  # SPK_02 not resolved
