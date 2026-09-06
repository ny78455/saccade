"""
aue/eval/run_eval.py
AUE Evaluation Harness — implements Section 3.22 metrics.

Metrics computed:
  - WER/CER:  Word/Character Error Rate for speech transcription.
  - DER:       Diarization Error Rate (speaker assignment).
  - F1/IoU:   For music segmentation and sound-event detection.
  - Temporal localization error: mean absolute start/end timestamp error.

Usage:
    python eval/run_eval.py --predictions audio_events.jsonl --reference reference.jsonl

Reference format (one JSON per line):
  {"start_ms": 0, "end_ms": 5000, "type": "speech", "transcript": "Hello world",
   "speaker": "SPK_01", "music": false, "sound_event": null}

Qualitative spot-checks (printed to stdout):
  These are the Section 3.22 QA questions:
  - "Who said this?" — speaker diarization spot-check
  - "What did John say before the explosion?" — cross-event retrieval spot-check
  Full automated QA scoring is future work (requires a retrieval/memory layer
  that does not exist yet in V1 of the Cognitive Movie Understanding System).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WER/CER
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return text.lower().split()


def _edit_distance(a: List, b: List) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = list(reference.lower())
    hyp_chars = list(hypothesis.lower())
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return _edit_distance(ref_chars, hyp_chars) / len(ref_chars)


# ---------------------------------------------------------------------------
# DER (simplified: speaker label agreement)
# ---------------------------------------------------------------------------

def compute_der(
    pred_segments: List[dict],
    ref_segments: List[dict],
) -> float:
    """
    Simplified DER: fraction of reference speech duration where the predicted
    speaker label disagrees with the reference label.
    Full pyannote DER (with collar) is future work.
    """
    total_ref_ms = 0.0
    error_ms = 0.0

    for ref in ref_segments:
        r_start = float(ref.get("start_ms", 0))
        r_end = float(ref.get("end_ms", 0))
        r_spk = str(ref.get("speaker", ""))
        dur = r_end - r_start
        total_ref_ms += dur

        # Find overlapping predicted segment
        best_overlap = 0.0
        best_pred_spk = None
        for pred in pred_segments:
            p_start = float(pred.get("start_ms", 0))
            p_end = float(pred.get("end_ms", 0))
            p_spk = str(pred.get("speaker_id", ""))
            overlap = max(0.0, min(r_end, p_end) - max(r_start, p_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_pred_spk = p_spk

        # Count error if speaker disagrees or no prediction
        if best_pred_spk is None or best_pred_spk != r_spk:
            error_ms += dur

    if total_ref_ms <= 0:
        return 0.0
    return error_ms / total_ref_ms


# ---------------------------------------------------------------------------
# IoU for segmentation (music / SED)
# ---------------------------------------------------------------------------

def compute_segment_iou(pred_segments: List[dict], ref_segments: List[dict]) -> float:
    """Macro-average IoU over reference segments."""
    if not ref_segments:
        return 0.0
    ious = []
    for ref in ref_segments:
        r_start = float(ref.get("start_ms", 0))
        r_end = float(ref.get("end_ms", 0))
        best_iou = 0.0
        for pred in pred_segments:
            p_start = float(pred.get("start_ms", 0))
            p_end = float(pred.get("end_ms", 0))
            inter = max(0.0, min(r_end, p_end) - max(r_start, p_start))
            union = (r_end - r_start) + (p_end - p_start) - inter
            iou = inter / max(union, 1e-9)
            best_iou = max(best_iou, iou)
        ious.append(best_iou)
    return float(sum(ious) / len(ious)) if ious else 0.0


# ---------------------------------------------------------------------------
# Temporal localization error
# ---------------------------------------------------------------------------

def compute_temporal_localization_error(
    pred_segments: List[dict],
    ref_segments: List[dict],
) -> dict:
    """Mean absolute error in start_ms and end_ms for matched segments."""
    if not ref_segments or not pred_segments:
        return {"mean_start_error_ms": None, "mean_end_error_ms": None}

    start_errors = []
    end_errors = []

    for ref in ref_segments:
        r_start = float(ref.get("start_ms", 0))
        r_end = float(ref.get("end_ms", 0))
        # Match to nearest predicted segment by center point
        r_center = (r_start + r_end) / 2
        best_dist = float("inf")
        best_pred = None
        for pred in pred_segments:
            p_start = float(pred.get("start_ms", 0))
            p_end = float(pred.get("end_ms", 0))
            p_center = (p_start + p_end) / 2
            dist = abs(r_center - p_center)
            if dist < best_dist:
                best_dist = dist
                best_pred = pred

        if best_pred:
            start_errors.append(abs(r_start - float(best_pred.get("start_ms", 0))))
            end_errors.append(abs(r_end - float(best_pred.get("end_ms", 0))))

    return {
        "mean_start_error_ms": round(sum(start_errors) / len(start_errors), 1) if start_errors else None,
        "mean_end_error_ms": round(sum(end_errors) / len(end_errors), 1) if end_errors else None,
    }


# ---------------------------------------------------------------------------
# Load predictions
# ---------------------------------------------------------------------------

def _load_jsonl(path: str) -> List[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _extract_speech_segments(records: List[dict]) -> List[dict]:
    segs = []
    for r in records:
        for s in r.get("speech_segments", []):
            segs.append(s)
    return segs


def _extract_music_segments(records: List[dict]) -> List[dict]:
    segs = []
    for r in records:
        for m in r.get("music_segments", []):
            segs.append(m)
    return segs


def _extract_sound_events(records: List[dict]) -> List[dict]:
    evs = []
    for r in records:
        for e in r.get("sound_events", []):
            evs.append(e)
    return evs


# ---------------------------------------------------------------------------
# Qualitative spot-checks (Section 3.22)
# ---------------------------------------------------------------------------

def qualitative_spotcheck(pred_records: List[dict]) -> None:
    """
    Print qualitative spot-checks per Section 3.22.
    These are manual review questions — not auto-scored.
    Full automated QA scoring requires a retrieval/memory layer (future work).
    """
    print("\n" + "=" * 60)
    print("QUALITATIVE SPOT-CHECKS (Section 3.22)")
    print("=" * 60)

    # Q1: Who said what?
    print("\n[Q1] Speaker attribution samples:")
    all_segs = _extract_speech_segments(pred_records)
    shown = 0
    for seg in all_segs:
        if seg.get("transcript") and not seg.get("low_confidence", True):
            speaker = seg.get("speaker_id", "?")
            char = seg.get("character_id", "unresolved")
            t = seg.get("transcript", "")[:80]
            print(f"  [{seg.get('start_ms', 0)/1000:.1f}s] {speaker} ({char}): {t!r}")
            shown += 1
            if shown >= 5:
                break
    if shown == 0:
        print("  (no high-confidence segments found)")

    # Q2: Cross-event query (simplified)
    print("\n[Q2] Sound events with high confidence (for cross-event retrieval):")
    all_sed = _extract_sound_events(pred_records)
    explosions = [e for e in all_sed if e.get("confidence", 0) > 0.7]
    if explosions:
        for ev in explosions[:3]:
            print(f"  [{ev['start_ms']/1000:.1f}–{ev['end_ms']/1000:.1f}s] "
                  f"{ev['event_type']} (conf={ev['confidence']:.2f})")
    else:
        print("  (no high-confidence sound events found)")

    print("\n[NOTE] Full automated QA scoring (retrieval-based) is future work.")
    print("       See DECISIONS.md §AUE-8 for V1 scope boundaries.")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="AUE Evaluation Harness (Section 3.22)")
    parser.add_argument("--predictions", "-p", required=True, help="Predicted audio_events.jsonl")
    parser.add_argument(
        "--reference", "-r", default=None,
        help="Reference annotations JSONL (optional — spot-checks only if omitted).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    pred_records = _load_jsonl(args.predictions)
    print(f"\nLoaded {len(pred_records)} prediction records from {args.predictions}")

    pred_speech = _extract_speech_segments(pred_records)
    pred_music = _extract_music_segments(pred_records)
    pred_sed = _extract_sound_events(pred_records)

    print(f"  Speech segments: {len(pred_speech)}")
    print(f"  Music segments:  {len(pred_music)}")
    print(f"  Sound events:    {len(pred_sed)}")

    low_conf = sum(1 for s in pred_speech if s.get("low_confidence", False))
    if pred_speech:
        print(f"  Low-confidence speech: {low_conf}/{len(pred_speech)} ({100*low_conf//len(pred_speech)}%)")

    if args.reference:
        ref_records = _load_jsonl(args.reference)
        ref_speech = _extract_speech_segments(ref_records)
        ref_music = _extract_music_segments(ref_records)
        ref_sed = _extract_sound_events(ref_records)

        print("\n--- SPEECH METRICS ---")
        wers = []
        cers = []
        for ref_seg in ref_speech:
            ref_text = ref_seg.get("transcript", "")
            # Find matching predicted segment by time
            start = float(ref_seg.get("start_ms", 0))
            end = float(ref_seg.get("end_ms", 0))
            for pred_seg in pred_speech:
                p_start = float(pred_seg.get("start_ms", 0))
                p_end = float(pred_seg.get("end_ms", 0))
                if p_start < end and start < p_end:
                    pred_text = pred_seg.get("transcript", "")
                    wers.append(compute_wer(ref_text, pred_text))
                    cers.append(compute_cer(ref_text, pred_text))
                    break
        if wers:
            print(f"  WER: {sum(wers)/len(wers):.3f}  (n={len(wers)})")
            print(f"  CER: {sum(cers)/len(cers):.3f}  (n={len(cers)})")
        else:
            print("  No matched speech segments for WER/CER computation.")

        print("\n--- DIARIZATION METRICS ---")
        der = compute_der(pred_speech, ref_speech)
        print(f"  DER (simplified): {der:.3f}")
        print("  NOTE: This is a simplified DER without collar. "
              "Full pyannote DER is future work. See DECISIONS.md §AUE-3.")

        print("\n--- MUSIC SEGMENTATION ---")
        music_iou = compute_segment_iou(pred_music, ref_music)
        print(f"  Music IoU: {music_iou:.3f}")

        print("\n--- SOUND EVENT DETECTION ---")
        sed_iou = compute_segment_iou(pred_sed, ref_sed)
        print(f"  SED IoU: {sed_iou:.3f}")

        print("\n--- TEMPORAL LOCALIZATION ---")
        speech_loc = compute_temporal_localization_error(pred_speech, ref_speech)
        print(f"  Speech: start_err={speech_loc['mean_start_error_ms']} ms, "
              f"end_err={speech_loc['mean_end_error_ms']} ms")
        music_loc = compute_temporal_localization_error(pred_music, ref_music)
        print(f"  Music:  start_err={music_loc['mean_start_error_ms']} ms, "
              f"end_err={music_loc['mean_end_error_ms']} ms")
    else:
        print("\n[No reference provided — skipping quantitative metrics.]")

    qualitative_spotcheck(pred_records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
