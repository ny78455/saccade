"""
aue/cli.py
Command-line interface for the Audio Understanding Engine (AUE).

Usage:
    python cli.py --input movie.mp4 --aese-events events.jsonl --output audio_events.jsonl
    python cli.py --input movie.mp4 --output audio_events.jsonl  # standalone, no AESE events
    python cli.py --input movie.mp4 --config custom.yaml --output audio_events.jsonl

The output is a JSONL file where each line is an AudioUnderstanding segment
enriched with AESE event association context.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Allow running from the aue/ directory without installing the package
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aue.cli")


def _load_aese_events(events_path: str) -> list:
    """Load AESE events from a JSONL file. Returns list of dicts."""
    if not events_path:
        return []
    path = Path(events_path)
    if not path.exists():
        logger.error("AESE events file not found: %s", events_path)
        return []
    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed AESE event line: %s", exc)
    logger.info("Loaded %d AESE events from %s", len(events), events_path)
    return events


def _audio_understanding_to_dict(au) -> dict:
    """Convert an AudioUnderstanding to a JSON-serializable dict."""
    import numpy as np

    def _maybe_list(arr):
        if arr is None:
            return None
        if isinstance(arr, np.ndarray):
            return arr.tolist()
        return arr

    return {
        "start_ms": au.start_ms,
        "end_ms": au.end_ms,
        "speech_segments": [
            {
                "segment_id": s.segment_id,
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "speaker_id": s.speaker_id,
                "character_id": s.character_id,
                "transcript": s.transcript,
                "words": [
                    {"word": w.word, "start_ms": w.start_ms, "end_ms": w.end_ms}
                    for w in s.words
                ],
                "language": s.language,
                "asr_confidence": round(s.asr_confidence, 4),
                "low_confidence": s.low_confidence,
            }
            for s in au.speech_segments
        ],
        "speakers": au.speakers,
        "music_segments": [
            {
                "start_ms": m.start_ms,
                "end_ms": m.end_ms,
                "music_probability": round(m.music_probability, 4),
                "emotion": {k: round(v, 4) for k, v in m.emotion.items()},
                "intensity": round(m.intensity, 4),
                "tension": round(m.tension, 4),
                "tempo_bpm": m.tempo_bpm,
                "confidence": round(m.confidence, 4),
            }
            for m in au.music_segments
        ],
        "sound_events": [
            {
                "event_type": e.event_type,
                "start_ms": e.start_ms,
                "end_ms": e.end_ms,
                "confidence": round(e.confidence, 4),
                "top_predictions": [
                    {"label": lbl, "confidence": round(conf, 4)}
                    for lbl, conf in e.top_predictions[:3]
                ],
            }
            for e in au.sound_events
        ],
        "acoustic_emotion": {k: round(v, 4) for k, v in au.acoustic_emotion.items()},
        "audio_embedding": _maybe_list(au.audio_embedding),
        "confidence": round(au.confidence, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AUE — Audio Understanding Engine (Module 3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py --input movie.mp4 --aese-events events.jsonl --output audio_events.jsonl
  python cli.py --input movie.mp4 --output out.jsonl --whisper-model base
  python cli.py --input movie.mp4 --config custom.yaml --output out.jsonl
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input video file")
    parser.add_argument(
        "--aese-events", "-e", default=None,
        help="Path to AESE events.jsonl (from Module 2). Optional — AUE runs standalone if omitted.",
    )
    parser.add_argument("--output", "-o", required=True, help="Output JSONL path")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to AUE config YAML. Defaults to config.default.yaml.",
    )
    parser.add_argument(
        "--whisper-model", default=None,
        help="Override whisper model size (base/small/medium/large-v3).",
    )
    parser.add_argument(
        "--vad-threshold", type=float, default=None,
        help="Override VAD speech threshold (0.0–1.0). Default: 0.5.",
    )
    parser.add_argument(
        "--no-association", action="store_true",
        help="Skip AESE event association (output raw AudioUnderstanding segments only).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input
    if not os.path.exists(args.input):
        logger.error("Input file not found: %s", args.input)
        return 1

    # Load config
    from aue.config import load_config
    from aue.types import AUEConfig
    config = load_config(args.config)

    if args.whisper_model:
        config.whisper_model_size = args.whisper_model
    if args.vad_threshold is not None:
        config.vad_speech_threshold = args.vad_threshold

    logger.info(
        "AUE CLI: input=%s | whisper=%s | vad_threshold=%.2f",
        args.input, config.whisper_model_size, config.vad_speech_threshold,
    )

    # Load AESE events
    aese_events = _load_aese_events(args.aese_events) if args.aese_events else []

    # Run pipeline
    t_start = time.perf_counter()
    from aue.pipeline import run
    audio_understandings = run(
        video_path=args.input,
        aese_events=aese_events,
        config=config,
    )
    t_elapsed = time.perf_counter() - t_start

    logger.info(
        "AUE CLI: pipeline complete in %.1f s | %d AudioUnderstanding objects",
        t_elapsed, len(audio_understandings),
    )

    # AESE event association
    associated_events = []
    if aese_events and not args.no_association:
        from aue.aese_association import associate_with_events
        associated_events = associate_with_events(audio_understandings, aese_events)
        logger.info(
            "AUE CLI: associated %d audio segments with %d AESE events.",
            len(audio_understandings), len(associated_events),
        )

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        # Write AudioUnderstanding segments
        for au in audio_understandings:
            d = _audio_understanding_to_dict(au)
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if associated_events:
        assoc_path = out_path.parent / (out_path.stem + "_aese_associated.jsonl")
        with open(assoc_path, "w", encoding="utf-8") as f:
            for ev_dict in associated_events:
                f.write(json.dumps(ev_dict, ensure_ascii=False) + "\n")
        logger.info("AUE CLI: AESE association written to %s", assoc_path)

    logger.info("AUE CLI: output written to %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
