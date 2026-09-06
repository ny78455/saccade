# AUE — Audio Understanding Engine (Module 3)

Part of the Cognitive Movie Understanding System (ASVL → AESE → AUE).

## Purpose

AUE processes a movie's raw audio track into structured `AudioUnderstanding`
objects covering speech, speakers, music, sound events, acoustic emotion, and
audio embeddings — then associates them with AESE's `Event` objects via temporal
overlap.

## Architecture

```
video.mp4
    │
    ▼
[preload_audio()]  ← EXACTLY ONCE — full waveform in memory
    │
    ▼ waveform (in-memory, never re-decoded)
    │
    ├── [VAD (Silero-VAD v5)] ──────────────────────────────── gates ASR + diarization
    │       │
    │       ├── [ASR (faster-whisper int8)] ── SpeechSegment[]
    │       └── [Speaker Embedding (resemblyzer)] ── SpeakerClusterer ── SPK_01, SPK_02...
    │
    ├── [Music Detection (PANNs CNN14)] ─────────────────────── MusicSegment[]
    │       └── [Music Emotion (heuristic)] ─── dict[str, float]
    │
    ├── [Sound Event Detection (shared PANNs)] ──────────────── SoundEvent[]
    │       └── Ambiguous inputs → top_predictions, NOT forced single label
    │
    └── [Acoustic Emotion (waveform-only)] ──────────────────── dict[str, float]
            └── Transcript INTENTIONALLY IGNORED (independence enforced by tests)
                │
                ▼
        [Fusion Layer] ─────────────────────────────── audio_embedding (128-dim)
                │
                ▼
        [AudioUnderstanding] ─────────────────────────────────────────────────────┐
                │                                                                  │
                ▼ (after full-track)                                               │
        [Batch Consolidation]                                                      │
            - Global speaker cluster merge                                         │
            - SpeakerCharacterBinder retroactive resolution                        │
                │                                                                  │
                ▼                                                                  │
        [AESE Association] ── temporal overlap join ── enriched events.jsonl ◄────┘
```

## Quick Start

```bash
cd c:\Users\Nitin\Tribev2\aue

# Install dependencies
pip install -r requirements.txt

# Run on a video (with AESE events for speaker-character binding)
python cli.py \
    --input ../comedy.mp4 \
    --aese-events ../aese/events.jsonl \
    --output audio_events.jsonl

# Run standalone (no AESE events)
python cli.py --input ../comedy.mp4 --output audio_events.jsonl

# Evaluate against reference annotations
python eval/run_eval.py --predictions audio_events.jsonl --reference ref.jsonl

# Run all tests
pytest tests/ -v
```

## Configuration

Edit `config.default.yaml` or pass `--config custom.yaml`:

```yaml
rolling_buffer_seconds: 20.0      # Buffer window size
vad_speech_threshold: 0.5         # Silero-VAD threshold
whisper_model_size: "small"       # base | small | medium | large-v3
speaker_cluster_threshold: 0.45   # Distance threshold (mirrors AESE)
sound_event_confidence_threshold: 0.5
music_detection_threshold: 0.5
```

## Output Format

`audio_events.jsonl` — one `AudioUnderstanding` per rolling buffer window:
```json
{
  "start_ms": 0.0,
  "end_ms": 20000.0,
  "speech_segments": [
    {
      "segment_id": "SP_00000",
      "start_ms": 1200.0, "end_ms": 4500.0,
      "speaker_id": "SPK_01",
      "character_id": "Person A",
      "transcript": "Hello world",
      "words": [{"word": "Hello", "start_ms": 1200.0, "end_ms": 1600.0}],
      "language": "en",
      "asr_confidence": -0.3,
      "low_confidence": false
    }
  ],
  "music_segments": [
    {
      "start_ms": 5000.0, "end_ms": 9000.0,
      "music_probability": 0.87,
      "emotion": {"suspense": 0.72, "action": 0.41, "calm": 0.05, ...},
      "intensity": 0.6, "tension": 0.55, "tempo_bpm": 140.0, "confidence": 0.87
    }
  ],
  "sound_events": [
    {
      "event_type": "Explosion",
      "start_ms": 12000.0, "end_ms": 13000.0,
      "confidence": 0.91,
      "top_predictions": [{"label": "Explosion", "confidence": 0.91}]
    }
  ],
  "acoustic_emotion": {"fear": 0.12, "anger": 0.05, "sadness": 0.08, ...},
  "audio_embedding": [0.12, -0.03, ...],
  "confidence": 0.85
}
```

`audio_events_aese_associated.jsonl` — AESE events enriched with audio context.

## Key Design Patterns (inherited from ASVL + AESE)

| Pattern | Where it came from | AUE implementation |
|---|---|---|
| Single in-memory audio decode | ASVL anti-reload rule | `audio_source.preload_audio()` |
| Exemplar-gallery clustering | AESE §21.3 CharacterClusterer | `SpeakerClusterer` |
| Single-subject evidence binding | AESE CharacterNameBinder | `SpeakerCharacterBinder` |
| Retroactive batch resolution | AESE `apply_resolved_names()` | `_consolidate_speakers_and_bind()` |
| Multi-label emotion output (never string) | spec §3.21 | `MusicSegment.emotion`, `acoustic_emotion` |
| Low-confidence flagging (not discarding) | honesty principle | `SpeechSegment.low_confidence` |
| RTF measurement (not assumed) | spec §6 | logged at end of `pipeline.py::run()` |

## V1 Scope Boundaries (Simplified Stubs — NOT Hidden)

- **Music emotion:** Heuristic (tempo + energy + chroma), not a pretrained model. See DECISIONS.md §AUE-6.
- **Acoustic emotion:** Heuristic (pitch + energy + pauses), not wav2vec SER. See DECISIONS.md §AUE-7.
- **Diarization:** resemblyzer GE2E + exemplar gallery (~25-35% DER). Full pyannote out of scope. See DECISIONS.md §AUE-3.
- **Speaker-character binding:** Temporal overlap + single-visible-character. No lip-sync. See DECISIONS.md §AUE-4.
- **Fusion:** Temporal mean pooling. No learned fusion. See DECISIONS.md §AUE-10.
- **Dialogue semantic score:** Word count proxy. No embedding similarity. See `importance.py`.
- **DER metric:** Simplified (no pyannote collar). See `eval/run_eval.py`.
- **RTF §AUE-8:** Placeholder — fill in after running on comedy.mp4.

## Test Suite

```
tests/test_audio_source.py   # Single-decode regression guard
tests/test_vad.py            # VAD gating correctness
tests/test_speech.py         # Low-confidence flagging, word timestamps
tests/test_diarization.py    # Identity consistency (anti-drift)
tests/test_association.py    # Speaker-character binding evidence discipline
tests/test_music.py          # Multi-label emotion, never single string
tests/test_sound_events.py   # Ambiguous top-k, never forced label
tests/test_integration.py    # Full pipeline, AESE association correctness
```

Run all: `pytest tests/ -v`
