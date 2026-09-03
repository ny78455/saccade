# AESE — Adaptive Event Segmentation Engine (Module 2)

Converts a stream of `FramePacket`s (Module 1 ASVL output) into discrete
semantic `Event` objects representing coherent, human-legible units of a film.
Operates **online** — no access to future frames, max 2-second decision delay.

---

## Quick Start

```bash
# Install dependencies (from aese/ root)
pip install -r requirements.txt

# Run on Module 1's manifest (manifest-replay mode)
python cli.py --input ../asvl/out/manifest.jsonl --output events.jsonl

# Run with real video frames (enables scene label, character count, real embeddings)
python cli.py --input ../asvl/out/manifest.jsonl --output events.jsonl \
              --video ../comedy.mp4

# Run with custom config
python cli.py --input ../asvl/out/manifest.jsonl --output events.jsonl \
              --config config.default.yaml --threshold 0.65

# Run evaluation
python eval/run_eval.py --events events.jsonl

# Run tests
python -m pytest tests/ -v
```

---

## Architecture

```
FramePackets (Module 1)
       │
       ▼
FeatureAggregator ──→ one TemporalFeature / second
       │
       ▼
ContextBuffer (45s rolling window)
       │
       ├──→ BoundarySignals (A-F)
       ├──→ EmbeddingChange (G)
       └──→ PredictionError (H)
              │
              ▼
         BoundaryFusion (weighted sum)
              │
              ▼
    CandidateDetector + ConfidenceHold (≤2s)
              │
              ▼
       EventConstructor
              │
              ├──→ KeyframeSelector
              ├──→ EventEmbedding (mean pool)
              └──→ EventClassifier (rule-based)
                       │
                       ▼
                  OnlineMerger
                       │
                       ▼
                  EventGraph
                       │
                       ▼
                  yield Event
```

---

## Output Format

One `Event` per line in `events.jsonl`:

```json
{
  "event_id": 0,
  "start_time_ms": 0.0,
  "end_time_ms": 26000.0,
  "duration_ms": 26000.0,
  "importance": 0.31,
  "confidence": 0.87,
  "summary": "Dialogue event in indoor, 2 people present",
  "boundary_reason": "scene_change",
  "event_type": "Dialogue",
  "location_label": "indoor",
  "character_count_range": [1, 2],
  "max_characters_seen": 2
}
```

> **Note on `summary`:** Summaries are **template-generated strings** — NOT
> LLM-generated prose. The format is `"<action> event in <scene>, <n> people present"`.
> This is an intentional design decision (§5.9). A real language generation step
> is out of scope for Module 2.

> **Note on `character_count_range`:** AESE does **not** perform character
> identification or re-identification. `character_count_range` reports the
> distinct face *counts* observed per second within the event, not distinct
> individuals. `max_characters_seen` is the single largest count observed.
> e.g. `[0, 1, 2]` means some seconds had 0 faces detected, some had 1, some had 2
> — it does **not** mean 3 people were identified.
> See DECISIONS.md §4 and §14.

---

## Stub Inventory

The following components are explicitly stubbed (heuristics, not trained models).
See `DECISIONS.md` for full reasoning.

| Component | File | Status | What it does | What it doesn't do |
|---|---|---|---|---|
| Embedding | `adapters/embedding.py` | **Real** (CLIP) | CLIP ViT-B/32 + hash fallback | No fine-tuning |
| Scene label | `adapters/scene_label.py` | **STUB** | Zero-shot CLIP against 12 labels | Not a scene-graph model |
| Character detection | `adapters/character_stub.py` | **STUB** | Face count via Haar/DNN | No identity, tracking, or names |
| Action label | `adapters/action_stub.py` | **STUB** | 3-bucket motion_score threshold | Not an action recognition model |
| Music mood | `adapters/music_mood.py` | **STUB heuristic** | Energy + flux threshold bucket | Not a music classifier |
| Camera cue | `adapters/camera_cues.py` | **Real** | Derived from Module 1 scene_change | No pan/tilt/zoom/dissolve |
| Emotion signal | `boundary/signals.py` | **STUB (always 0.0)** | Returns 0.0 | No emotion model at all |
| Prediction error | `boundary/prediction_error.py` | **V1 heuristic** | Linear extrapolation | Not a temporal transformer |
| Boundary fusion | `boundary/fusion.py` | **V1 heuristic** | Weighted sum | Not a learned MLP |
| Summary | `event_constructor.py` | **Template** | Format string | Not LLM-generated |
| Event type | `event_classifier.py` | **Rule-based** | 4 priority rules | Not a learned classifier |
| Event graph causes | `event_graph.py` | **Omitted** | Returns [] | No causal inference |

---

## Non-Functional Requirements (§8)

| Requirement | Target | How verified |
|---|---|---|
| Latency per decision | p95 < 100ms | Logged at end-of-run |
| Peak RSS | < 1GB | Logged via psutil |
| Max boundary delay | ≤ 2000ms | `test_integration.py::test_candidate_detector_max_hold_not_exceeded` |
| No future frames | Online only | Static inspection — `pipeline.py` is a pure generator with no lookahead |
| No fragmentation | < clip duration events | `test_integration.py::test_no_fragmentation_single_scene` |

---

## Configuration

See `config.default.yaml` for all options.

**Weights note:** The source spec weights summed to **1.05** (a bug). They are
renormalized to 1.0 in `config.default.yaml` and in `AESEConfig` defaults.
See `DECISIONS.md §1`.

---

## Running in Live Mode (from Module 1)

```python
from asvl.pipeline import run as asvl_run
from asvl.config import load_config as asvl_load_config
from aese.pipeline import run as aese_run
from aese.config import load_config as aese_load_config

asvl_config = asvl_load_config()
aese_config = aese_load_config()

packet_stream = asvl_run("comedy.mp4", asvl_config)
for event in aese_run(packet_stream, aese_config):
    print(event)
```

Note: In live mode, `FramePacket.image` is populated with real RGB frames,
enabling full-quality embeddings and scene label inference.

---

## Known Limitations

1. **Manifest-replay mode has degraded embedding quality** — black placeholder
   frames produce hash/histogram embeddings, not CLIP semantics.
   Use `--video` flag to provide real frames.

2. **Character detection is face-count-only** — no identity or tracking.
   Character boundary signals are weak in scenes with consistent cast.

3. **Emotion signal is always 0.0** — the emotion weight in the fusion
   formula is effectively inactive until a real emotion model is added.

4. **Summary is template-based** — not suitable for end-user display as-is.
   A downstream LLM summarization pass would be needed.

---

## Accuracy Limits and Known Constraints

### Scene Classification

AESE uses zero-shot CLIP classification against a fixed 15-label vocabulary
(`kitchen`, `living room`, `bedroom`, `office`, `hallway`, `street`, `village`,
`forest`, `beach`, `outdoor field`, `vehicle interior`, `rooftop`, `restaurant`,
`stage/studio`, `unknown`).

**This is a coarse heuristic, not a trained scene-recognition model.** Expect it to be
correct "often, not always" — particularly for:
- Tight close-ups (minimal background context)
- Low-light or stylized cinematography
- Transitional or ambiguous shots (e.g. a window seat that is both "indoor" and "outdoor")
- Labels outside the fixed vocabulary

When CLIP is unavailable (no GPU or model not downloaded), AESE falls back to a
color-temperature heuristic that is less accurate still. The CLI prints which mode
is active at startup (`SCENE CLASSIFICATION MODE`).

**Do not rely on scene labels for precise location identification or downstream reasoning
that requires factual accuracy.**

### Character Identity

AESE uses face detection to count people per second and cluster faces into consistent
anonymous IDs (e.g. "Person A", "Person B") across a single video. This is **not**
real-world identity recognition.

- **Default behavior:** anonymous labels only. AESE will never automatically identify
  or name a real person, including public figures.
- **Opt-in named matching:** if you supply labeled reference photos via
  `--character-references refs/john.jpg=John`, AESE will match detected faces against
  those specific references and use the provided name when the match score is within
  threshold. This only works for faces you have explicitly enrolled.
- **Accuracy ceiling:** face matching uses CLIP image embeddings of face crops, which
  is approximate. Profile shots, disguises, or low-resolution frames may fail to match.

### Narrative Summaries

- Without `--video`: summaries are template strings (`"Dialogue event in office, 2 people present"`).
- With `--video` and FastVLM available: summaries describe what is *visible* in the keyframe.
- With `--video` + `--subtitles`: summaries also reference *what is said* during the event.
- Without `--subtitles`: summaries describe appearance only, not dialogue content.

**Summaries are validated against a filler-pattern list before use.** Conversational filler
("Let me know if you need help", bare `---` dividers) is replaced with the template fallback.

> **VLM Hallucination Warning:** Summary generation uses a small open-weight VLM
> (FastVLM 0.5B or Gemma-4-E2B) and can occasionally hallucinate or miss salient
> objects, especially in complex or low-light frames. This is a known limitation of
> small generative models, not a bug that can be fully eliminated by prompting alone.
> **It is not a substitute for human review in safety- or compliance-critical
> applications.** See DECISIONS.md §20.4 for the anti-hallucination prompt rationale
> and a before/after comparison log.
