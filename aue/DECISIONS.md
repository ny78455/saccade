# DECISIONS.md
# Engineering Assumptions & Design Decisions — AUE Module 3

This file documents every assumption, stub, heuristic, and design decision
made during AUE (Audio Understanding Engine) Module 3 implementation.
Required by the engineering contract §0 Role Instruction.

---

## §AUE-1. INPUT GAP — AUE DECODES ITS OWN AUDIO

**Issue:** The source spec (§1.1) assumes Module 1 (ASVL) streams an
`AudioPacket(timestamp, waveform, sample_rate, duration, audio_energy, spectral_features)`
to AUE. ASVL does NOT produce this stream.

**Reality:** ASVL (asvl/asvl/audio.py) decodes the full audio track once
internally via `_extract_audio_numpy()` and exposes only scalar `AudioFeatures`
(energy, MFCC mean, spectral_flux, speech_prob) per 250ms window via
`build_audio_index()` / `get_audio_at()`. No raw waveform is streamed downstream.

**Resolution:** AUE decodes and preloads the full movie audio track itself,
directly from the source video file, exactly like ASVL already does internally:

```python
# aue/aue/audio_source.py::preload_audio()
y, sr = librosa.load(video_path, sr=16000, mono=True)
```

Rationale:
- ASVL's `audio_energy` per window is still used as a cheap cross-check
  where available (passed via FramePacket.audio_energy from ASVL output).
- AUE cannot reuse ASVL's internal waveform array without coupling the
  modules at the memory level — architecturally undesirable.
- Decoding the audio track twice (once by ASVL, once by AUE) is an
  acceptable one-time cost vs. architectural coupling.

**Key rule:** `preload_audio()` is called EXACTLY ONCE per AUE run, at the
top of `pipeline.py::run()`. Every downstream module slices from the
returned numpy array. This is the same discipline as ASVL's own audio preload.

---

## §AUE-2. WHISPER MODEL SIZE — "small" CHOSEN

**Decision:** `whisper_model_size = "small"` is the default.

**Rationale:**
- `base` (74M params): fastest, ~5% higher WER on movie dialogue vs `small`.
- `small` (244M params): ~4× faster than `medium` on CPU/GPU, WER acceptable
  for movie dialogue (primarily English or consistent-language films).
- `medium` (769M params): higher accuracy but RTF often > 1 on CPU.
- `large-v3` (1.5B params): best accuracy, RTF >> 1 on CPU without GPU.

Users needing higher accuracy for multilingual or accented speech should
override via `--whisper-model medium` or `--whisper-model large-v3`.

**RTF benchmark (PLACEHOLDER — fill in after measuring on comedy.mp4):**

| Model | audio duration | processing time | RTF |
|-------|---------------|----------------|-----|
| base  | — s           | — s            | —   |
| small | — s           | — s            | —   |

---

## §AUE-3. LIGHTWEIGHT DIARIZATION VS. FULL PYANNOTE

**Decision:** V1 uses `resemblyzer` (GE2E embeddings, ~17MB) + online exemplar-gallery
clustering instead of the full `pyannote.audio` pipeline.

**Rationale:**
- pyannote.audio offline resegmentation pipeline: >500MB, requires global context
  before producing speaker labels, RTF often >> 1 on CPU.
- resemblyzer: ~17MB, runs per-segment, compatible with rolling-buffer streaming.
- Exemplar-gallery clustering (from AESE §21.3): immune to centroid drift.

**Expected DER tradeoff (honest):**
- resemblyzer + exemplar gallery: expected DER ~25-35% on multi-speaker movie audio.
- Full pyannote: expected DER ~12-18% on similar content.
- V1 accepts this tradeoff. Run `eval/run_eval.py` to measure actual DER.

**Fallback:** If resemblyzer is unavailable (webrtcvad build failure on Windows),
MFCC mean vectors via librosa are used. MFCC-based speaker embeddings are
less discriminative — DER degrades further (estimated ~40-50%).

---

## §AUE-4. SPEAKER↔CHARACTER BINDING — REUSES AESE's CHARTERNAMEIBINDER PATTERN

**Decision:** `SpeakerCharacterBinder` is a direct port of AESE's `CharacterNameBinder`
(aese/adapters/character_naming.py) rather than a new mechanism.

**Rationale:** The underlying problem is identical:
- CharacterNameBinder: anonymous cluster ID → named entity, via subtitle vocatives
- SpeakerCharacterBinder: anonymous speaker cluster → visual character, via temporal overlap

Building a second, subtly different mechanism for the same problem class creates
inconsistency and doubles maintenance burden. Direct reuse/port is the correct choice.

**Single-subject evidence discipline:**
- Bind only when EXACTLY ONE character is visible in the concurrent AESE Event.
- Conflicting votes → unresolved (an unresolved SPK_01 is correct output).
- min_votes_to_resolve=2 (stricter than AESE's 1, because audio-visual temporal
  overlap is a noisier signal than subtitle vocative naming).

**V1 scope boundary (lip-sync association):**
Lip-movement-based speaker attribution is OUT OF SCOPE. Requires a dedicated
lip-sync model and per-frame face-mouth tracking — a substantial separate project.
V1 uses temporal overlap + single-visible-character evidence only.

---

## §AUE-5. SHARED PANNS MODEL — ONE CNN FOR MUSIC DETECTION + SED

**Decision:** A single PANNs CNN14 model (AudioSet 527 classes) is used for
both music detection and sound-event detection.

**Rationale:** Both tasks overlap significantly on AudioSet. Loading two separate
models for overlapping functionality doubles memory and startup latency. The shared
model is loaded once via `music/music_detection.py::get_panns_model()` and
imported by `sound_events/sed_model.py`.

**PANNs model size:** ~500MB checkpoint on first download to `~/.panns_data/`.
Override via `panns_checkpoint_dir` in config.

**Fallback:** If panns-inference is unavailable, a heuristic based on spectral
centroid + ZCR + RMS energy is used for both music detection and SED. This
degrades accuracy significantly (estimated F1 reduction ~40%) but does not crash.

---

## §AUE-6. MUSIC EMOTION — V1 HEURISTIC MAPPING

**Decision:** V1 uses a tempo + energy + chroma_mode → emotion score heuristic.
A pretrained music-emotion model (musicnn, Music Transformer) is future work.

**Why heuristic:**
- No widely-available pretrained music-emotion model with a permissive license
  installs via a single pip command with the other AUE dependencies.
- The heuristic is transparent, debuggable, and consistently returns
  `dict[str, float]` — the contract is met even if accuracy is limited.

**V1 accuracy expectation:** ~60-70% on obvious categories (action, calm).
More nuanced categories (romantic, melancholic) are approximated from tempo +
key mode — not reliable for fine-grained analysis.

**Output contract (NEVER RELAXED):**
- Always returns `dict[str, float]`.
- Scores are NOT required to sum to 1.0 (multi-label, not softmax).
- All 7 labels always present: action, suspense, calm, melancholic, joyful, tense, romantic.

---

## §AUE-7. ACOUSTIC EMOTION — V1 HEURISTIC MAPPING

**Decision:** V1 uses pitch + energy + pause + speaking rate → emotion heuristic.
A pretrained speech emotion recognition (SER) model (wav2vec, SpeechBrain) is future work.

**Independence from transcript (enforced design constraint):**
The `transcript` parameter of `compute_acoustic_emotion()` is intentionally
ignored. Acoustic emotion must estimate HOW something was said (vocal delivery),
not WHAT was said (semantic content). The independence is verified by tests.

**V1 accuracy:** ~50-60% on clear emotional speech. Subtle emotions ambiguous.

---

## §AUE-8. RTF BENCHMARK (PLACEHOLDER — FILL IN AFTER MEASUREMENT)

**Instruction:** Run the CLI on comedy.mp4 and record results here.

```
python cli.py --input ../comedy.mp4 --aese-events ../aese/events.jsonl \
              --output audio_events.jsonl --whisper-model small
```

**Results (PLACEHOLDER — run and fill in):**

| Component       | Time (s) | % of total |
|----------------|---------|-----------|
| Audio decode    | —       | —         |
| VAD             | —       | —         |
| ASR (Whisper)   | —       | —         |
| Diarization     | —       | —         |
| Music detection | —       | —         |
| SED             | —       | —         |
| Acoustic emotion| —       | —         |
| Fusion          | —       | —         |
| **Total**       | —       | **100%**  |
| Audio duration  | —       | —         |
| **RTF**         | **—**   | —         |

**RTF < 1 target:** PLACEHOLDER — verify after running on real clip.

**Known bottleneck:** Whisper ASR is expected to dominate. If RTF > 1,
the primary mitigation is: (1) reduce to `--whisper-model base`, (2) enable
GPU inference, (3) reduce rolling buffer size to reduce per-buffer ASR overhead.

---

## §AUE-9. WEIGHTS SUM — VERIFIED CORRECT

**The source spec's AUE importance_weights sum to exactly 1.0:**
speech(0.30) + music(0.20) + sound_event(0.20) + dialogue_semantic(0.15) + context(0.15) = 1.00 ✓

No renormalization needed (unlike AESE which had a 1.05 bug in §1).
The acceptance test `_assert_importance_weights_sum(AUEConfig())` still runs
at import time in `types.py` to guard against future edits.

---

## §AUE-10. FUSION LAYER — V1 TEMPORAL MEAN POOLING

**Decision:** Fusion uses temporal mean pooling of concatenated per-modality
feature vectors (128-dim output).

**Rationale:** Mirrors AESE's `pool_event_embedding()` approach. The honest V1
baseline before attention-weighted or learned fusion is validated.

**Future work:** Attention-weighted pooling or a small learned MLP fusion network
trained on ground-truth audio-visual alignment.
