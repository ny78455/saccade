# DECISIONS.md
# Engineering Assumptions & Design Decisions — AESE Module 2

This file documents every assumption, stub, heuristic, and design decision
made during AESE (Adaptive Event Segmentation Engine) Module 2 implementation.
Required by the engineering contract §0 Role Instruction.

---

## 1. WEIGHTS SUM TO 1.05 — SPEC BUG CAUGHT AND RENORMALIZED

**Issue:** The source spec §26 defines boundary signal weights:
```
prediction_error: 0.25, scene: 0.20, dialogue: 0.15, emotion: 0.10,
character: 0.15, embedding: 0.15, music: 0.05
Sum = 1.05 ← BUG
```

**Fix:** Renormalized by dividing each weight by 1.05 so they sum to exactly 1.0:
```
prediction_error: 0.238095, scene: 0.190476, dialogue: 0.142857,
emotion: 0.095238, character: 0.142857, embedding: 0.142857, music: 0.047619
Sum = 1.000000 ✓
```

**Verification:** `_assert_weights_sum(AESEConfig())` runs at module import time in
`types.py`. An AssertionError is raised if the sum drifts from 1.0 by more than 1e-4.
Tests in `test_fusion.py::test_weights_sum_to_one` assert this explicitly.

---

## 2. CLIP Model — Real Implementation with Hash/Histogram Fallback

**Decision:** `adapters/embedding.py` uses OpenCLIP (ViT-B/32, frozen, off-the-shelf).
- Image embedding: CLIP image encoder → shape (512,)
- Text embedding: CLIP text encoder → shape (512,)
- Fusion: "concat" (default) → (1024,); "mean" → (512,)

**Fallback (# STUB):** If `open-clip-torch` is unavailable or the model download
fails, the embedding falls back to:
- Perceptual hash (8×8=64-D) + color histogram (32 bins × 3 channels = 96-D) = 160-D
- Concat fusion yields 320-D in fallback mode
- This is logged as a WARNING at startup and marked `# STUB` in the code
- The fallback is NOT a semantic embedding; downstream event coherence will be degraded

**Impact:** Build never blocks on model availability. CLIP quality is not assumed.

---

## 3. SCENE LABEL — STUB (Zero-shot CLIP, ~60-70% accuracy expected)

**Decision:** `adapters/scene_label.py` uses zero-shot CLIP text-image similarity
against a fixed 12-label set: indoor, outdoor, vehicle interior, street, nature,
building exterior, office, restaurant, kitchen, bedroom, nighttime, unknown.

**NOT:** A real scene-graph model, VQA model, or places-365 classifier.

**Precision expectation:** ~60-70% on common movie scenes. Not reliable enough
for fine-grained location understanding. Used only for coarse scene-change detection
at the boundary signal level.

**Further fallback:** If CLIP is unavailable, uses a color-temperature heuristic
(blue dominance → outdoor; warm dominance → indoor). Even less reliable.

---

## 4. CHARACTER DETECTION — STUB (Face count only, no identity)

**Decision:** `adapters/character_stub.py` counts detected faces using:
1. OpenCV DNN SSD ResNet10 (if model files present in `models/` dir)
2. Haar cascade frontal face (bundled with OpenCV, less accurate)
3. Returns 0 if neither detector is available

**NOT implemented:** Character re-identification, face tracking, name assignment,
multi-angle recognition. These are explicitly out of scope (§1.2).

**Future work:** Replace with a face-tracking + re-ID pipeline (e.g. DeepFace,
InsightFace, or a custom ArcFace embedding tracker).

---

## 5. ACTION LABEL — STUB (3-bucket threshold on motion_score)

**Decision:** `adapters/action_stub.py` buckets Module 1's `motion_score` into:
- `motion_score < 0.2` → "static"
- `motion_score < 0.5` → "walking"
- `else`               → "fast_action"

**NOT:** A real action recognition model (no optical flow classification,
no pose estimation, no temporal convolutions).

**Future work:** Replace with SlowFast, X3D, or a lightweight MobileNet-based
action classifier.

---

## 6. SPECTRAL FLUX — STUB (Audio energy delta, not real spectral analysis)

**Decision:** Module 1 does not expose spectral flux. `adapters/music_mood.py`
estimates it as `|curr_audio_energy - prev_audio_energy|`.

**NOT:** True spectral flux (requires FFT of raw audio frames across consecutive windows).

**Impact:** Music mood bucketing (calm/tense/energetic) is coarser than it would be
with real spectral analysis. False "tense" labels possible during any energy transition.

---

## 7. TEMPORAL CONTEXT BUFFER — 45s (vs Module 1's 10s)

**Decision:** `context_buffer.py` defaults to 45 seconds, larger than Module 1's 10s.

**Rationale:** Event coherence requires more temporal context than adaptive frame
sampling decisions. A 10s window is often smaller than a single conversation beat;
a 45s window can hold a complete short scene (dialogue → action → resolution).

Module 1's 10s buffer is sized for frame-level sampling decisions (is this frame
important relative to the last 10 seconds?). Module 2's 45s buffer is sized for
semantic event decisions (does this embedding break the pattern of the last 45 seconds?).

---

## 8. EMOTION SIGNAL — INTENTIONAL ZERO (No emotion model in scope)

**Decision:** `boundary/signals.py::emotion_signal()` ALWAYS returns `0.0`.

**Rationale:** Module 1 provides no emotion-related signal, and no emotion model
is in scope for Module 2 (§1.2). Returning a fabricated non-zero value (e.g. using
audio energy as an emotion proxy) would:
1. Corrupt every downstream boundary decision with a signal having no semantic
   relationship to actual emotion transitions
2. Silently make the system appear to have emotion detection capability it doesn't have

**Documented zero:** This is an intentional, documented stub — not a missing feature.
The emotion weight (0.095238) in `AESEConfig.weights` is effectively wasted, since
the signal is always 0.0. This is acceptable for V1; the weight exists for when a
real emotion model is added.

**Future work:** Replace with a lightweight valence/arousal model (e.g. fine-tuned
CLIP on FER-2013/AffectNet, or a fine-tuned wav2vec on emotional speech).

---

## 9. PREDICTION ERROR MODEL — V1 LINEAR EXTRAPOLATION (Not a transformer)

**Decision:** `boundary/prediction_error.py` uses linear extrapolation from the
last 2 embeddings to predict the next, then measures cosine distance between
predicted and actual.

**NOT:** A trained temporal transformer. The contract §5.5 explicitly states:
"Build the simple version first — NOT a trained transformer — that's future work."

**Extension point:** A clear `# TODO: replace with trained temporal transformer per
Section 14` comment is placed in `prediction_error.py`.

---

## 10. EVENT GRAPH 'causes' EDGES — NOT IMPLEMENTED

**Decision:** `event_graph.py` implements `before` edges only (temporal order).
`causes` edges always return an empty list. `EventNode` has no `causes` field.

**Rationale:** Causal inference between events requires LLM reasoning or a trained
causal model — both are explicitly out of scope for Module 2 (§5.15).

**Future work:** Replace with an LLM-based causal reasoning pass (GPT-4o, Gemini)
or a trained entailment classifier over event embeddings.

---

## 11. EVENT SUMMARY — TEMPLATE-BASED (Not LLM-generated)

**Decision:** `event_constructor.py::_make_summary()` produces template strings:
```
"{action_display} event in {scene_label}, {n} people present"
```

**NOT:** LLM-generated prose. No LLM API calls are made in this module.

**Rationale:** The source spec §5.9 explicitly states: "No LLM call is in scope
for this module — implement summary as a template-based string."

**Future work:** Add an LLM summarization step downstream that takes the event
embedding + temporal features and generates natural language descriptions.

---

## 12. IMAGE NOT AVAILABLE IN MANIFEST-REPLAY MODE

**Decision:** `manifest.jsonl` (Module 1's output) stores metadata only — no pixel
data. When running `cli.py --input manifest.jsonl` without `--video`, all
image-dependent adapters (embedding, scene label, character count) receive a
black placeholder frame `(64×64×3, all zeros)`.

**Impact:**
- Embeddings in replay mode are the stub fallback (hash + histogram of black frame)
- Scene labels default to "unknown"
- Character count returns 0
- Camera cues derived from `scene_change` flag are still accurate (not image-dependent)

**Mitigation:** Use `--video comedy.mp4` flag to load real frames from the video.

---

## 13. MUSIC MOOD LABELING — HEURISTIC (3-bucket energy/flux threshold)

**Decision:** `adapters/music_mood.py` uses:
- `audio_energy ≥ 0.25` → "energetic"
- `audio_energy < 0.08 AND spectral_flux < 0.08` → "calm"
- `else` → "tense"

**NOT:** A music genre or mood classifier (no trained model).

**Future work:** Replace with musicnn, MusicCNN, or fine-tuned YAMNet.

---

## 14. `characters` → `character_count_range` — BREAKING SCHEMA CHANGE

**Date:** 2026-08-17

**Decision:** The `Event.characters` field was renamed to `character_count_range`; a new
`max_characters_seen` field was added. Both the Python dataclass and the JSON output format
changed simultaneously.

**Rationale:** `characters: [0, 1, 2]` was misread by a reviewer as "3 identified entities."
The field has never done entity identification — it stores the *sorted unique set of per-second
face counts* observed during the event. Renaming to `character_count_range` makes this
unambiguous. `max_characters_seen` provides a single headline number for display.

**Affected files:** `types.py`, `event_constructor.py`, `event_split.py`, `event_merge.py`,
`cli.py`, `README.md`, all downstream test files.

**Downstream impact:** Any consumer of `events.jsonl` that reads the `characters` key will
break after this change. Update consumers to use `character_count_range` and
`max_characters_seen`.

**AESE does NOT perform character identification or re-identification.** See §4 for the full
character detection stub inventory.

---

## 15. HARD-TRIGGER BOUNDARY LAYER — Deterministic override for unambiguous cues

**Date:** 2026-08-17

**Decision:** Two hard-trigger checks were added to `boundary/candidate_detector.py`,
evaluated before the weighted-fusion path:

1. **Camera cut** (`camera_cue == "cut"`): returns `is_boundary=True, confidence=0.95`.
   A Module-1-confirmed hard cut is near-deterministic boundary evidence and must not be
   diluted by an otherwise-quiet clip.

2. **Sustained action transition**: returns `is_boundary=True, confidence=0.85` when the
   last 3 features in the context buffer show [non-action, fast_action, fast_action].
   Requires 2 consecutive fast_action seconds to avoid triggering on a single noisy
   motion spike (e.g. a hard camera shake).

**Not removed:** The weighted-fusion path still runs for ambiguous cases (gradual emotional
shifts, topic changes, music transitions) that have no single deterministic cue.

**Rationale:** The weighted sum is architecturally correct for ambiguous signals but wrong
for unambiguous ones. A genuine hard cut should not need to out-vote a quiet clip via score
dilution.

---

## 16. CHARACTER COUNTING — GENERATIVE VLM REMOVED FROM HOT PATH

**Date:** 2026-08-18

**Decision:** `adapters/character_stub.py::count_characters()` is now OpenCV-only.
The FastVLM `count_people()` call (V2 primary path) was removed.

**Root cause of regression:** `count_people()` issued a free-text prompt ("How many people
are in this image?") and regex-parsed the response for an integer. When the model returned
conversational filler ("Let me know if you need anything else."), the regex found no integer
and the function silently returned 0. This caused `max_characters_seen=0` for every event
across an entire run.

**Architectural principle violated:** Character counting is a classification/detection task
(input: image → output: integer), not a generation task. Generative free-text output must
never be the sole source of a numeric field. The VLM was asked to *describe* the image
and then its description was parsed for numbers — this conflates generation with classification
and makes the result fragile to any phrasing variation.

**Retained:** `fastvlm.count_people()` still exists as a standalone helper. It is not deleted
and may be reintroduced later behind its own structured-output contract (e.g. constrained
decoding or a classifier head), not as a free-text parse.

**Detector chain after fix:**
  1. OpenCV FaceDetectorYN / YuNet ONNX (OpenCV 5+)
  2. OpenCV DNN SSD ResNet10 (OpenCV 4)
  3. OpenCV Haar cascade frontal + profile (OpenCV 4)
  4. 0 — last resort

**Regression test:** `tests/test_character_detection_regression.py::test_count_characters_never_calls_generative_vlm`
patches `fastvlm.count_people` to raise if called and asserts `count_characters()` runs cleanly.

---

## 17. GENERATIVE SUMMARY — MOVED TO POST-FINALIZATION, OFF THE HOT PATH

**Date:** 2026-08-18

**Decision:** `event_constructor._close_event()` no longer calls any generative model.
The VLM summary call was extracted to `aese/summary.py::generate_summary()`, which is
called **once per finalized event** in `pipeline.py::_finalize_event()`, after
`OnlineMerger`, `EventClassifier`, and contiguous-ID assignment have all completed.

**Root cause of runtime regression:** `_make_vlm_or_template_summary()` was called inside
`_close_event()`, which fires on every candidate event boundary (~81 times for an 81s clip).
Each VLM call takes ~5-8 seconds, producing a ~12-minute runtime for an 81-second clip
(~90× overhead). The `<100ms/decision` requirement in §8 was violated by construction.

**New architecture:**
```
[Per-second hot path, <100ms]
  aggregator → boundary detection → event construction → merge → classify
                                    ↑ NO VLM CALLS HERE ↑

[Post-finalization, once per event, off the hot path]
  _finalize_event() → generate_summary() → yield event
```

**VLM call count improvement:** From ~81 calls (per second) to ~8 calls (per finalized event)
for the 81s fight clip — a ~10× reduction, and the primary fix for the runtime regression.

**Output validation in summary.py:** `_validate_or_fallback()` gates every VLM response:
  - Length < 5 chars → template fallback
  - Filler pattern match (6 compiled regexes) → template fallback
  - More than 1 newline → template fallback
  - Any exception → template fallback

`build_template_summary()` (from `event_constructor.py`) is the permanent safety net.
It is never deprecated, never optional, always called first to produce the fallback.

**Regression tests:**
  - `test_regression_fight_clip.py::test_no_conversational_filler_in_any_summary`
  - `test_regression_fight_clip.py::test_generative_summary_call_count_matches_event_count`
  - `test_performance_gates.py::test_per_second_decision_latency` (p95 < 100ms)
  - `test_performance_gates.py::test_full_pipeline_runtime_budget` (< 90s for 81s clip)

---

## 18. GEMMA-4 INFERENCE EFFICIENCY — Call-Frequency Fix + Per-Call Optimisations

**Date:** 2026-08-24

---

### Fix 18.1 — Scene label called once per second, not once per raw frame (`aggregator.py`)

**Root cause (confirmed bug, line 192 before fix):**
```python
# BEFORE — per-packet list comprehension: N VLM calls/second at N fps
scene_labels = [label_scene(p.image) for p in real_image_packets]
scene_label = _majority_vote(scene_labels) if scene_labels else "unknown"
```
During 5–10 fps action segments, `real_image_packets` contained 5–10 elements.
With the `gemma4` backend active, each `label_scene()` call triggers a full
generative forward pass.  This inflated call volume by up to 10× per second.

**Fix:** Select the middle packet as the representative frame and call
`label_scene()` exactly once:
```python
# AFTER — exactly 1 VLM call per second
_rep_packet = real_image_packets[len(real_image_packets) // 2]
scene_label = label_scene(_rep_packet.image) if _rep_packet.image is not None else "unknown"
```
**Quality tradeoff:** None.  Scene content does not change meaningfully within a
single one-second window.  Majority-voting N generative calls that all see the
same scene is pure redundancy.

**Regression test:** `tests/test_gemma4_speed_regression.py::test_scene_label_called_once_per_second_not_per_frame`

---

### Fix 18.2 — Merged scene-label + caption call (`gemma4.py`)

`describe_scene_and_caption()` encodes the image through the vision tower once
and extracts both the scene label and the event caption in a single generation
call, using a structured `SCENE: / CAPTION:` output format.  This halves the
image-encoding cost per event compared to calling `describe_scene()` +
`caption_event()` separately.

`vlm_router.describe_scene_and_caption()` routes to the native combined path
on `gemma4` and falls back to two sequential calls on `fastvlm` / `yunet`,
preserving the identical `(str, str)` return type for all callers.

**Regression tests:**
- `test_combined_scene_and_caption_parses_correctly` — happy-path parse
- `test_combined_scene_and_caption_malformed_falls_back` — missing fields degrade to `("unknown", "")` without raising
- `test_combined_scene_label_not_in_vocabulary_falls_back_to_unknown` — hallucinated labels rejected

---

### Fix 18.3 — Keyframe downscaling before inference (`gemma4.py::_prepare_image`)

**Decision:** Keyframes are resized to a maximum longest dimension of 512px
(`_MAX_DIM = 512`) using LANCZOS resampling before being passed to the Gemma-4
vision tower.

**Rationale:** Vision-language models internally tile or patch high-resolution
inputs.  A 1920×1080 frame produces ~4× more image tokens than a 512×288 frame.
Beyond the model's effective working resolution, additional pixels increase
prefill compute without improving caption quality on typical movie-frame content.
512px is Gemma-4-E2B's documented recommended input size.

**REQUIRED BEFORE PRODUCTION — quality sign-off:**
Run a side-by-side caption comparison on at least 5 representative keyframes
(one each from dialogue, action, transition, close-up, and low-light scenes)
at original resolution vs 512px.  Record results here before enabling this
setting in a production pipeline.  If any meaningful quality regression is
observed, increase `_MAX_DIM` or add a per-backend override.

**Regression tests:** `test_prepare_image_downscales_large_frame`,
`test_prepare_image_preserves_aspect_ratio`, `test_prepare_image_leaves_small_frame_unchanged`

---

### Fix 18.4 — Explicit dtype and attention implementation (`gemma4.py::_ensure_loaded`)

**dtype:** Replaced `dtype="auto"` with explicit `torch.bfloat16` (CUDA) /
`torch.float32` (CPU).  `bfloat16` is native on Ampere and newer GPUs, avoids
`fp16` overflow risk on activations, and removes the ambiguity of `"auto"`.

**Attention implementation:** Set `attn_implementation="sdpa"` (PyTorch Scaled
Dot-Product Attention, available since PyTorch 2.0).  SDPA fuses QKV
projections and is meaningfully faster than eager attention on modern hardware.
Falls back to `"eager"` with a logged warning if the installed `transformers`
version predates `attn_implementation` support (pre-4.36).

**CPU-only honesty note:** If `torch.cuda.is_available()` returns `False`, AESE
logs a WARNING at startup:
> "Running a 2B multimodal model on CPU will be significantly slower than GPU
> regardless of efficiency fixes.  Consider `--vlm fastvlm` as a CPU-friendly
> alternative."
These inference-efficiency fixes (dtype, SDPA, downscaling) reduce per-call
overhead but cannot overcome the fundamental throughput gap between CPU and GPU
for a 2B-parameter multimodal model.  Do not interpret these fixes as a promise
of CPU parity.

---

## 19. FP16 WEIGHT LOADING — Gemma-4 and FastVLM (AESE Contract §19)

**Date:** 2026-09-02

---

### 19.1 — Shared `_select_dtype()` utility (`aese/adapters/_dtype_utils.py`)

**Decision:** The inline `torch.float16 if torch.cuda.is_available() else torch.float32`
expression that existed independently in `fastvlm.py` has been extracted into a shared
`_select_dtype()` function in a new `_dtype_utils.py` module.  Both `gemma4.py` and
`fastvlm.py` now import and call this function.

**Rule:**
- `torch.cuda.is_available()` returns `True` → `torch.float16`
- Otherwise (CPU, MPS, no torch) → `torch.float32`

**Why NOT `float16` on CPU:**
Most CPU kernels either lack optimised fp16 implementations or internally upcast to
fp32 anyway, paying the conversion cost for no speed benefit.  Forcing fp16 on CPU
is the single most common way this kind of change makes things *slower*, not faster.
Apple Silicon (MPS) has partial fp16 support but has not been verified for this model
class — treated conservatively as fp32.

**Why NOT force `bfloat16` here:**
`bfloat16` has the same exponent range as fp32 and is therefore more stable than fp16
on activations with large magnitudes (no overflow risk).  Gemma-4 was previously loaded
in `bfloat16` (§18.4).  This contract explicitly requests `float16` for the measured
GPU throughput benefit; the degenerate-output guard (§19.2) catches the rare NaN/inf
case rather than letting it pass through silently.

**Regression tests:**
- `test_select_dtype_returns_float16_on_cuda` — mock CUDA=True → assert `torch.float16`
- `test_select_dtype_returns_float32_on_cpu` — mock CUDA=False → assert `torch.float32`
- `test_select_dtype_not_float16_on_cpu` — explicit negative guard

---

### 19.2 — Degenerate-output guard in `gemma4._ask()` (`gemma4.py`)

**Defect risk:** fp16's reduced exponent range (max ~65,504 vs fp32's ~3.4×10³⁸) can
occasionally produce `NaN`/`inf` in logits for certain inputs — particularly long
contexts or unusual image content.  This does not crash; it silently produces an empty
or single-character decode.

**Fix:** Added a post-decode length check in `_ask()`:
```python
if not raw or len(raw) < 2:
    logger.debug("AESE Gemma-4: suspiciously short/empty output ...")
    return ""
```
Returning `""` routes the call through the exact same fallback path (`"" → template
summary`) that already handles every other VLM failure — no new failure mode, no
special-casing needed in callers.

**Regression tests:**
- `test_ask_degenerate_empty_output_returns_empty_string` — empty decode → `""`
- `test_ask_single_char_output_returns_empty_string` — single-char decode → `""`
- `test_ask_valid_output_passes_through` — valid caption is NOT swallowed by guard

---

### 19.3 — Before/after benchmark (`eval/benchmark_fp16.py`)

**Required before shipping to production:**  Run `python eval/benchmark_fp16.py` on
GPU hardware and record results here.  A precision change needs measured evidence.

```
python eval/benchmark_fp16.py --iterations 5 --output eval/benchmark_fp16_results.json
```

**Results (PLACEHOLDER — fill in after running on GPU):**

| Metric | fp32 baseline | fp16 |
|--------|--------------|------|
| GPU | — | — |
| Inference (5 frames) | — s | — s |
| ms / frame | — | — |
| Speedup | | **—x** |

**Caption quality comparison (PLACEHOLDER — fill in after running):**

> Record the side-by-side captions printed by the benchmark here.
> If fp16 captions are meaningfully degraded vs fp32, revert `gemma4.py` to
> `bfloat16` and update this section with the rationale.

---

### 19.4 — CPU-only honesty note

If `torch.cuda.is_available()` returns `False` at runtime:
- `_select_dtype()` returns `torch.float32` — no fp16 loading, no benefit attempted.
- `_ensure_loaded()` logs a WARNING recommending `--vlm fastvlm` as a CPU-friendly
  alternative (FastVLM-0.5B vs Gemma-4-E2B has ~4× fewer parameters).
- If CPU speed is the real constraint, **int4/int8 quantization via `bitsandbytes`**
  is the correct next step.  That is a different, larger technique with different
  tradeoffs and is explicitly out of scope for this contract.

---

## 20. PERCEPTION FIXES — Salience Keyframe, End-Card Detection, Anti-Hallucination, Caption Reconciliation

**Date:** 2026-09-03

Four real perception failures identified on the Andhadhun clip (162s, 11 events).
Each fixed with the minimum surgical change, no regression to call-frequency work.

---

### 20.1 — Salience-Based Keyframe Selection (`keyframe.py`)

**Root cause:** `select_keyframe(strategy="lowest_blur")` was the default for the
summary/caption pipeline.  "Lowest blur" = lowest motion_score = the calmest frame.
For a 26s event containing a dramatic reveal at second 20, the calmest frame is
precisely the wrong one to caption.

**Fix:** New `"most_salient"` strategy selects the frame with the highest
`motion_score + novelty_score` — the peak of narrative activity, not of visual
stillness.  Changed to the new default in `pipeline.py::EventConstructor` init.
`"lowest_blur"` is kept as a named option for thumbnail generation where visual
clarity matters more than narrative content.

**Quality tradeoff:** High-motion frames may have motion blur, producing slightly
less sharp VLM input than a fully static frame would.  This is acceptable because
narrative accuracy (what is happening) matters more for event summaries than
photographic clarity.

**Regression tests:** `test_salient_keyframe_picks_motion_novelty_peak`,
`test_salient_keyframe_not_lowest_blur`, `test_default_strategy_is_most_salient`

---

### 20.2 — Gated Secondary Keyframe for Long Events (`keyframe.py`, `pipeline.py`)

**Root cause:** Even with salience-based primary selection, a single frame cannot
cover all meaningful moments in a long event (26s in the Andhadhun clip: a calm
first half followed by a dramatic reveal).

**Fix:** `needs_secondary_frame(features, primary_idx, duration_s)` returns a
secondary index when ALL of:
  1. `duration_s > 15.0` — event is long enough to plausibly have multiple moments
  2. Salience of candidate > `0.7` — high in absolute terms, not just locally higher
  3. Candidate timestamp > `8 s` away from primary — plausibly a different scene moment

When all three fire, `_finalize_event()` issues ONE additional `caption_frame_delta()`
call and appends the one-sentence addendum to the primary summary.

**Speed protection:** This gate is designed to fire on ≪ 10% of events.  For the
81s fight clip used in §17's speed regression, all events are ≤ 26s and most have
low overall salience — the gate fires only when there is real, distant, high-energy
content the primary frame missed.  The recent per-event call-frequency fix (§18)
is not regressed by this change.

**Regression tests:** `test_needs_secondary_long_event_with_spike`,
`test_needs_secondary_short_event_returns_none`,
`test_needs_secondary_no_qualifying_distant_spike_returns_none`,
`test_needs_secondary_below_salience_threshold_returns_none`

---

### 20.3 — Deterministic Graphics/End-Card Detection (`scene_label.py`)

**Root cause:** Event 10 (162–175s, solid black frame with red Netflix 'N' logo)
was classified as `"office"` by the heuristic fallback in `_heuristic_scene_label()`.
The VLM path was not active in this run, and the color-temperature heuristic
defaults to `"office"` for any warm/neutral frame it can't classify as outdoor.

**Fix:** `is_graphics_or_endcard(image)` runs before any VLM or CLIP call:
```
color_std  = np.std(image)        # variance across all pixels and channels
edge_density = count_nonzero(Canny(gray)) / total_pixels
if color_std < 18.0 and edge_density < 0.015: return "graphics/end card"
```
Returns `"graphics/end card"` — NOT added to `SCENE_LABELS` since it is a pre-check
bypass, not a model classification.  The existing `image.max() < 5` black-frame guard
runs first (returns `"unknown"` for pure fade-to-black, not end-card).

**Threshold rationale:**
- `color_std < 18.0`: A real filmed scene with any texture/colour variation exceeds
  this easily. A solid-background title card does not.
- `edge_density < 0.015`: A real scene with actors and furniture has many edges.
  A logo on a plain background has a low fraction of edge pixels across the full frame.

**REQUIRED before production:** Validate both thresholds against real dark-but-detailed
film frames (night scenes, dimly lit interiors, silhouette shots).  A genuinely dark
dramatic scene should have retained edge structure (actor silhouettes, props) giving
`edge_density` comfortably above `0.015`, even if `color_std` is low.

**Regression tests:** `test_endcard_detected_without_vlm_call`,
`test_dark_film_frame_not_classified_as_endcard`, `test_pure_black_frame_not_endcard`

---

### 20.4 — Anti-Hallucination + Anomaly-Attention Prompt (`summary.py`)

**Root cause:** Event 4 summary contained "A woman is standing near the table,
gesturing while two other people are seated" when the actual scene contained a
body and blood (confirmed in the ground truth).  The previous `SUMMARY_SYSTEM_PROMPT`
instructed the model to describe the scene but did not explicitly forbid
prop-based inference or require attention to anomalies.

**Fix:** `SUMMARY_SYSTEM_PROMPT` updated with:
- Explicit prohibition on prop-based assumption ("do not assume people are eating
  just because a dining table is present")
- Explicit anomaly-attention directive ("pay close attention to anything unusual,
  unexpected, or anomalous — items on the floor, unusual body positions, weapons")
- Epistemic humility clause ("if you are not confident about a detail, do not
  state it as fact")

**This is a mitigation, not a guarantee.** A 2B/E2B-class VLM will still
hallucinate sometimes.  The before/after on Event 4 should be re-run when a GPU
is available and recorded here.  README.md now contains an explicit disclaimer.

**Before (Event 4):** *"A woman is standing near the table, gesturing while two
other people are seated."*
**After (PLACEHOLDER — run on GPU with updated prompt and record here):**

---

### 20.5 — Caption-Text Character Count Reconciliation (`caption_person_count.py`, `pipeline.py`)

**Root cause:** Events 1, 2, 3, 6 all have `max_characters_seen=1` despite the VLM's
own generated caption describing two people ("a man who is partially visible on the
right", "A man with a beard and sunglasses is visible on the right side").  OpenCV
face detection cannot count people who are facing away, partially cropped, or viewed
from behind — a common alternating-angle pattern in dialogue.

**Fix:** `estimate_person_count_from_caption(summary_text)` parses the generated
summary for person-count signals:
  1. Explicit count words: "two people", "three individuals" → direct numeric return
  2. Person-phrase count: distinct singular mentions ("a man", "a woman", "the person")

Applied in `_finalize_event()` AFTER `generate_summary()`. Only raises
`max_characters_seen`; never lowers a face-detector-confirmed count.

**Honesty rule:** The face-detector count is a hard real observation (a face was
actually seen).  The caption estimate is a softer signal (VLM can also be wrong).
Treating the caption as a floor rather than a replacement keeps the system honest:
  - If face detector found 2 and caption says "a man" → stays 2
  - If face detector found 1 and caption says "a man and a woman" → raised to 2
  - Every raise is logged at INFO level so discrepancies remain visible

**Regression tests:** `test_caption_raises_count_when_face_detector_missed`,
`test_caption_explicit_two_people`, `test_caption_explicit_three_individuals`,
`test_caption_no_people_returns_none`, `test_caption_never_lowers_detector_count`

