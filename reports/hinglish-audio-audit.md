# TurnSense Stage 1 — Hinglish Audio & EoT-Bench Pipeline Audit

| Field | Value |
|---|---|
| Project | TurnSense — Hinglish End-of-Turn Detection Evaluation |
| Audit date | 2026-11-04 |
| Environment (verified once) | `~/miniforge3-arm64/envs/turnsense/bin/python` (Python 3.11.18, arm64) |
| FFmpeg | `/opt/homebrew/bin/ffmpeg` / `ffprobe` 8.1.2 (SoX: not installed, not needed) |
| Status | **Audit complete. No Hinglish model evaluation run. No ground-truth labels fabricated.** |

---

## 1. Environment verification (once)

- `~/miniforge3-arm64/envs/turnsense/bin/python` → `arm64`, Python 3.11.18.
- Available audit libs (present, not installed): `soundfile 0.14.0`, `librosa 0.11.0`, `numpy 1.26.4`, `pandas`, `pyarrow`, `torch 2.15.0` (verified), `onnxruntime 1.31.1` (verified), `silero-vad`, `scikit-learn` (present in env listing; import not re-probed this run).
- `whisper` / `faster-whisper`: **not present** in turnsense env → no draft ASR transcripts generated in this audit (see §4 blocker).
- `livekit.local_inference` import probed but its output was consumed by the pip probe in Stage 1; **treat as unverified this run**. The v1-mini cloud adapter is out of scope regardless (quota exhausted, user instruction: do not retry).

---

## 2. Audio file inspection results

| Check | `hinglish_conversation_001_general_16k.wav` | `hinglish_conversation_002_cricket_16k.wav` |
|---|---|---|
| Processed file exists | ✅ `data/processed/hinglish_conversation_001_general_16k.wav` (7.0 MB) | ✅ `data/processed/hinglish_conversation_002_cricket_16k.wav` (7.9 MB) |
| Decodable (FFmpeg → null) | ✅ `001_DECODE_OK` | ✅ `002_DECODE_OK` |
| Duration | **230.955 s** (3 min 50.955 s) | **260.267 s** (4 min 20.267 s) |
| Sample rate | 16 000 Hz | 16 000 Hz |
| Channels | 1 (mono) | 1 (mono) |
| Codec / depth | `pcm_s16le`, 16-bit | `pcm_s16le`, 16-bit |
| Originals (untouched, for reference) | `data/hinglish_conversation_001_general.wav` — 48 kHz, stereo, 230.955 s, 44.3 MB | `data/hinglish_conversation_002_cricket.wav` — 48 kHz, stereo, 260.267 s, 50.0 MB |
| Samples (processed) | 3 695 275 | 4 164 267 |
| Combined processed duration | **491.221 s ≈ 8 min 11 s** | — |
| Peak amplitude | 0.875 (−1.2 dBFS) | 0.836 (−1.6 dBFS) |
| RMS (overall) | 0.083 (−21.7 dBFS), median 1 s window −22.0 dB | 0.088 (−21.1 dBFS), median 1 s window −21.3 dB |
| Clipped samples (‖x‖ ≥ 0.999) | 0 (0.0000 %) | 0 (0.0000 %) |
| Exact-zero fraction | 0.26 % | 0.27 % |
| Silent frames (< −45 dBFS RMS, 100 ms windows) | 399 / 2309 (17.3 %) | 489 / 2602 (18.8 %) |
| Quiet 1 s windows (< −40 dBFS) | 6 / 230 | 12 / 260 |
| Longest sub-−45 dBFS run | 1.3 s | **4.2 s** (58.0–62.2 s) — the only ≥2 s run in either file |
| Spectral centroid (mean) | 1 647 Hz; rolloff 3 045 Hz; ZCR 0.132 | 1 783 Hz; rolloff 3 277 Hz; ZCR 0.146 |
| PYIN F0 (voiced frames) | 5 188 / 7 218 voiced, median 162 Hz, p10 115 / p90 313 Hz, range 60–399 Hz | 5 584 / 8 134 voiced, median 161 Hz, p10 117 / p90 296 Hz, range 60–399 Hz |

**File integrity:** Both processed WAVs decode cleanly, match the claimed 16 kHz / mono / PCM-16 format, and their durations match the originals to the sample (230.955 s / 260.267 s). No corruption, no truncation.

**Audio quality findings:**
- No clipping detected. Peak/RMS levels are healthy conversational levels (no evidence of abnormally low volume; median 1-second windows around −22 dBFS in both files).
- Silence proportion 17–19 % is normal for conversation with pauses. Long silences are rare: 001 has none ≥ 2 s; 002 has exactly one 4.2 s run (58.0–62.2 s) — a candidate for "recording pause / topic break" to be examined by a human, not auto-labeled as EOT.
- Pause inventory at 100 ms / −45 dBFS / ≥ 0.2 s: **001 ≈ 75 candidates; 002 ≈ 78 candidates** (see §2.1). 27 and 28 of these exceed 0.5 s respectively.
- F0 distributions are broad (p10–p90 ≈ 115–310 Hz) and **do not by themselves prove two distinct speakers**, since F0 ranges of different speakers (and of one speaker across utterances) overlap substantially. This audit makes **no claim** about speaker count from signal stats alone (see §4).

### 2.1 Pause-candidate inventory (unlabeled, NOT ground truth)

`data/annotations-hinglish/pause_candidates.csv` (153 rows: 75 + 78) contains energy-derived
candidate intervals only. Every row carries `boundary_type=AMBIGUOUS`,
`review_status=needs_review`, `speaker_id=""`, `language=uncertain`, and the note
*"energy-based candidate only; NOT a label"*. **These are review prompts, not annotations.**
They must not be used as evaluation labels.

---

## 3. Speaker & language review limitations (what this audit cannot establish)

The instruction was explicit: signal statistics cannot establish speaker audibility, language
accuracy, or genuine conversational turn boundaries. Accordingly, this audit records the
following as **unresolved pending human listening + transcription**:

1. **Whether both speakers are audible, and who speaks when.** No diarization was run; F0 spread
   is compatible with one or two speakers and proves nothing. Speaker IDs (`speaker_1` / `speaker_2`
   / `overlap` / `uncertain`) must be assigned by a human listener (or a diarization model later,
   itself human-verified). The only provenance statement available is the researcher's: two speakers
   (themselves + brother).
2. **Whether genuine Hindi–English code-switching is present, and where.** No ASR was run (Whisper
   unavailable in the turnsense env; see §1), and no listening was performed here. Language tags
   (`hindi` / `english` / `mixed` / `uncertain`) therefore remain `uncertain` in the template.
   F0/centroid/ZCR stats say nothing about language.
3. **Overlapping speech segments.** Cannot be determined from energy alone; needs listening. Should be
   marked per-candidate (`overlap` flag / `speaker_id=overlap`) during annotation.
4. **Genuine conversational EOT boundaries.** Energy dips cannot distinguish hesitation, emphasis
   pauses, floor-holding, backchannels, and true turn-yielding. This is exactly the research
   question — it must be answered by an annotator listening to + reading full context, **not** by
   a silence threshold (prohibited by the Stage 1 rules).
5. **02's 4.2 s gap (58.0–62.2 s)** and other long pauses: no label is assigned; they are flagged
   candidates only.

**Blocker for draft transcripts:** Whisper / faster-whisper is not installed in the turnsense env
and installing dependencies is restricted. A human transcript (or a user-approved ASR step) is
required before language tags and transcripts can be filled.

---

## 4. EoT-Bench compatibility findings

All findings below are from reading the vendored source in
`vendor/eot-bench/eot_harness/` (citations inline). No benchmark code was modified.

### 4.1 Dataset & prediction contracts

- **Required dataset fields** (`schemas.py:11-16`): `id, language, audio, silence_spans`
  (optional: `messages, words`). No speaker fields exist anywhere in the contract.
- **Required prediction columns** (`schemas.py:23-31`): `id, language, span_index, timestamp,
  silence_dur, p_eot, label` where `label ∈ {hold, eot}` only (`schemas.py:33-36`).
  There is **no `AMBIGUOUS` label and no speaker column** in the prediction contract.
- **Audio cells** accept either decoded `array/sampling_rate` or encoded `bytes`
  (`io.py:221-232`); stereo is mean-mixed to mono on the bytes path.

### 4.2 How HOLD vs EOT spans are constructed — the critical finding

`iter_transformed_row` (`io.py:67-103`) and `build_span_set` (same rule at `io.py:189`):

```python
label = "eot" if span_index == len(silence_spans) - 1 else "hold"   # io.py:85
```

**The framework labels the LAST silence span of every row as `eot` and all earlier spans as `hold`,
mechanically.** There is no per-span ground-truth field; the benchmark's label convention is
"final pause = EOT, mid-turn pauses = HOLD". This bakes in the exact assumption the Stage 1 brief
warns against ("a completed turn / utterance-final pause is not automatically conversational EOT";
"speaker change alone must not define EOT"). The Hindi benchmark's 400 EOT spans are one per
recording (400 recordings, exactly 400 EOT spans — verified in `span_set.parquet`) for this reason.

**Consequences for the Hinglish conversations:**
- A 231 s two-speaker conversation cannot be entered as one row: it would receive exactly **one**
  EOT label (its final pause), with every mid-conversation turn-yield silently mislabeled HOLD.
- The existing instruction "do not assume utterance-final pause is conversational EOT" is
  **directly violated by the framework's default rule**. Silently feeding the conversations in
  this way would fabricate wrong ground truth.
- **Notably**, whether one-spoken-utterance = one row is a *benchmark design choice* for
  human-to-agent user turns (README: "Each row is a complete user turn… The final silence span is
  the true end of the user's turn"), not a property of our data. A conversational dataset must
  define its own turns.

### 4.3 One speaker per recording?

- The schema has **no speaker identity field at all**: no `speaker_id` on rows, spans, or predictions.
  `adapted for one speaker per recording` is inherited implicitly, not enforced — the code never
  checks how many speakers are in an `audio` cell.
- Adapters see **audio prefix + optional transcript `messages` only**:
  Smart Turn (`smart_turn_adapter.py:44-74`) takes the last 8 s of audio through a Whisper feature
  extractor → ONNX `p(eot)`; LiveKit mini (`livekit_turn_detector_mini_adapter.py:82-88`)
  takes the last 1.2 s PCM-16 → `livekit.local_inference.EOT`; Silero (`silero_vad_adapter.py`)
  scores `min(silence_duration, 1.0)`. None receive speaker labels; **speaker identity cannot be
  preserved by either adapter** without changing adapter signatures.
- `supports_language()` (`languages.py:6-38`): both adapters accept any of 14 benchmark languages
  including `hi`, but a `language` string outside that set (e.g. `hinglish` / `mixed`) is
  **rejected/skipped**. Hinglish data must therefore be coarsened to `language=hi` (or split by
  per-utterance language) for the existing CLI — a forced, lossy choice that must be documented,
  or the language support list must be extended in a fork.

### 4.4 What must change to evaluate the two Hinglish conversations

1. **Dataset construction (new).** The CLI requires Hugging Face datasets (`--path/--name/--split`),
   so the conversations must be packaged as one HF-style dataset **or** the CLI must gain a local
   loader. Each *annotated conversational turn* (not the whole tape) must become a row
   `{id, language, audio, silence_spans}` where `silence_spans` are the candidate
   pauses *inside* that turn. **But this requires knowing each turn's boundaries first** —
   which requires the human SET of EOT/HOLD labels the framework currently derives (§4.2).
2. **Label semantics (must fork).** Replace the mechanical last-span=EOT rule with an explicit
   per-span EOT/HOLD ground truth from `data/annotations-hinglish/` (reviewed), and *exclude*
   `AMBIGUOUS` spans from the denominator. Note the metric layer already tolerates
   unscored holds (`official_score_table` allows holds shorter than the 0.2 s score point) but
   hard-fails on missing EOT scores (`metrics.py:120-124`).
3. **Speaker metadata (schema extension).** Add `speaker_id` (at minimum on spans) so results can be
   broken down by speaker (false-cutoff rate for Speaker 1 vs 2). This requires touching
   `schemas.py`, `io.py` (`iter_transformed_row`/`build_span_set`), adapters (they currently ignore
   non-audio fields — speaker could be carried as **metadata passed through, not a model input**),
   and `metrics.py` grouping. Alternatively keep speaker analysis *outside* the harness by joining
   on `(id, span_index, timestamp)` post-hoc — **recommended** as the least invasive path.
4. **Language handling (document or extend).** Set `language=hi` for Hinglish rows with an explicit
   caveat that code-switching is collapsed, or extend `BENCHMARK_LANGUAGES` (+ adapter
   `supports_language`) to accept `mixed` verbatim. Extending is scientifically preferable but
   changes benchmark methodology and breaks comparability with the Hindi run notation — declare it.
5. **Audio windowing semantics — examine before running.** Smart Turn always scores the trailing
   8 s, LiveKit mini the trailing 1.2 s, at each 0.1 s grid point inside a span
   (`io.py:_time_grid`, `inference_interval=0.1`). On real turns this means the "future" of the
   *same turn* is legitimately visible at later grid points (causal within-span scoring); but at a
   turn *boundary* the next speaker's voice enters the trailing window — the harness's causal
   framing assumes a single user turn captured in isolation. Two speakers in one tape therefore
   requires turn-slicing *first* (see step 1), else cross-speaker contamination biases scores.

### 4.5 Bottom line on compatibility

| Question | Answer |
|---|---|
| Supports two-speaker conversation directly? | **No.** No speaker field exists; mechanical labeling collapses a conversation to one EOT per row. |
| Assumes one speaker per recording? | **Implicitly yes** (schema has no speaker concept; adapters receive context-free audio prefixes). |
| Recordings usable as-is? | **No.** Turn slicing + human labels + HF/local dataset packaging required first. |
| Can HoLD/EOT be entered per-span? | **Only by forking** `io.py`'s label rule. As vendored, labels are derived, never read. |
| Speaker identity preserved in predictions? | **No** — must be carried externally (join post-hoc) or via schema fork. |

**Verdict: the existing Hindi benchmark results remain valid as Hindi results. The Hinglish
evaluation must build (per the annotation + pipeline work below) on the *unlabeled* conversations
first. Do not run `eot-harness predict` on the raw WAVs.**

---

## 5. Required annotation schema (proposed, §4 of brief)

### 5.1 Per-candidate pause annotation — `data/annotations-hinglish/pause_candidates.csv`

| Column | Type | Notes |
|---|---|---|
| `recording_id` | string | Fixed per file (e.g. `hinglish_001_general`). |
| `candidate_id` | string | Stable key, e.g. `hinglish_001_general__c000`. |
| `start_time`, `end_time`, `duration` | float (s) | Pre-filled from energy detection; annotator may adjust after listening. |
| `speaker_id` | `speaker_1` / `speaker_2` / `overlap` / `uncertain` | Who was speaking *before* this pause. |
| `transcript` | string | Words in the utterance(s) immediately before the pause (edit AFTER ASR/human draft available). |
| `language` | `hindi` / `english` / `mixed` / `uncertain` | Predominant language of the preceding utterance. |
| `boundary_type` | `HOLD` / `EOT` / `AMBIGUOUS` | HOLD = speaker resumes / floor retained; EOT = speaker yields, other starts or conversation seeks response; AMBIGUOUS = genuinely undecidable. **Not** derived from duration. |
| `annotation_confidence` | `high` / `medium` / `low` | Annotator self-rating. |
| `review_status` | `needs_review` / `reviewed` / `adjudicated` | Starts `needs_review` for all 153 rows. **No row will be auto-advanced.** |
| `notes` | string | Rationale for every non-HOLD label and every AMBIGUOUS. |

A companion **turn table** (one row per detected turn, created during review) should hold:
`turn_id, recording_id, speaker_id, start_time, end_time, transcript, language`
so EOT boundaries and turn structure can be audited independently of the pause candidates.

Emitting the as-yet-unverified transcript draft was **skipped**: ASR (Whisper) is not installed in
the turnsense env, and no transcript is invented here. The 153-row candidate file above is
deliberately transcript-free and its pre-filled values are audit metadata only.

### 5.2 How a human annotator identifies a genuine conversational EOT

1. Listen to the **full** preceding utterance *and* what follows the pause (± at least a few
   seconds of context). A pause alone never decides.
2. **HOLD** if the same speaker resumes with continuation intonation, a connective, self-repair,
   or the pause functions as hesitation/emphasis while they still hold the floor.
3. **EOT** only if the speaker's contribution is complete **and** the floor is yielded — the other
   speaker begins, or the pause is a response-seeking completion *and* the conversational cue
   (question, handoff, answer) shows they intended to finish. Sentence completion is **not**
   sufficient; a speaker may finish a sentence mid-turn.
4. **Not EOT by default:** a speaker change is evidence *corroborating* EOT, not proof — e.g. a
   backchannel ("haan", "okay") during the other's turn must not be scored as a turn yield.
5. **AMBIGUOUS** whenever the evidence is genuinely split (overlapping speech, inaudible content,
   sigh/laughter, ambiguous prosody). Guessing is prohibited; AMBIGUOUS is a first-class outcome,
   later excluded from AUC/AP denominators or analyzed separately.
6. Record `confidence` and a one-line rationale in `notes`. Double-annotate a subset to estimate
   inter-annotator agreement before running models.

---

## 6. What is missing before Hinglish evaluation (checklist)

- [ ] Human listening pass over both recordings → `speaker_id` on every candidate.
- [ ] Transcript per utterance (manual, or user-approved ASR + human correction).
- [ ] `language` per utterance (`hindi`/`english`/`mixed`), enabling code-switch-point analysis.
- [ ] `boundary_type` on every candidate after listening (with `notes` rationale); **no auto-labeling**.
- [ ] Oversight review (`review_status → reviewed/adjudicated`) and an agreement estimate on a
   double-annotated subset.
- [ ] Turn slicing table (turn boundaries) derived from the reviewed pauses.
- [ ] Packaging turn-sliced rows as an HF-style input (or a CLI local-loader fork), with `language`
   collapsed to `hi` **or** a declared `BENCHMARK_LANGUAGES` extension.
- [ ] Fork (or adapter-compatible shim) replacing the mechanical last-span=EOT rule with reviewed
   per-span labels; AMBIGUOUS rows excluded from scoring.
- [ ] Decision on Silero baseline semantics for two-speaker audio (its `min(silence/1s,1)` score is
   span-agnostic and works as-is; keep for comparability with the Hindi Silero result).
- [ ] Reproduce Hindi summary numbers from prediction files (already verified exactly: AUC/AP match
   0.7189/0.5609, 0.8422/0.7651, 0.5299/0.4351) and freeze the script before scoring Hinglish.

---

## 7. Recommended next steps (in order)

1. **Human annotation sprint** (this is the critical path; models cannot be scored without it):
   use `data/annotations-hinglish/pause_candidates.csv` as the review queue, listening at each
   candidate with context, completing `speaker_id`, `transcript`, `language`, `boundary_type`,
   `confidence`, `notes`. Target ~150 pauses; flag overlaps/abbreviations.
2. **Turn table** from step 1 (deduplicate candidate-level pauses into turns).
3. **Fork decisions** (§4.4) documented in a methodology note *before* any Hinglish score is
   computed, so Hindi↔Hinglish comparability caveats are pre-registered.
4. **Pipeline**: slice the two WAVs into turn clips, build spans per turn from reviewed labels,
   run Smart Turn v3.2 + LiveKit v1-mini + Silero (existing adapters unchanged), compute
   AUC/AP/false-cutoff/latency + per-speaker breakdown from external joins.
5. **Failure analysis**: inspect code-switch-adjacent pauses and the 4.2 s 002 gap as case studies.
6. **Report**: write the Stage 1 report with the methodology fork declared; fold into
   (or supersede) `reports/results.md` as the author decides. Preserve Hindi results untouched.

---

## Appendix — verified Hindi benchmark (re-checked this audit)

| Model | Points | Recordings | AUC | AP | Mean frontier latency (ms) | Verified how |
|---|---|---|---|---|---|---|
| Smart Turn v3.2 | 8,212 | 400 | **0.7189** | **0.5609** | **725.52** | Recomputed via `official_score_table(0.2 s)` + 0.2–5.0 s hold filter from `predictions.parquet` — exact match to `metrics-smart-turn-hi/summary.json` |
| LiveKit v1-mini | 8,212 | 400 | **0.8422** | **0.7651** | **682.46** | Same recomputation from `...live_kit_turn_detector_mini_adapter__9e7a68f5e3/predictions.parquet` — exact |
| Silero VAD | 8,212 | 400 | **0.5299** | **0.4351** | **791.33** | Same recomputation from `...silero_vad_adapter__558ea51e93/predictions.parquet` — exact |

Caveats, unchanged: these are **Hindi** results. `span_set.parquet` has 1,074 spans (674 HOLD /
400 EOT), and the framework's label rule implies exactly one EOT per recording (see `io.py:85`).
Medians of prediction-point AUC differ from the official span-point AUC (e.g. point-level
recomputes to ~0.77–0.80) — a methodology artifact of one-score-per-span, not a data error.
LiveKit v1 cloud run (`experiments/eot-bench/livekit-v1-hi/`) remains span_set-only, consistent
with the HTTP 429 quota block. **Do not cite any of these numbers as Hinglish performance.**
