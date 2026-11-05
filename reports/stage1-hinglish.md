# TurnSense Stage 1 — Hinglish Pilot Evaluation

| Field | Value |
|---|---|
| Date | 2026-11-05 (Phases A–F) |
| Environment | `~/miniforge3-arm64/envs/turnsense` (Python 3.11.18, arm64) |
| Corpus | 3 conversational Hinglish recordings, 2 speakers, ~13 min |
| Labels | 123 reviewed pauses (100 EOT / 22 HOLD / 1 AMBIGUOUS pre-conversation) |
| Scored this phase | **116 causal-onset pauses** (100 EOT / 16 HOLD; 6 unclosed-tail HOLDs + 1 pre-conversation unscored) |
| Status | **Evaluation complete. Metrics are honest, including weak AUC. See limitations.** |

---

## 1. What was built (Phases A–E audit trail)

**Phase A — Integrity.** Master file `pause_candidates.csv` verified: 123 reviewed
candidates, unique IDs, audit correction `hinglish_003_movies__c066` present as HOLD
in both master and `recording_003_candidates.csv`, blank transcripts/languages confirmed
as the gap to fill. Hindi benchmark numbers re-verified exactly (see §3).

**Phase B — Transcription.** Installed **faster-whisper 1.2.1 + ctranslate2 4.8.5**
(the turnsense env had none). `distil-large-v3`/`small` were tried and REJECTED with
evidence: small int8 = 213 s for 60 s audio with garbled Devanagari/CJK artifacts;
distil-large-v3 timed out (>600 s) on a 15 s slice. Selected **`base` int8**
(231 s file in ~9–10 s, clean romanized Hinglish, e.g. *"YAR, I have been thinking
about AI lately… Mujinii lakta, completely replace karega…"*). Outputs in
`data/transcripts/` (226 segments, word timestamps). **No diarization, no `speaker_id`
column anywhere in ASR output.**

**Phase C — Alignment.** `tools/align_transcripts.py` → `pause_candidates_enriched.csv`
(123 rows). Preserves all manual fields verbatim; adds `preceding_text`/`following_text`
(timestamp-nearest segments), heuristic `preceding_language`/`following_language`
(script + roman lexicon; `uncertain` when thin), alignment confidence (mean 0.46,
range 0.00–1.00 — ASR segments are long utterances, so per-utterance alignment is
coarse by design), and `code_switch_around_pause` (31/123 = 25%).

**Phase D — Turn dataset.** `tools/build_turn_dataset.py` → `turns.csv` (102 turns) +
`pauses.csv` (123 pauses) + `dataset_manifest.json` + **`evaluation_protocol.md`**
(pre-registered protocol). Turn ends at each EOT pause; HOLDs internal; 6 tail HOLDs
and c000 unscored. 4 turns retain a mid-turn HOLD by the non-closing speaker (flagged,
not hidden). All 7 pipeline tests pass (`tests/test_turnsense_pipeline.py`).

**Phase E — Scoring.** `tools/score_hinglish.py` with the **unmodified vendored adapters**
(Smart Turn v3.2, LiveKit v1-mini local) on causal-onset prefixes. **No cloud v1 retry.**

---

## 2. Hinglish results (causal onset, `silence_dur=0`)

| Model | n | AUC | AP | ms/pause | RTF (CPU) | False cutoffs @0.5 | EOT detect @0.5 |
|---|---|---|---|---|---|---|---|
| Smart Turn v3.2 | 116 (100 EOT/16 HOLD) | **0.516** | **0.863** | 51.4 ms | 0.008 | 4/16 (25%) | 36/100 (36%) |
| LiveKit v1-mini | 116 | **0.450** | **0.871** | 10.5 ms | 0.002 | 11/16 (69%) | 52/100 (52%) |

**Score POINT matters little here.** At the harness-conventional +0.2 s into-silence
variant (`scored_pauses_*_at-0.2s.csv`): Smart Turn AUC 0.496 / AP 0.846,
LiveKit-mini AUC 0.604 / AP 0.902. The onset protocol is therefore not a
measurement artifact — at conversational floor-yield moments, these models have
weak prospect discrimination in this corpus regardless of 200 ms of silence.

**AUC below chance for LiveKit-mini (0.45) is itself a finding:** the model's
p(eot) is *higher on HOLDs* (median 0.56) than on EOTs (0.52) at onset. AP > AUC
in both rows reflects the 100/16 class imbalance (a constant high score yields
AP ≈ 0.86 from prevalence alone — **do not cite AP without that context**).

### Breakdowns (exploratory, n small — read with care)
- **Per recording:** Smart Turn AUC 0.66 (001 general, 66/3), 0.36 (002 cricket,
  19/5), 0.53 (003 movies, 15/8). LiveKit-mini 0.35 / 0.38 / 0.47. Recording
  differences dominate; no recording-level generalization is licensed.
- **Per speaker (Smart Turn):** speaker_1 AUC 0.52 (53/10), speaker_2 AUC 0.52
  (47/6). LiveKit-mini: 0.41 vs 0.52. No meaningful speaker effect given n.
- **Per preceding language (Smart Turn):** English n=54 EOT median p=0.14;
  Hindi romanized n=16 median p=0.71 (higher predicted EOT — directionally
  interesting, underpowered to claim). Mixed n=9 (all EOT, no HOLDs to compare).
- **Code-switch around pause (31/116):** no stable model-level effect visible at
  this sample size; the heuristic itself agrees with only a subset of cases.
  Treat all code-switch breakdowns as hypothesis-generating only.

### Failure cases (quoted draft-ASR context; verify against audio)
- **Smart Turn false cutoffs @0.5 (4/16):** e.g. `003__c066` p=0.95 — the audited
  HOLD (*"…actually ek tis notis ki…"* → *"ah, chai question…"*); Smart Turn
  reads statement completion as EOT. LiveKit-mini false cutoffs (11/16) include
  fluent mid-turn fillers (*"…rancho character is so interesting"* →
  self-continuation) across both models — the classic premature-cutoff mode.
- **Smart Turn misses (55/100 < 0.2):** e.g. `001__c018` *"Are you prepared for
  that?"* p=0.006; `001__c027` *"Like what? What else…"* p=0.007 — short
  question-form EOTs scored as HOLD-like at onset. These are genuinely
  difficult without silence evidence.

---

## 3. Hindi benchmark (verified, for caveated comparison)

| Model | AUC | AP | Mean frontier latency | n |
|---|---|---|---|---|
| Smart Turn v3.2 | 0.7189 | 0.5609 | 725.52 ms | 400 rec. / 8,212 pts / 952 spans |
| LiveKit v1-mini | 0.8422 | 0.7651 | 682.46 ms | 400 rec. / 8,212 pts / 952 spans |
| Silero + silence policy | 0.5299 | 0.4351 | 791.33 ms | 400 rec. / 8,212 pts / 952 spans |

LiveKit cloud v1 remains **not run** (HTTP 429). **Never cite the Hindi numbers as
Hinglish results.** The Hinglish pilot (conversational floor-yield on 2 speakers)
and the Hindi benchmark (utterance-final on single-speaker agent turns) measure
different phenomena under different protocols (0 s vs 0.2 s score points among
others); quantitative AUC gaps **cannot** be attributed to code-switching without
controls this pilot does not have.

---

## 4. Latency vs throughput (kept separate, as required)
- **Throughput:** Smart Turn 51 ms/pause (RTF 0.008), LiveKit-mini 11 ms/pause
  (RTF 0.002) on CPU — both comfortably real-time for onset scoring.
- **Policy latency (Hindi frontier):** 682–791 ms of *silence tolerated* per the
  harness sweep (not measured for Hinglish onset scoring — onset has 0 ms of
  silence by construction). Do not compare throughput to frontier latency.

---

## 5. Limitations (must travel with any citation)
1. 2 speakers, 3 recordings, ~13 min, 116 scored pauses. **Pilot scale.**
2. Energy-detector candidate bias; durations <0.2 s or masked pauses unevaluated.
3. Draft ASR + script/lexicon language heuristic are provenance, not ground truth.
4. Manual speaker IDs; no automatic diarization anywhere.
5. Audit (31 checks, 30/31 confirmed) is targeted QC, not full verification;
   24/24 samples from 002/003 were temporal/quantile bins — deliberately
   NOT representative sampling.
6. AP inflated by 100/16 prevalence; always report AUC alongside it.
7. Score-point asymmetry (0 s vs 0.2 s) blocks naive Hindi↔Hinglish ranking.

---

## 6. Files & reproducibility
- Scripts: `tools/transcribe.py`, `tools/align_transcripts.py`,
  `tools/build_turn_dataset.py`, `tools/score_hinglish.py`.
- Transcripts: `data/transcripts/` (3 JSON + combined_segments.csv + metadata).
- Enriched + turn data: `data/annotations-hinglish/*enriched.csv`, `turns.csv`,
  `pauses.csv`, `dataset_manifest.json`, `evaluation_protocol.md`.
- Scores: `experiments/eot-bench/hinglish/predictions_*.parquet`,
  `scored_pauses_*.csv`, `metrics_*.json`, plus `*_at-0.2s.csv` variant.
- Tests: `tests/test_turnsense_pipeline.py` (7 passed).
- All manual labels preserved; no WAVs modified; HiACC untouched (CC BY-NC 4.0).

## 7. Next steps (not done here)
- Human-correct ASR on the 116 scored pauses; re-run alignment with §2.1 note.
- A powered follow-up (more speakers/recordings) before any code-switch claim.
- Silero silence baseline on Hinglish causal protocol (optional, for completeness).
- Optional normal-vs-Hinglish failure-analysis note once human review closes.
