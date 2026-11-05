# TurnSense Results (canonical)

| Field | Value |
|---|---|
| Date | 2026-11-05 |
| Status | Hindi benchmark complete; Hinglish pilot evaluated (116 scored pauses) |

---

## 1. Hindi benchmark (EoT-Bench validation, hi)

Each experiment: 400 recordings, 8,212 prediction points, 952 scored spans.
Verified exactly by recomputing `official_score_table(0.2s)` from
`predictions.parquet` — do not present as Hinglish results.

| Model | AUC | AP | Mean frontier latency |
|---|---:|---:|---:|
| Smart Turn v3.2 | 0.7189 | 0.5609 | 725.52 ms |
| LiveKit Turn Detector v1-mini | 0.8422 | 0.7651 | 682.46 ms |
| Silero VAD + silence policy | 0.5299 | 0.4351 | 791.33 ms |

Full LiveKit cloud v1: attempted, **blocked by HTTP 429**, no valid predictions —
not a completed experiment. Silero baseline is a **silence-duration policy**, not
a native Silero EOT probability model.

Source files:
- `experiments/eot-bench/livekit__eot-bench-data__validation__min_silence_100ms/hi/`
  (`span_set.parquet`, 1,074 spans = 674 hold + 400 eot; per-model
  `predictions.parquet` + `manifest.json`)
- `experiments/eot-bench/metrics-*/summary.json` + `tradeoff.parquet`

---

## 2. Hinglish conversational pilot (causal onset, `silence_dur=0`)

3 recordings, 2 speakers, 123 reviewed pauses → **116 scored** (100 EOT / 16 HOLD).
Detailed analysis: `reports/stage1-hinglish.md`. Protocol:
`data/annotations-hinglish/evaluation_protocol.md`.

| Model | AUC | AP | ms/pause | False cutoffs @0.5 | EOT detect @0.5 |
|---|---:|---:|---|---|---|
| Smart Turn v3.2 | 0.516 | 0.863 | 51.4 | 4/16 (25%) | 36/100 (36%) |
| LiveKit v1-mini | 0.450 | 0.871 | 10.5 | 11/16 (69%) | 52/100 (52%) |

AP is inflated by the 100/16 prevalence (constant-high baseline ≈ 0.86) — always
report AUC alongside AP. At the harness-conventional +0.2 s variant scores are
similar (Smart Turn 0.496/0.846; LiveKit-mini 0.604/0.902), so the weakness is
not a pure score-point artifact.

**Comparability caveat (mandatory):** Hindi scores at 0.2 s INTO silence
(`score_point_s=0.2`) on single-speaker utterance-final labels; Hinglish scores
at pause ONSET on conversational floor-yield labels. Different phenomena,
different protocols — AUC gaps cannot be attributed to code-switching without
controls this pilot lacks. No generalization claim from either dataset beyond
its population.

Source files (new this phase — do not overwrite Hindi):
- `experiments/eot-bench/hinglish/predictions_*.parquet`,
  `scored_pauses_*.csv`, `metrics_*.json`, `*_at-0.2s.csv`
- `data/annotations-hinglish/pause_candidates_enriched.csv`,
  `turns.csv`, `pauses.csv`, `dataset_manifest.json`
- `data/transcripts/` (faster-whisper `base` draft ASR; no diarization)
- Scripts: `tools/transcribe.py`, `align_transcripts.py`,
  `build_turn_dataset.py`, `score_hinglish.py`
- Tests: `tests/test_turnsense_pipeline.py` (7 passed)
