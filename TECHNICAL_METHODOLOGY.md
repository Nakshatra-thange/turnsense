# TurnSense — Technical Methodology and Reproducibility

## 1. System overview

TurnSense evaluates existing end-of-turn detection systems on Hindi benchmark data and conversational Hinglish recordings.

The workflow consists of:

1. Benchmark preparation.
2. Conversational audio collection.
3. Pause candidate generation.
4. Human annotation and audit.
5. Automatic transcription.
6. Transcript alignment.
7. Speaker-aware turn construction.
8. Causal model evaluation.
9. Results analysis.

## 2. Environment

- Platform: Apple Silicon macOS
- Architecture: ARM64
- Python: 3.11.18
- Environment: Conda `turnsense`
- Transcription: faster-whisper 1.2.1
- CTranslate2: 4.8.5

The transcription workflow selected the multilingual Whisper Base model with INT8 computation after larger alternatives showed poor practical performance and transcription quality on the tested Hinglish audio.

The selected configuration produced approximately 9–10 seconds of transcription runtime per recording.

Transcripts preserve Romanized Hinglish output rather than silently converting it into another script.

## 3. Conversational corpus

Three processed recordings:

```text
data/processed/
├── hinglish_conversation_001_general_16k.wav
├── hinglish_conversation_002_cricket_16k.wav
└── hinglish_conversation_003_movies_16k.wav
```

All files are mono, 16 kHz WAV.

The corpus contains three short two-speaker conversations.

## 4. Pause generation and annotation

Pause candidates were generated using energy-based detection:

- RMS window: 100 ms
- Silence threshold: approximately -45 dBFS
- Contiguous silent windows merged
- Pauses shorter than 200 ms discarded

Human reviewers assigned HOLD, EOT, or AMBIGUOUS labels.

A targeted EOT audit examined 31 candidates:
- 30 confirmed EOT.
- 1 corrected to HOLD.

Final reviewed annotation distribution:

| Label | Count |
|---|---:|
| EOT | 100 |
| HOLD | 22 |
| AMBIGUOUS | 1 |
| Total | 123 |

The audit was a targeted quality check, not independent verification of every annotation.

## 5. Transcription and alignment

faster-whisper generated timestamped transcription segments and word timestamps.

Outputs were stored under:

`data/transcripts/`

The enriched annotation file:

`data/annotations-hinglish/pause_candidates_enriched.csv`

contains 123 rows and adds preceding/following draft-ASR context, heuristic language information, and alignment confidence.

Manual annotation fields were preserved.

Language classification is heuristic. The presence of a code-switch flag does not constitute independently verified linguistic annotation.

## 6. Turn-level representation

The turn-level dataset contains:

- `turns.csv`: 102 turns.
- `pauses.csv`: 123 pause records.
- `evaluation_protocol.md`: evaluation assumptions and methodology.

The protocol documents causal evaluation, exclusions, and comparability limitations.

The final binary scoring set contains 116 examples:
- 100 EOT.
- 16 HOLD.

Six tail HOLD examples and the initial pre-conversation candidate were excluded. Four mid-turn HOLD cases were explicitly flagged.

## 7. Model evaluation

Two existing systems were evaluated:

- Smart Turn v3.2.
- LiveKit Turn Detector v1-mini.

Evaluation used the vendored EoT-Bench code with:
- Causal onset.
- `silence_dur = 0`.
- 0.5-second transcript lag.

A second timing variant using a 0.2-second adjustment was also evaluated.

Metrics:
- ROC AUC.
- Average precision.
- False cutoffs at a selected 0.5 operating threshold.

False cutoff rate is calculated as:

\[
\text{False Cutoff Rate} =
\frac{\text{HOLD examples predicted as EOT}}
{\text{Scored HOLD examples}}
\]

AP must be interpreted against the EOT prevalence of 100/116, approximately 86.2%.

## 8. Validation and tests

The implementation passed seven automated tests.

The master annotation file was reconciled:
- 123 reviewed candidates.
- Unique candidate IDs.
- Audit correction present.
- Manual annotation fields preserved through enrichment.

Hindi benchmark results were re-verified against the original prediction files.

## 9. Reproducibility and limitations

Important limitations:

- The conversational corpus is small.
- Speaker diversity is limited.
- Pause candidates depend on energy-based detection.
- Automatic transcription is draft evidence, not ground truth.
- Language flags are heuristic.
- The binary evaluation contains only 16 HOLD examples.
- The targeted audit does not establish complete annotation reliability.
- Hindi and Hinglish results use different data and evaluation structures.

Large prediction files are excluded from Git through `.gitignore`.

The original Hindi benchmark outputs have been preserved.

## 10. Repository outputs

Key reports:

```text
reports/
├── results.md
├── stage1-hinglish.md
├── experiment-002.md
└── experiment-003.md
```

Key conversational data:

```text
data/
├── transcripts/
└── annotations-hinglish/
    ├── pause_candidates.csv
    ├── pause_candidates_enriched.csv
    ├── eot_audit_results.csv
    └── ...
```

Additional evaluation artifacts include the turn and pause datasets and the documented evaluation protocol.

## 11. Reproduction principle

All results should be interpreted with their corresponding dataset, prediction files, evaluation configuration, and exclusions.

Do not treat the Hindi benchmark results as Hinglish results, or the small Hinglish pilot as evidence of production-wide performance.