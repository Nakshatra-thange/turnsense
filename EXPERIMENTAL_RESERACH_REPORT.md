# TurnSense — Experimental Results

## 1. Hindi benchmark results

Three systems were evaluated on the EoT-Bench Hindi validation set.

Dataset statistics:
- 400 recordings
- 8,212 prediction points
- 952 scored spans
- 552 HOLD and 400 EOT spans

| Model | AUC | AP | Mean frontier latency | Runtime |
|---|---:|---:|---:|---:|
| Smart Turn v3.2 | 0.7189 | 0.5609 | 725.52 ms | 4m 41s |
| LiveKit Mini | 0.8422 | 0.7651 | 682.46 ms | 1m 24s |
| Silero VAD + silence policy | 0.5299 | 0.4351 | 791.33 ms | 5m 11s |

### Interpretation

LiveKit Mini achieved the highest AUC and AP, followed by Smart Turn.

The Silero baseline performed close to chance. This experiment used a silence-duration policy built on Silero VAD and should not be interpreted as an evaluation of Silero VAD's general speech-detection capabilities.

A full LiveKit cloud v1 experiment was attempted but blocked by a 429 quota error. No valid predictions were produced.

## 2. Conversational Hinglish results

The Hinglish pilot consisted of three conversational recordings.

Initial annotation:
- 123 reviewed pause candidates.
- 100 EOT.
- 22 HOLD.
- 1 AMBIGUOUS.

The final binary evaluation included:
- 116 scored boundaries.
- 100 EOT.
- 16 HOLD.

Six tail HOLD examples and the initial pre-conversation candidate were not scored. Four mid-turn HOLD cases were explicitly flagged in the evaluation documentation.

### Model performance

| Metric | Smart Turn v3.2 | LiveKit Mini |
|---|---:|---:|
| AUC | 0.516 | 0.450 |
| AP | 0.863 | 0.871 |
| False cutoffs at 0.5 | 4/16 | 11/16 |
| False-cutoff rate | 25% | 68.75% |

### Interpretation

Both systems showed weak discrimination on this pilot.

Smart Turn achieved an AUC close to 0.5, indicating near-chance ranking performance.

LiveKit Mini achieved an AUC below 0.5, indicating that its ranking was worse than chance under this particular evaluation setup.

The AP values appear high, but the EOT class represents 86.2% of scored examples. The prevalence baseline for AP is therefore approximately 0.862.

Consequently, AP values of 0.863 and 0.871 should not be interpreted as strong performance.

### Operating-point robustness

A second evaluation variant introduced a 0.2-second adjustment:

| Model | Original AUC | Adjusted AUC | Original AP | Adjusted AP |
|---|---:|---:|---:|---:|
| Smart Turn | 0.516 | 0.496 | 0.863 | 0.846 |
| LiveKit Mini | 0.450 | 0.604 | 0.871 | 0.902 |

The changes show that the results are sensitive to the evaluation timing configuration. Therefore, the original results should be reported alongside the adjusted variant rather than presenting either as definitive.

## 3. Cross-stage comparison

The Hindi benchmark and conversational Hinglish results show a substantial difference.

| Model | Hindi AUC | Hinglish AUC |
|---|---:|---:|
| Smart Turn v3.2 | 0.7189 | 0.516 |
| LiveKit Mini | 0.8422 | 0.450 |

This comparison is suggestive, not a controlled language-only comparison. The datasets, recording conditions, annotation structures, and evaluation protocols differ.

It cannot establish that code-switching alone caused the performance difference.

## 4. Main findings

- LiveKit Mini led the Hindi benchmark.
- Neither model showed convincing discrimination on the conversational Hinglish pilot.
- False cutoffs were particularly frequent for LiveKit Mini.
- AP was strongly influenced by EOT prevalence.
- Timing configuration affected the results.

## 5. Limitations

The Hinglish evaluation has only 16 scored HOLD examples. The false-cutoff rates are therefore sensitive to individual examples.

The corpus contains three recordings and limited speaker diversity. The results cannot be generalized to all Hinglish speakers, accents, or conversational environments.

The energy-based pause-generation process also conditions which boundaries are available for evaluation.

## 6. Conclusion

The results establish a useful pilot baseline and highlight the need for larger conversational datasets and more comprehensive HOLD coverage.

Future experiments should prioritize additional speakers, more natural conversational contexts, and systematic analysis of false-cutoff cases.