# TurnSense — Experimental Results

## Experiment 001: Smart Turn v3.2 on Hindi Benchmark

**Date:** November 3, 2026  
**Status:** Completed  
**Model:** Smart Turn v3.2  
**Evaluation Framework:** LiveKit EoT-Bench

---

## 1. Objective

Establish an initial baseline by evaluating an existing End-of-Turn (EOT) detection model on the Hindi validation dataset.

This experiment is part of a broader research project investigating the performance of existing EOT detection systems on Hinglish and code-switched speech.

**Important:** This experiment uses the Hindi benchmark. It is not yet an evaluation on a verified Hinglish dataset.

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | `livekit/eot-bench-data` |
| Configuration | `hi` |
| Split | `validation` |
| Original recordings | 400 |
| Model | Smart Turn v3.2 |
| Model adapter | `SmartTurnAudioAdapter` |
| Inference interval | 100 ms |
| Transcript lag | 500 ms |
| Batch size | 32 |
| Score point | 0.2 seconds |
| Minimum silence span | 100 ms |

The benchmark progressively provides audio to the model and evaluates its predictions at successive points during silence.

## 3. Dataset Statistics

The prediction run processed:

- 8,212 prediction points.
- 1,074 eligible silence spans in the original span set.
- 952 spans included in the final metrics calculation.

Breakdown of scored spans:

| Category | Count |
|---|---:|
| HOLD | 552 |
| EOT | 400 |
| Total scored spans | 952 |
| Missing scores | 0 |

**Note:** The metrics calculation applies a minimum HOLD-span duration of 0.2 seconds and a maximum of 5 seconds. This explains why the scored span count differs from the original span-set count.

## 4. Model Performance

### Classification metrics

| Metric | Result |
|---|---:|
| AUC | 0.7189 |
| Average Precision (AP) | 0.5609 |

**AUC (Area Under the ROC Curve):** Measures how well the model separates HOLD and EOT examples across possible thresholds.

An AUC of 0.7189 means that, for a randomly selected EOT and HOLD pair, the model ranks the EOT example higher approximately 71.9% of the time.

**Average Precision (AP):** Summarises precision-recall performance across different score thresholds.

Neither metric represents classification accuracy.

### Operational performance

The benchmark reports operating points with different tolerated cutoff budgets.

| Cutoff budget | Threshold | Cutoff rate | Mean latency | Detection rate |
|---|---:|---:|---:|---:|
| 2% | 0.02 | 1.99% | 813 ms | 93.5% |
| 5% | 0.04 | 4.71% | 639 ms | 90.25% |
| 10% | 0.02 | 9.24% | 533 ms | 93.5% |

The cutoff rate represents the proportion of HOLD spans that trigger an early response under the selected policy.

The detection rate represents the proportion of EOT spans detected within the policy's timeout.

These results demonstrate the trade-off between response latency and the risk of interrupting a speaker.

### Additional metric

| Metric | Result |
|---|---:|
| Mean frontier latency (0–10% cutoff range) | 725.52 ms |

## 5. Runtime Observations

- Total prediction points: 8,212.
- Total prediction runtime: approximately 4 minutes 41 seconds.
- Average processing speed: approximately 29.19 prediction points/second.
- ONNX Runtime reported that CUDA was unavailable on the machine.
- The evaluation nevertheless completed successfully.

## 6. Initial Observations

1. Smart Turn successfully generated predictions for all 8,212 evaluation points.
2. No missing scores were reported across the 952 scored spans.
3. The AUC of 0.7189 indicates useful separation between HOLD and EOT examples, although there is room for improvement.
4. The operating-point results show that latency and early-cutoff risk need to be considered together.
5. The Hindi benchmark provides an initial reference point for subsequent detector comparisons.

These observations are limited to the evaluated Hindi validation dataset.

## 7. Limitations

- This experiment does not evaluate a verified Hinglish dataset.
- Hindi benchmark results cannot be treated as Hinglish performance.
- AUC and AP do not directly measure real-world interruption behaviour.
- Only one detector has been evaluated so far.
- The results are from our local reproduction, not copied from a published leaderboard.

## 8. Next Steps

1. Evaluate additional existing EOT detectors using the same benchmark.
2. Compare their classification and operational metrics.
3. Investigate the Hindi dataset's mixed-script examples and verify actual code-switching through audio review.
4. Collect and annotate a pilot Hinglish dataset.
5. Evaluate existing detectors on the Hinglish pilot dataset.
6. Analyse false cutoffs, delayed responses and language-specific failure cases.

## 9. Conclusion

The first TurnSense experiment successfully evaluated Smart Turn v3.2 on LiveKit's Hindi validation benchmark.

The experiment establishes an initial baseline:

- **AUC:** 0.7189
- **AP:** 0.5609
- **Mean frontier latency:** 725.52 ms

This is an initial benchmark result, not evidence of performance on Hinglish speech.

The central research objective remains evaluating how well existing EOT detection systems handle Hinglish and Hindi-English code-switching.

---

*TurnSense — Investigating End-of-Turn Detection in Hinglish Voice Conversations.*