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

# Experiment 002 — LiveKit Turn Detector v1-mini

**Date:** November 3, 2026  
**Status:** Completed  
**Dataset:** LiveKit EoT-Bench Hindi Validation  
**Model:** LiveKit Turn Detector v1-mini

## 1. Objective

Evaluate LiveKit Turn Detector v1-mini on the same Hindi validation dataset used in Experiment 001.

The purpose is to establish a second baseline and enable a controlled comparison with Smart Turn v3.2.

## 2. Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | `livekit/eot-bench-data` |
| Configuration | `hi` |
| Split | `validation` |
| Original recordings | 400 |
| Model | LiveKit Turn Detector v1-mini |
| Adapter | `LiveKitTurnDetectorMiniAdapter` |
| Inference interval | 100 ms |
| Transcript lag | 500 ms |
| Batch size | 32 |
| Score point | 0.2 seconds |
| Audio context | Last 1.2 seconds |

The model was executed locally using `livekit-local-inference`.

## 3. Prediction Results

| Metric | Result |
|---|---:|
| Total prediction points | 8,212 |
| Completed prediction points | 8,212 |
| Runtime | 84 seconds |
| Processing speed | 97.34 points/second |

The prediction run completed successfully.

## 4. Dataset Statistics

The metrics calculation used the same scored span set as Experiment 001.

| Category | Count |
|---|---:|
| HOLD spans | 552 |
| EOT spans | 400 |
| Total scored spans | 952 |
| Missing scores | 0 |

## 5. Classification Performance

| Metric | Result |
|---|---:|
| AUC | 0.8422 |
| Average Precision | 0.7651 |

The AUC indicates stronger separation between HOLD and EOT examples than observed in Experiment 001.

The higher Average Precision also indicates stronger precision-recall performance.

## 6. Operational Performance

| Cutoff budget | Threshold | Cutoff rate | Mean latency | Detection rate |
|---|---:|---:|---:|---:|
| 2% | 0.11 | 1.81% | 803 ms | 98.5% |
| 5% | 0.36 | 4.89% | 624 ms | 75.25% |
| 10% | 0.30 | 9.78% | 505 ms | 82.5% |

Mean frontier latency across the 0–10% cutoff range: **682.46 ms**.

## 7. Initial Observations

1. The model completed all 8,212 prediction points without missing scores.
2. AUC of 0.8422 indicates stronger class separation than Smart Turn v3.2 in Experiment 001.
3. AP of 0.7651 indicates stronger precision-recall performance.
4. Mean frontier latency was approximately 682 ms.
5. Detection rate varied across operating points, showing that better classification metrics do not automatically guarantee better operational detection behaviour.

## 8. Conclusion

LiveKit Turn Detector v1-mini achieved stronger AUC and Average Precision than Smart Turn v3.2 on the Hindi validation benchmark.

However, operational detection rates varied across cutoff budgets. The models must therefore be compared using both classification and operational metrics.

These results are specific to the Hindi validation dataset and do not establish performance on Hinglish speech.

## 9. Next Steps

- Complete comparisons with additional EOT detectors.
- Investigate operating-point trade-offs.
- Curate and annotate genuine Hinglish audio examples.
- Evaluate existing detectors on the Hinglish pilot dataset.
- Analyse language-specific failure cases.

---

*TurnSense — Investigating End-of-Turn Detection in Hinglish Voice Conversations.*

# Experiment 003: Silero VAD + Silence-Based Endpointing

## 1. Objective

Evaluate whether a lightweight Voice Activity Detection (VAD) model, combined with a silence-duration policy, can serve as an effective baseline for end-of-turn (EOT) detection.

Unlike dedicated EOT models, Silero VAD detects speech activity rather than directly predicting whether a speaker has finished their turn.

## 2. Experimental setup

| Parameter | Configuration |
|---|---|
| VAD model | Silero VAD |
| Package | silero-vad 6.2.3 |
| Dataset | LiveKit EoT-Bench |
| Language configuration | Hindi (`hi`) |
| Split | Validation |
| Recordings | 400 |
| Inference interval | 100 ms |
| Transcript lag | 500 ms |
| Scored spans | 952 |
| HOLD spans | 552 |
| EOT spans | 400 |

### Endpointing policy

The adapter processes each causal audio prefix using Silero VAD, identifies the end of the last detected speech segment, and calculates the trailing silence duration.

The silence duration is converted into an EOT score:

`p_eot = min(detected_silence_duration / 1.0, 1.0)`

The resulting score is passed to EoT-Bench for evaluation.

This is a VAD-based endpointing policy, not a native EOT probability produced by Silero.

## 3. Results

| Metric | Result |
|---|---:|
| AUC | 0.5299 |
| Average Precision | 0.4351 |
| Mean frontier latency | 791.33 ms |
| Scored spans | 952 |
| Missing scores | 0 |
| Prediction points | 8,212 |
| Runtime | 5 min 11 sec |
| Throughput | 26.41 prediction points/sec |

### Operating points

| Cutoff budget | Detection rate | Mean latency | Timeout rate |
|---|---:|---:|---:|
| 2% | 55.25% | 834.25 ms | 44.75% |
| 5% | 100% | 700 ms | 0% |
| 10% | 100% | 600 ms | 0% |

## 4. Additional analysis

The Pearson correlation between the predicted EOT scores and the benchmark's silence duration was:

**r = 0.9126**

Prediction score statistics:

| Statistic | Value |
|---|---:|
| Mean | 0.4695 |
| Median | 0.4040 |
| Standard deviation | 0.3628 |
| Minimum | 0.0000 |
| Maximum | 1.0000 |
| Unique scores | 1,124 |

The high correlation indicates that the adapter's scores are strongly driven by silence duration.

This suggests that the current policy behaves largely like a silence-duration endpointing rule. It does not establish that Silero's underlying speech detector is ineffective.

## 5. Comparison with previous experiments

| Model | AUC | AP | Mean frontier latency |
|---|---:|---:|---:|
| Smart Turn v3.2 | 0.7189 | 0.5609 | 725.52 ms |
| LiveKit Turn Detector v1-mini | 0.8422 | 0.7651 | 682.46 ms |
| Silero VAD + Silence Policy | 0.5299 | 0.4351 | 791.33 ms |

LiveKit Mini currently provides the strongest ranking performance and lowest mean frontier latency among these three evaluated systems.

Silero's current policy performs substantially worse on AUC and AP.

## 6. Interpretation and limitations

1. Silero VAD is a speech-activity detector, not a dedicated conversational endpointing model.
2. The adapter converts detected trailing silence into an EOT score using a simple duration-based formula.
3. The high correlation with silence duration indicates that the current policy contributes limited demonstrated information beyond pause length.
4. The adapter independently processes each causal audio prefix rather than maintaining one continuous VAD state across the recording.
5. These results are from the Hindi validation configuration. They must not be presented as Hinglish performance.
6. The operating-point results show a trade-off between false interruptions and completed-turn detection.

## 7. Conclusion

The experiment successfully integrated Silero VAD into EoT-Bench and generated predictions for all 8,212 inference points.

The current VAD-plus-silence policy achieved an AUC of 0.5299 and AP of 0.4351, making it the weakest of the three evaluated approaches.

The results support retaining this implementation as a lightweight baseline, rather than treating it as a competitive dedicated EOT detector.

## 8. Next research direction

Move from general Hindi benchmark evaluation towards identifying, manually verifying, and evaluating genuine Hindi-English code-switched speech.

The next phase will investigate whether existing endpointing systems behave differently on Hinglish conversations, particularly around pauses within incomplete sentences and code-switched utterances.