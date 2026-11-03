Yes. TurnSense v1 is primarily an evaluation and benchmarking project, not a model-building project.

I want us to be very clear about the scope before we start so that we don't accidentally turn it into another enormous project.

# 1. What exactly are we building?

# TurnSense

A Hinglish End-of-Turn Detection Evaluation Study

Research question:

How well do existing voice-agent turn-detection models identify when a speaker has finished speaking, particularly in Hinglish and code-switched conversations?

Our objective is to evaluate existing systems, identify their weaknesses, and publish reproducible findings.

# 2. What we ARE doing

In scope

| Activity               | What we'll do                                             |
| ---------------------- | --------------------------------------------------------- |
| Research               | Understand existing end-of-turn detection systems         |
| Benchmarking           | Evaluate existing models on common datasets               |
| Baseline reproduction  | Reproduce selected published results                      |
| Hinglish data          | Collect 100–200 labelled candidate pauses                 |
| Comparative evaluation | Compare 3–4 existing detectors                            |
| Error analysis         | Identify where and why models fail                        |
| Documentation          | Publish methodology, code and findings                    |
| Demo                   | Build a small interface to demonstrate evaluation results |

### Models we intend to evaluate

1. LiveKit Audio Turn Detector v1 or v1-mini.
2. Pipecat Smart Turn v3.2.
3. One or two additional systems, depending on compatibility and access.

VAD-only will also be included as a simple baseline.



# 4. What will the final output look like?

1\. GitHub repository

Reproducible evaluation scripts, model adapters, dataset documentation and results.

2\. Hinglish pilot dataset

100–200 labelled candidate pauses with annotation guidelines and consent.

3\. Comparative results

Model performance, false cutoffs, latency, language-wise analysis and failure cases.

4\. Research report

A clear technical report explaining what we tested, what we discovered and what remains unknown.

5\. Small evaluation demo

Play audio samples and compare model predictions against the human annotations.

# 5. Week 1 — Our immediate objective

We are NOT collecting Hinglish recordings or training anything this week.

We are establishing our experimental foundation.

| Task                  | Expected result                       |
| --------------------- | ------------------------------------- |
| Study EoT-Bench       | Understand its methodology            |
| Inspect its dataset   | Understand audio, labels and metadata |
| Set up the repository | Working project environment           |
| Run VAD baseline      | First baseline results                |
| Run Smart Turn        | First ML model results                |
| Run LiveKit adapter   | Second ML model results, if supported |
| Compare outputs       | Initial baseline report               |

Week 1 deliverable

A working evaluation pipeline that can take an audio example and produce comparable end-of-turn predictions from existing systems.

# 6. Our project boundaries

I want us to follow three rules:

1. Evidence before architecture. We don't build a new model until we know what existing models get wrong.
2. Finish before expanding. We complete the pilot before adding more datasets, languages or models.
3. Honest results. If existing models perform well on Hinglish, that is a valid finding too.

One additional distinction: the 100–200 example Hinglish dataset is our pilot contribution. We won't call it a representative benchmark.

## Final scope confirmation

TurnSense v1 = Evaluation + Benchmarking + Hinglish pilot dataset + Research report.

No new ML model. No fusion architecture. No full voice agent.

Scope locked for v1

Once we finish Week 1, we'll have actual baseline results to guide Week 2.

That's what we're building.