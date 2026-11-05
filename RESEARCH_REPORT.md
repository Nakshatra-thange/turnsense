# TurnSense: Evaluating End-of-Turn Detection in Hindi and Hinglish Voice Conversations

## Abstract

End-of-turn (EOT) detection is a critical component of voice-agent systems. A system must determine when a speaker has finished speaking before generating a response. Premature detection causes interruptions, while delayed detection increases conversational latency.

TurnSense investigates how existing EOT detection systems perform on Hindi benchmark data and natural conversational Hinglish recordings.

We evaluated Smart Turn v3.2, LiveKit Turn Detector v1-mini, and a Silero VAD-based silence-duration baseline on Hindi validation data. We subsequently evaluated Smart Turn and LiveKit Mini on a manually annotated conversational Hinglish pilot.

The Hindi experiments showed stronger results, particularly for LiveKit Mini. However, performance was substantially weaker on the conversational Hinglish pilot. These findings motivate further investigation into the reliability of existing EOT systems in code-switched conversational settings.

## 1. Research question

How well do existing voice-agent end-of-turn detection systems identify when a speaker has finished speaking, particularly in Hindi-English code-switched conversations?

We distinguish three conversational states:

- **EOT:** The speaker has completed their contribution and yielded the floor.
- **HOLD:** The speaker pauses but intends to continue.
- **AMBIGUOUS:** The available context does not support a reliable decision.

A grammatical sentence boundary or silence interval alone does not establish EOT.

## 2. Experimental approach

The study was conducted in two stages.

**Stage 1 — Hindi benchmark evaluation**

Three approaches were evaluated using the EoT-Bench Hindi validation set:

- Smart Turn v3.2
- LiveKit Turn Detector v1-mini
- Silero VAD with a silence-duration policy

**Stage 2 — Conversational Hinglish evaluation**

Three natural conversational recordings were collected, covering general conversation, cricket, and movies.

A total of 123 pause candidates were manually reviewed. A targeted audit examined 31 originally EOT-labelled candidates, resulting in 30 confirmed EOT decisions and one correction to HOLD.

Transcription, timestamp alignment, and speaker-aware turn construction were performed before model evaluation.

## 3. Results

On Hindi validation data, LiveKit Mini achieved the strongest ranking performance, with an AUC of 0.8422 and AP of 0.7651.

On the conversational Hinglish pilot, performance was substantially weaker:

| Model | AUC | AP |
|---|---:|---:|
| Smart Turn v3.2 | 0.516 | 0.863 |
| LiveKit v1-mini | 0.450 | 0.871 |

The Hinglish evaluation contained 116 scored boundaries: 100 EOT and 16 HOLD.

At the selected 0.5 operating threshold, Smart Turn produced 4 false cutoffs out of 16 HOLD examples, while LiveKit Mini produced 11.

The high AP values must be interpreted against the high EOT prevalence of 86.2%. They do not independently demonstrate strong discrimination.

## 4. Key observations

1. Stronger Hindi benchmark performance did not translate into similarly strong performance on the conversational Hinglish pilot.
2. False cutoffs remain an important issue, particularly for LiveKit Mini in this evaluation.
3. EOT detection requires conversational context rather than relying exclusively on silence or grammatical completion.
4. Evaluation methodology and class distribution materially affect reported performance.

## 5. Limitations

This is an exploratory pilot study, not a production-scale evaluation.

- Only three conversational recordings were used.
- The conversational corpus has limited speaker diversity.
- Only 16 HOLD examples were included in the final binary evaluation.
- Pause candidates were generated through energy-based detection, which may miss other relevant boundaries.
- Language identification around code-switches is heuristic.
- The targeted annotation audit did not independently verify every label.
- Results are specific to the evaluation protocol and operating point.

## 6. Conclusion

TurnSense establishes an initial evaluation pipeline for studying EOT detection in conversational Hindi and Hinglish.

The observed difference between Hindi benchmark performance and conversational Hinglish performance suggests that existing EOT systems require further evaluation on natural code-switched speech.

Future work should expand the conversational corpus, improve annotation coverage, and investigate failure patterns around pauses, code-switching, hesitation, and speaker continuation.

The current results should be treated as a pilot baseline rather than evidence of general performance across Hinglish conversations.