#!/usr/bin/env python
"""Phase D: Build a speaker-aware conversational turn dataset.

Input:  data/annotations-hinglish/pause_candidates_enriched.csv (123 reviewed)
        data/transcripts/hinglish_*.json (for segment text provenance only)
Output: data/annotations-hinglish/turns.csv
        data/annotations-hinglish/pauses.csv
        data/annotations-hinglish/dataset_manifest.json
        data/annotations-hinglish/evaluation_protocol.md

Turn definition (no future leakage, no invented labels):
  - Iterate reviewed candidates in timestamp order per recording.
  - A turn accumulates pauses until it hits an EOT-labelled pause, which
    CLOSES the turn. Everything before that pause and including the
    pause belongs to the closing turn.
  - HOLD pauses are internal to the open turn (speaker continues).
  - AMBIGUOUS c000 (initial silence, no speaker) opens no turn and is
    emitted as a separate 'pre_conversation' unit, never scored.
  - If a segment ENDS with an unclosed turn at recording end (no EOT),
    it is closed at the last pause and flagged 'incomplete_turn_end'.

Each resulting turn row contains:
  turn_id (recording_id__tNNN), recording_id, speaker_id (of the EOT
  pause that closed it, or 'uncertain' if the turn never closed),
  start_time (first pause start or preceding segment start),
  end_time (closing pause end), transcript (ASR preceding_text of
  closing pause ## fallback: joined preceding texts),
  language (preceding_language of closing pause),
  provenance (manual reviewed + ASR draft, human audio), and
  uncertainty (alignment confidence, speaker consistency).

The pauses table carries per-pause label + turn assignment so that
evaluation can use HOLD/EOT directly and EXCLUDE AMBIGUOUS.
Neither table is fed verbatim into EoT-Bench (its schema cannot hold
speaker_id or per-span labels) — the bridge for scoring is documented
in evaluation_protocol.md.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RECORDINGS = ["hinglish_001_general", "hinglish_002_cricket", "hinglish_003_movies"]


def build(recording_id: str, enriched: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    sub = enriched[enriched["recording_id"] == recording_id].copy()
    sub = sub.sort_values("start_time").reset_index(drop=True)
    turns: list[dict] = []
    pauses: list[dict] = []
    turn_idx = 0
    open_pauses: list[dict] = []  # candidate rows in the currently-open turn
    # Pre-conversation handling: c000 AMBIGUOUS has no speaker. Any pause
    # with empty speaker_id before the first real pause is pre-conversation.
    for _, row in sub.iterrows():
        cand = row["candidate_id"]
        speaker = (row["speaker_id"] or "").strip()
        boundary = row["boundary_type"].strip()
        # Pre-conversation silences: empty speaker -> not a conversational pause.
        if not speaker:
            pauses.append(
                {
                    "candidate_id": cand,
                    "recording_id": recording_id,
                    "start_time": row["start_time"],
                    "end_time": row["end_time"],
                    "duration": row["duration"],
                    "speaker_id": "",
                    "boundary_type": boundary,
                    "turn_id": "pre_conversation",
                    "position_in_turn": "pre_conversation",
                    "scored": False,
                    "preceding_text": row.get("preceding_text", ""),
                    "following_text": row.get("following_text", ""),
                    "preceding_language": row.get("preceding_language", "uncertain"),
                    "following_language": row.get("following_language", "uncertain"),
                    "alignment_confidence": row.get("alignment_confidence", 0.0),
                }
            )
            continue
        # Open-turn accumulation.
        open_pauses.append(row)
        if boundary == "EOT":
            turn_id = f"{recording_id}__t{turn_idx:03d}"
            # Speaker consistency: all pauses in the turn SHOULD share the
            # closing speaker (HOLDs keep speaker, EOT yields). Collect distinct.
            speakers = sorted({(p["speaker_id"] or "").strip() for p in open_pauses if (p["speaker_id"] or "").strip()})
            closing = open_pauses[-1]
            # Transcript: closing pause's preceding_text (draft ASR); fall back to join.
            transcript = (closing.get("preceding_text", "") or "").strip()
            if not transcript and open_pauses:
                transcript = " ".join(
                    (p.get("preceding_text", "") or "").strip() for p in open_pauses if (p.get("preceding_text", "") or "").strip()
                ).strip()
            turns.append(
                {
                    "turn_id": turn_id,
                    "recording_id": recording_id,
                    "speaker_id": closing["speaker_id"],
                    "start_time": open_pauses[0]["start_time"],
                    "end_time": closing["end_time"],
                    "transcript": transcript,
                    "language": closing.get("preceding_language", "uncertain"),
                    "preceding_text": closing.get("preceding_text", ""),
                    "following_text": closing.get("following_text", ""),
                    "following_language": closing.get("following_language", "uncertain"),
                    "code_switch_around_pause": closing.get("code_switch_around_pause", False),
                    "n_pauses": len(open_pauses),
                    "n_hold": sum(1 for p in open_pauses if p["boundary_type"] == "HOLD"),
                    "speakers_in_turn": ";".join(speakers),
                    "uncertainty": f"speaker_consistent={len(speakers) == 1}; alignment_conf={closing.get('alignment_confidence', 0.0)}",
                    "provenance": "manual reviewed labels + faster-whisper base draft ASR; human audio",
                    "turn_complete": True,
                }
            )
            # Emit pause rows for this turn (internal HOLDs + final EOT).
            for p in open_pauses:
                pos = "final" if p["candidate_id"] == cand else "internal"
                pauses.append(
                    {
                        "candidate_id": p["candidate_id"],
                        "recording_id": recording_id,
                        "start_time": p["start_time"],
                        "end_time": p["end_time"],
                        "duration": p["duration"],
                        "speaker_id": p["speaker_id"],
                        "boundary_type": p["boundary_type"],
                        "turn_id": turn_id,
                        "position_in_turn": pos,
                        "scored": True,
                        "preceding_text": p.get("preceding_text", ""),
                        "following_text": p.get("following_text", ""),
                        "preceding_language": p.get("preceding_language", "uncertain"),
                        "following_language": p.get("following_language", "uncertain"),
                        "code_switch_around_pause": p.get("code_switch_around_pause", False),
                        "alignment_confidence": p.get("alignment_confidence", 0.0),
                    }
                )
            open_pauses = []
            turn_idx += 1
    # Incomplete tail: open pauses without a closing EOT.
    if open_pauses:
        turn_id = f"{recording_id}__t{turn_idx:03d}"
        closing = open_pauses[-1]
        speakers = sorted({(p["speaker_id"] or "").strip() for p in open_pauses if (p["speaker_id"] or "").strip()})
        transcript = (closing.get("preceding_text", "") or "").strip()
        turns.append(
            {
                "turn_id": turn_id,
                "recording_id": recording_id,
                "speaker_id": closing["speaker_id"],
                "start_time": open_pauses[0]["start_time"],
                "end_time": closing["end_time"],
                "transcript": transcript,
                "language": closing.get("preceding_language", "uncertain"),
                "preceding_text": closing.get("preceding_text", ""),
                "following_text": closing.get("following_text", ""),
                "following_language": closing.get("following_language", "uncertain"),
                "code_switch_around_pause": closing.get("code_switch_around_pause", False),
                "n_pauses": len(open_pauses),
                "n_hold": sum(1 for p in open_pauses if p["boundary_type"] == "HOLD"),
                "speakers_in_turn": ";".join(speakers),
                "uncertainty": "incomplete_turn_end; no closing EOT; turns ends at recording end",
                "provenance": "manual reviewed labels + faster-whisper base draft ASR; human audio",
                "turn_complete": False,
            }
        )
        for p in open_pauses:
            pauses.append(
                {
                    "candidate_id": p["candidate_id"],
                    "recording_id": recording_id,
                    "start_time": p["start_time"],
                    "end_time": p["end_time"],
                    "duration": p["duration"],
                    "speaker_id": p["speaker_id"],
                    "boundary_type": p["boundary_type"],
                    "turn_id": turn_id,
                    "position_in_turn": "uncLOSED_tail",
                    "scored": False,
                    "preceding_text": p.get("preceding_text", ""),
                    "following_text": p.get("following_text", ""),
                    "preceding_language": p.get("preceding_language", "uncertain"),
                    "following_language": p.get("following_language", "uncertain"),
                    "code_switch_around_pause": p.get("code_switch_around_pause", False),
                    "alignment_confidence": p.get("alignment_confidence", 0.0),
                }
            )
    return turns, pauses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enriched", default="data/annotations-hinglish/pause_candidates_enriched.csv")
    parser.add_argument("--turns-out", default="data/annotations-hinglish/turns.csv")
    parser.add_argument("--pauses-out", default="data/annotations-hinglish/pauses.csv")
    parser.add_argument("--manifest-out", default="data/annotations-hinglish/dataset_manifest.json")
    args = parser.parse_args()

    enriched = pd.read_csv(args.enriched, keep_default_na=False)
    all_turns, all_pauses = [], []
    for rid in RECORDINGS:
        t, p = build(rid, enriched)
        all_turns.extend(t)
        all_pauses.extend(p)

    turns_df = pd.DataFrame(all_turns)
    pauses_df = pd.DataFrame(all_pauses)
    turns_df.to_csv(args.turns_out, index=False)
    pauses_df.to_csv(args.pauses_out, index=False)
    print(f"wrote {args.turns_out} ({len(turns_df)} turns)")
    print(f"wrote {args.pauses_out} ({len(pauses_df)} pauses)")

    # Validations.
    assert turns_df["turn_id"].duplicated().sum() == 0, "duplicate turn_id"
    assert pauses_df["candidate_id"].duplicated().sum() == 0, "duplicate candidate_id"
    scored = pauses_df[pauses_df["scored"] == True]  # noqa: E712
    print(f"scored pauses: {len(scored)} (expected 122 = 123 - 1 pre_conversation c000)")
    print(f"  EOT scored: {(scored['boundary_type'] == 'EOT').sum()}  HOLD scored: {(scored['boundary_type'] == 'HOLD').sum()}  "
          f"AMBIGUOUS scored: {(scored['boundary_type'] == 'AMBIGUOUS').sum()}")
    unscored = pauses_df[pauses_df["scored"] == False]  # noqa: E712
    print(f"unscored pauses: {len(unscored)} -> {unscored.groupby(['turn_id', 'position_in_turn']).size().to_dict()}")
    print(f"turns complete: {turns_df['turn_complete'].value_counts().to_dict()}")

    manifest = {
        "dataset_date": "2026-11-05",
        "turns": int(len(turns_df)),
        "pauses": int(len(pauses_df)),
        "scored_pauses": int(len(scored)),
        "scored_eot": int((scored["boundary_type"] == "EOT").sum()),
        "scored_hold": int((scored["boundary_type"] == "HOLD").sum()),
        "turn_definition": "A turn ENDS at an EOT-labelled pause; HOLDs are internal; pre_conversation and unclosed tails are unscored",
        "speaker_handling": "speaker_id is the EOT pause's manual speaker; turns with >1 distinct speaker are flagged in speakers_in_turn",
        "transcript_source": "faster-whisper base draft ASR preceding_text (romanized Hinglish, verbatim, not ground truth)",
        "excluded_from_scoring": "pre_conversation c000 (initial silence) + any unclosed tail; AMBIGUOUS never scored",
        "future_leakage": "Model inputs must be causal audio prefixes ONLY up to each pause timestamp; following_text is for analysis, NOT model input",
        "comparability_caveats": [
            "Hinglish results measure conversational EOT, not utterance-final pauses; not directly comparable to Hindi utterance-final benchmark without caveat",
            "Energy-detector candidate generation biases toward pauses above -45 dBFS / >=0.2s; recall for very short pauses is not evaluated",
            "Two speakers, three recordings, one research team — not a representative Hinglish population",
            "ASR + language heuristic are draft/provenance only; labels remain MANUAL",
        ],
    }
    with open(args.manifest_out, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {args.manifest_out}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
