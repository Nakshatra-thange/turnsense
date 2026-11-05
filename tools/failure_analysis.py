#!/usr/bin/env python
"""Phase 1 failure analysis: build reports/failure-analysis/failure_cases.csv.

Reads ONLY saved artifacts (no re-scoring, no re-training):
  data/annotations-hinglish/pauses.csv  (manual ground truth + ASR context)
  experiments/eot-bench/hinglish/scored_pauses_*.csv  (onset scores)
  experiments/eot-bench/hinglish/scored_pauses_*_at-0.2s.csv (timing variant)

For each scored pause (116), classifies model behavior at threshold 0.5
into the 8 required categories. Linguistic association labels use
evidence-backed heuristics only (explicit text patterns) and default to
`unclassified` where evidence is insufficient. Confidence ratings are
conservative (low/medium) because ASR is draft and N is small.

Outputs:
  reports/failure-analysis/failure_cases.csv  (116 rows)
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

FILLER_PATTERNS = [
    r"\bum+m?\b", r"\buh+\b", r"\bmm+\b", r"\bhmm\b", r"\bha+a+\b", r"\bhaan\b",
    r"\bokay\b", r"\bok\b", r"\bbut\b", r"\bso\b", r"\byou know\b", r"\blike\b",
    r"\b matlab\b", r"\bmujhe\b", r"\bwoh\b", r"\bna\b",
]
QUESTION_RE = re.compile(r"\?\s*$")
BACKCHANNEL_RE = re.compile(
    r"^(?:haan|haa|okay|ok|mm|mhm|acha|achha|hmm|yes|right|exactly|sure|true)\b",
    re.IGNORECASE,
)
HESITATION_RE = re.compile(r"\b(um|uh|erm|mm+|hmm+)\b|\.\.\.", re.IGNORECASE)


def has_filler(text: str) -> bool:
    t = f" {text.lower()} "
    return any(re.search(p, t) for p in FILLER_PATTERNS)


def classify(row: dict) -> tuple[str, str, str]:
    """Return (primary_category, evidence, confidence)."""
    gt = row["ground_truth"]
    code = bool(str(row.get("code_switch_around_pause", "")).lower() in ("true", "1"))
    prec_text = (row.get("preceding_text") or "").strip()
    foll_text = (row.get("following_text") or "").strip()
    dur = row.get("duration", "")
    # Filler/hesitation evidence (observed in draft ASR text, not audio).
    filler = has_filler(prec_text) or bool(HESITATION_RE.search(prec_text))
    is_question = bool(QUESTION_RE.search(prec_text))
    is_backchannel = bool(BACKCHANNEL_RE.match(prec_text.strip())) and len(prec_text.split()) <= 4
    # Score dynamics
    smart, live = row["smart_score"], row["live_score"]
    smart_pred, live_pred = row["smart_pred"], row["live_pred"]
    # Instability: models disagree OR scores near 0.5 (within 0.1) on either model.
    near_boundary = abs(smart - 0.5) < 0.1 or abs(live - 0.5) < 0.1
    disagree = smart_pred != live_pred

    if gt == "hold" and smart_pred == "eot" and live_pred == "eot":
        # Both false cutoff.
        if is_backchannel:
            return ("false_cutoff", f"both predict eot on HOLD; preceding is backchannel-like: {prec_text[:60]!r}", "medium")
        if filler:
            return ("false_cutoff", f"both predict eot on HOLD; filler/hesitation in preceding: {prec_text[:60]!r}", "medium")
        if code:
            return ("false_cutoff", f"both predict eot on HOLD; code-switch flagged (heuristic): preceding {prec_text[:60]!r}", "low")
        return ("false_cutoff", f"both predict eot on HOLD; preceding {prec_text[:60]!r}; no surface marker", "low")
    if gt == "hold" and (smart_pred == "eot" or live_pred == "eot"):
        which = "Smart Turn" if smart_pred == "eot" and live_pred != "eot" else ("LiveKit" if live_pred == "eot" and smart_pred != "eot" else "both")
        if is_backchannel:
            return ("false_cutoff", f"{which} predicts eot on HOLD; backchannel-like preceding: {prec_text[:60]!r}", "medium")
        if filler:
            return ("false_cutoff", f"{which} predicts eot on HOLD; filler in preceding: {prec_text[:60]!r}", "medium")
        if code:
            return ("false_cutoff", f"{which} predicts eot on HOLD; code-switch flagged: {prec_text[:60]!r}", "low")
        return ("false_cutoff", f"{which} predicts eot on HOLD; speaker continuation per following text", "low")
    if gt == "eot" and smart_pred == "hold" and live_pred == "hold":
        if is_question:
            return ("missed_eot", f"both predict hold on EOT; preceding is question-form: {prec_text[:60]!r}", "medium")
        if filler:
            return ("missed_eot", f"both predict hold on EOT; filler/discount markers in preceding: {prec_text[:60]!r}", "medium")
        if code:
            return ("missed_eot", f"both predict hold on EOT; code-switch flagged: {prec_text[:60]!r}", "low")
        return ("missed_eot", f"both predict hold on EOT; preceding {prec_text[:60]!r}", "low")
    if gt == "eot" and (smart_pred == "hold" or live_pred == "hold"):
        which = "Smart Turn" if smart_pred == "hold" and live_pred != "hold" else ("LiveKit" if live_pred == "hold" and smart_pred != "hold" else "both")
        if is_question:
            return ("missed_eot", f"{which} predicts hold on EOT; question-form preceding: {prec_text[:60]!r}", "medium")
        if code:
            return ("missed_eot", f"{which} predicts hold on EOT; code-switch flagged: {prec_text[:60]!r}", "low")
        return ("missed_eot", f"{which} predicts hold on EOT", "low")
    if near_boundary or disagree:
        return ("unstable_prediction", f"near-boundary/disagree: ST={smart:.2f} LK={live:.2f}", "medium")
    if gt == "eot" and smart_pred == "eot" and live_pred == "eot":
        # Detect late? Can't measure without trace; classify correct as unclassified
        # unless code-switch present as association note.
        if code:
            return ("possible_code_switch_association", f"both correct EOT; code-switch flagged near pause", "low")
        return ("unclassified", "both correct; no failure evidence", "low")
    return ("unclassified", f"GT={gt} ST={smart_pred} LK={live_pred}", "low")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pauses", default="data/annotations-hinglish/pauses.csv")
    parser.add_argument("--smart", default="experiments/eot-bench/hinglish/scored_pauses_smart-turn.csv")
    parser.add_argument("--live", default="experiments/eot-bench/hinglish/scored_pauses_livekit-mini.csv")
    parser.add_argument("--smart02", default="experiments/eot-bench/hinglish/scored_pauses_smart-turn_at-0.2s.csv")
    parser.add_argument("--live02", default="experiments/eot-bench/hinglish/scored_pauses_livekit-mini_at-0.2s.csv")
    parser.add_argument("--output", default="reports/failure-analysis/failure_cases.csv")
    args = parser.parse_args()

    pauses = pd.read_csv(args.pauses, keep_default_na=False)
    scored = pauses[pauses["scored"] == True].copy()  # noqa: E712
    smart = pd.read_csv(args.smart, keep_default_na=False)
    live = pd.read_csv(args.live, keep_default_na=False)
    smart02 = pd.read_csv(args.smart02, keep_default_na=False)
    live02 = pd.read_csv(args.live02, keep_default_na=False)

    df = scored.copy()
    df["ground_truth"] = (df["boundary_type"] == "EOT").map({True: "eot", False: "hold"})
    df = df.merge(smart[["candidate_id", "p_eot"]].rename(columns={"p_eot": "smart_score"}), on="candidate_id", how="left", validate="one_to_one")
    df = df.merge(live[["candidate_id", "p_eot"]].rename(columns={"p_eot": "live_score"}), on="candidate_id", how="left", validate="one_to_one")
    df = df.merge(smart02[["candidate_id", "p_eot"]].rename(columns={"p_eot": "smart_score_0.2s"}), on="candidate_id", how="left", validate="one_to_one")
    df = df.merge(live02[["candidate_id", "p_eot"]].rename(columns={"p_eot": "live_score_0.2s"}), on="candidate_id", how="left", validate="one_to_one")
    assert len(df) == 116 and df["smart_score"].notna().all() and df["live_score"].notna().all()

    df["smart_pred"] = (df["smart_score"] > 0.5).map({True: "eot", False: "hold"})
    df["live_pred"] = (df["live_score"] > 0.5).map({True: "eot", False: "hold"})
    df["model_agreement"] = (df["smart_pred"] == df["live_pred"]).map({True: "agree", False: "disagree"})
    df["score_diff_abs"] = (df["smart_score"] - df["live_score"]).abs().round(4)
    df["smart_pred_0.2s"] = (df["smart_score_0.2s"] > 0.5).map({True: "eot", False: "hold"})
    df["live_pred_0.2s"] = (df["live_score_0.2s"] > 0.5).map({True: "eot", False: "hold"})

    rows = []
    for _, r in df.iterrows():
        cat, ev, conf = classify(r.to_dict())
        # Override: delayed_eot / unstable need care. delayed requires trace; we have no
        # 0.2s score crossing data yet. Use 0.2s variant as a proxy: if at onset it's
        # hold but at +0.2s it's eot (EOT gt), that's delayed-by-silence evidence.
        if r["ground_truth"] == "eot":
            delayed_smart = (r["smart_pred"] == "hold") and (r["smart_pred_0.2s"] == "eot")
            delayed_live = (r["live_pred"] == "hold") and (r["live_pred_0.2s"] == "eot")
            if cat.startswith("missed"):
                if delayed_smart and delayed_live:
                    cat = "delayed_eot"
                    ev = f"both hold at onset, eot at +0.2s: {ev}"
                    conf = "medium"
                elif delayed_smart or delayed_live:
                    which = "Smart Turn" if delayed_smart else "LiveKit"
                    cat = "delayed_eot"
                    ev = f"{which} hold at onset, eot at +0.2s: {ev}"
                    conf = "medium"
            if cat.startswith("unstable") or True:
                pass
        # Language/code-switch secondary association flags (kept separate from primary).
        prec_text = (r.get("preceding_text") or "")
        lang = r.get("preceding_language", "uncertain")
        cs = str(r.get("code_switch_around_pause", "")).lower() in ("true", "1")
        lang_note = f"lang={lang}; code_switch_heuristic={cs}"
        broader = []
        if has_filler(prec_text):
            broader.append("possible_filler_or_hesitation_association")
        if bool(BACKCHANNEL_RE.match(prec_text.strip())) and len(prec_text.split()) <= 4:
            broader.append("possible_backchannel_or_handoff_association")
        rows.append(
            {
                "recording_id": r["recording_id"],
                "candidate_id": r["candidate_id"],
                "turn_id": r["turn_id"],
                "start_time": r["start_time"],
                "end_time": r["end_time"],
                "duration": r["duration"],
                "ground_truth": r["ground_truth"],
                "smart_score": round(float(r["smart_score"]), 4),
                "smart_prediction_0.5": r["smart_pred"],
                "live_score": round(float(r["live_score"]), 4),
                "live_prediction_0.5": r["live_pred"],
                "model_agreement": r["model_agreement"],
                "score_diff_abs": r["score_diff_abs"],
                "smart_score_0.2s": round(float(r["smart_score_0.2s"]), 4),
                "live_score_0.2s": round(float(r["live_score_0.2s"]), 4),
                "preceding_transcript": prec_text,
                "following_transcript": r.get("following_text", ""),
                "preceding_language": lang,
                "following_language": r.get("following_language", "uncertain"),
                "code_switch_around_pause": cs,
                "speaker_id": r.get("speaker_id", ""),
                "failure_category": cat,
                "evidence_rationale": f"{ev} | {lang_note}",
                "confidence": conf,
                "secondary_associations": ";".join(broader) if broader else "",
            }
        )

    out = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} rows)")
    print(out["failure_category"].value_counts().to_string())
    # Verification counts
    print("\n--- verification @0.5 (onset) ---")
    df["smart_pred"] = (df["smart_score"] > 0.5).map({True: "eot", False: "hold"})
    df["live_pred"] = (df["live_score"] > 0.5).map({True: "eot", False: "hold"})
    print("Smart Turn TP/FN/FP/TN:", ((df.ground_truth == "eot") & (df.smart_pred == "eot")).sum(),
          ((df.ground_truth == "eot") & (df.smart_pred == "hold")).sum(),
          ((df.ground_truth == "hold") & (df.smart_pred == "eot")).sum(),
          ((df.ground_truth == "hold") & (df.smart_pred == "hold")).sum())
    print("LiveKit TP/FN/FP/TN:", ((df.ground_truth == "eot") & (df.live_pred == "eot")).sum(),
          ((df.ground_truth == "eot") & (df.live_pred == "hold")).sum(),
          ((df.ground_truth == "hold") & (df.live_pred == "eot")).sum(),
          ((df.ground_truth == "hold") & (df.live_pred == "hold")).sum())
    print("both wrong:", ((df.smart_pred != df.ground_truth) & (df.live_pred != df.ground_truth)).sum())


if __name__ == "__main__":
    main()
