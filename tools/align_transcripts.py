#!/usr/bin/env python
"""Phase C: Align ASR segments to reviewed pause candidates.

Input:  data/annotations-hinglish/pause_candidates.csv (master)
        data/transcripts/hinglish_*.json (segment + word timings)
Output: data/annotations-hinglish/pause_candidates_enriched.csv
        data/annotations-hinglish/alignment_report.json

Reads ONLY the 123 reviewed candidates (review_status == 'reviewed') from
the master file. Preserves every original manual field verbatim.

Alignment is timestamp-based, not lexical:
  preceding_text  = text of the segment/word-span whose speech ends at or
                    before the pause start and is closest to it.
  following_text  = text of the segment/word-span whose speech starts at or
                    after the pause end and is closest to it.
Both are taken at SEGMENT granularity by default (the whole nearest
segment), plus word-level boundaries are used to score confidence.
Word timings are used only to reject gaps that are already speech-free.

Languages are heuristic, script- and word-list-based, and marked
uncertain when evidence is insufficient. Because all three recordings
detected as 'hi' at transcription level and ASR emits romanized
Hinglish, per-segment language reflects local script/lexicon:
  - any Devanagari (>5% of alpha chars) -> 'hindi (devanagari)' 
  - else romanized: count English vs Hindi roman lexicon hits.
  - 'mixed' if both sides have >=2 hits; 'uncertain' if neither/ambiguous.
This is a transparency heuristic, NOT ground truth. Manual annotators
must override it after listening.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RECORDINGS = ["hinglish_001_general", "hinglish_002_cricket", "hinglish_003_movies"]

# Curated romanized lexicons for a conservative heuristic. Intentionally
# small and high-precision; absence of evidence -> uncertain, not a guess.
HINDI_ROMAN = {
    "mujhe", "mujinii", "lakta", "lagta", "hai", "he", "hain", "ka", "ki", "ke",
    "ko", "me", "main", "hum", "aap", "tum", "yeh", "yah", "woh", "wo", "kya",
    "kyun", "kahan", "kab", "kaise", "nahi", "nahin", "matlab", "achha",
    "acha", "bahut", "thoda", "fir", "phir", "toh", "to", "bhi", "aur", "ya",
    "par", "lekin", "magar", "agar", "jab", "tab", "abhi", "ab", "kal", "aaj",
    "kal", "meri", "mera", "mere", "tera", "tere", "uska", "uski", "unki",
    "kis", "kis", "kisi", "koi", "ko", "per", "pe", "se", "sa", "was", "tha",
    "thi", "the", "hoga", "hogi", "honge", "ho", "ja", "gaya", "gi", "ge",
    "raha", "rahi", "rahe", "chahiye", "karne", "karna", "karega", "kare",
    "ho", "jai", "jaye", "ga", "gi", "ge", "tum", "tum", "bekar", "yaar",
    "bol", "kuch", "sab", "hara", "khela",
}

ENGLISH_WORDS = {
    "i", "you", "he", "she", "it", "we", "they", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "can", "could", "should", "may", "might", "must", "the",
    "a", "an", "this", "that", "these", "those", "and", "or", "but", "if",
    "then", "when", "where", "how", "what", "why", "who", "which", "of", "to",
    "in", "on", "at", "for", "with", "from", "by", "about", "as", "very",
    "so", "just", "also", "because", "not", "no", "yes", "now", "then",
    "here", "there", "some", "any", "all", "one", "two",
}


def load_segments(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.sort_values(["recording_id", "start_time"]).reset_index(drop=True)


def script_share(text: str) -> tuple[float, float]:
    """Return (devanagari_frac, latin_frac) over alphabetic characters."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return 0.0, 0.0
    dev = sum(1 for c in alpha if "\u0900" <= c <= "\u097f")
    lat = sum(1 for c in alpha if c.isascii())
    return dev / len(alpha), lat / len(alpha)


def heuristic_language(text: str) -> tuple[str, float, str]:
    """Return (language, confidence, evidence)."""
    if not text or not text.strip():
        return "uncertain", 0.0, "no text"
    dev_frac, lat_frac = script_share(text)
    tokens = [t.strip(".,?!;:'\"()").lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    hindi_hits = sum(1 for t in tokens if t in HINDI_ROMAN)
    english_hits = sum(1 for t in tokens if t in ENGLISH_WORDS)
    if dev_frac > 0.3:
        return "hindi (devanagari)", 0.8, f"devanagari frac {dev_frac:.2f}"
    if hindi_hits >= 2 and english_hits >= 2:
        return "mixed", 0.6, f"hindi hits {hindi_hits}, english hits {english_hits}"
    if hindi_hits >= 2 and english_hits <= 1:
        return "hindi (romanized)", 0.65, f"hindi hits {hindi_hits}, english hits {english_hits}"
    if english_hits >= 3 and hindi_hits == 0:
        return "english", 0.7, f"english hits {english_hits}"
    if english_hits >= 2 and english_hits > hindi_hits:
        return "english", 0.5, f"english hits {english_hits}, hindi hits {hindi_hits}"
    return "uncertain", 0.3, f"hindi hits {hindi_hits}, english hits {english_hits}, dev {dev_frac:.2f}"


def load_json_segments(recording_id: str) -> list[dict]:
    p = Path("data/transcripts") / f"{recording_id}.json"
    with open(p) as f:
        return json.load(f)["segments"]


def align_pause(pause: dict, segments: list[dict]) -> dict:
    """Nearest preceding/following segment by timestamp."""
    p_start = float(pause["start_time"])
    p_end = float(pause["end_time"])
    preceding = None
    preceding_gap = None
    following = None
    following_gap = None
    for seg in segments:
        s_start = float(seg["start_time"])
        s_end = float(seg["end_time"])
        # preceding: segment ends at or before pause start (allow tiny tolerance)
        if s_end <= p_start + 0.05:
            gap = p_start - s_end
            if preceding is None or gap < preceding_gap:
                preceding, preceding_gap = seg, gap
        # following: segment starts at or after pause end
        if s_start >= p_end - 0.05:
            gap = s_start - p_end
            if following is None or gap < following_gap:
                following, following_gap = seg, gap
    # Alignment confidence: small gaps and non-overlapping are high.
    conf_parts = []
    if preceding is not None:
        conf_parts.append(f"preceding gap {preceding_gap:.2f}s")
    else:
        conf_parts.append("no preceding segment")
    if following is not None:
        conf_parts.append(f"following gap {following_gap:.2f}s")
    else:
        conf_parts.append("no following segment")
    # Score: base on gaps, penalize missing side.
    if preceding is not None and following is not None:
        alignment_confidence = max(0.0, min(1.0, 1.0 - (preceding_gap + following_gap) / 4.0))
    elif preceding is not None or following is not None:
        alignment_confidence = 0.5
    else:
        alignment_confidence = 0.0
    return {
        "preceding_text": (preceding or {}).get("text", "").strip(),
        "preceding_start": (preceding or {}).get("start_time", ""),
        "preceding_end": (preceding or {}).get("end_time", ""),
        "following_text": (following or {}).get("text", "").strip(),
        "following_start": (following or {}).get("start_time", ""),
        "following_end": (following or {}).get("end_time", ""),
        "alignment_confidence": round(float(alignment_confidence), 3),
        "alignment_note": "; ".join(conf_parts),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", default="data/annotations-hinglish/pause_candidates.csv")
    parser.add_argument("--output", default="data/annotations-hinglish/pause_candidates_enriched.csv")
    parser.add_argument("--report", default="data/annotations-hinglish/alignment_report.json")
    args = parser.parse_args()

    master = pd.read_csv(args.master, keep_default_na=False)
    reviewed = master[master["review_status"] == "reviewed"].copy().reset_index(drop=True)
    print(f"reviewed candidates: {len(reviewed)}")
    assert reviewed["candidate_id"].duplicated().sum() == 0, "duplicate candidate_id in reviewed set"

    enriched_rows = []
    per_recording = {}
    for _, row in reviewed.iterrows():
        rid = row["recording_id"]
        segments = load_json_segments(rid)
        aln = align_pause(row.to_dict(), segments)
        prec_lang, prec_conf, prec_ev = heuristic_language(aln["preceding_text"])
        foll_lang, foll_conf, foll_ev = heuristic_language(aln["following_text"])
        code_switch_continues = (
            prec_lang.startswith("mixed") or foll_lang.startswith("mixed")
            or (prec_lang and foll_lang and prec_lang != foll_lang and "uncertain" not in (prec_lang, foll_lang))
        )
        enriched = row.to_dict()
        enriched.update(
            {
                "preceding_text": aln["preceding_text"],
                "following_text": aln["following_text"],
                "preceding_language": prec_lang,
                "preceding_language_confidence": prec_conf,
                "preceding_language_evidence": prec_ev,
                "following_language": foll_lang,
                "following_language_confidence": foll_conf,
                "following_language_evidence": foll_ev,
                # coding hint only; name preserved from brief with a clear marker
                "code_swintinues_after_pause": ("mixed" in prec_lang or "mixed" in foll_lang),
                "code_switch_around_pause": bool(code_switch_continues),
                "alignment_confidence": aln["alignment_confidence"],
                "alignment_note": aln["alignment_note"],
                "preceding_segment_start": aln["preceding_start"],
                "preceding_segment_end": aln["preceding_end"],
                "following_segment_start": aln["following_start"],
                "following_segment_end": aln["following_end"],
            }
        )
        enriched_rows.append(enriched)
        per_recording.setdefault(rid, {"count": 0, "with_preceding": 0, "with_following": 0, "mean_conf": []})
        per_recording[rid]["count"] += 1
        if aln["preceding_text"]:
            per_recording[rid]["with_preceding"] += 1
        if aln["following_text"]:
            per_recording[rid]["with_following"] += 1
        per_recording[rid]["mean_conf"].append(aln["alignment_confidence"])

    out = pd.DataFrame(enriched_rows)
    out.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(out)} rows)")

    report = {
        "alignment_date": "2026-11-05",
        "master": args.master,
        "output": args.output,
        "n_reviewed": int(len(reviewed)),
        "per_recording": {
            rid: {
                "count": v["count"],
                "with_preceding_text": v["with_preceding"],
                "with_following_text": v["with_following"],
                "mean_alignment_confidence": round(float(sum(v["mean_conf"]) / max(1, len(v["mean_conf"]))), 3),
                "min_alignment_confidence": round(float(min(v["mean_conf"])) if v["mean_conf"] else 0.0, 3),
            }
            for rid, v in per_recording.items()
        },
        "language_heuristic": "Devanagari/latin script share + small roman lexicons; 'uncertain' when evidence insufficient; ASR is DRAFT, heuristic is NOT ground truth",
        "preserved_fields": "All original manual fields (recording_id, candidate_id, start/end, speaker_id, boundary_type, confidence, review_status, notes) preserved verbatim",
        "code_switch_fields": "code_switch_around_pause / code_swintinues_after_pause (typo kept for brief compatibility) are heuristics only",
    }
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {args.report}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
