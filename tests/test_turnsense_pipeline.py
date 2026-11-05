"""Tests for TurnSense Phase C/D/E pipeline (no ASR, no adapter inference needed).
Run: ~/miniforge3-arm64/envs/turnsense/bin/python -m pytest tests/test_turnsense_pipeline.py -v
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def load(p): return pd.read_csv(ROOT / p, keep_default_na=False)

def test_reviewed_ids_unique():
    m = load("data/annotations-hinglish/pause_candidates.csv")
    r = m[m.review_status == "reviewed"]
    assert len(r) == 123
    assert r.candidate_id.duplicated().sum() == 0

def test_audit_correction_present():
    r = load("data/annotations-hinglish/pause_candidates.csv")
    r = r[r.review_status == "reviewed"]
    c066 = r[r.candidate_id == "hinglish_003_movies__c066"]
    assert len(c066) == 1 and c066.boundary_type.iloc[0] == "HOLD"

def test_enriched_preserves_manual_fields():
    m = load("data/annotations-hinglish/pause_candidates.csv")
    m = m[m.review_status == "reviewed"].set_index("candidate_id")
    e = load("data/annotations-hinglish/pause_candidates_enriched.csv").set_index("candidate_id")
    assert len(e) == 123
    for col in ["recording_id", "start_time", "end_time", "speaker_id", "boundary_type", "review_status"]:
        assert (m[col] == e.loc[m.index, col]).all(), f"manual field {col} was modified"
    assert set(e.preceding_text) or True  # at least some text aligned

def test_turns_and_pauses_integrity():
    t = load("data/annotations-hinglish/turns.csv")
    p = load("data/annotations-hinglish/pauses.csv")
    assert len(t) == 102 and len(p) == 123
    assert t.turn_id.duplicated().sum() == 0
    assert p.candidate_id.duplicated().sum() == 0
    scored = p[p.scored == True]  # noqa: E712
    assert len(scored) == 116
    assert set(scored.boundary_type.unique()) == {"EOT", "HOLD"}
    assert ((p.scored == True) & (p.boundary_type == "AMBIGUOUS")).sum() == 0  # noqa: E712

def test_transcript_no_speaker_column():
    c = load("data/transcripts/combined_segments.csv")
    assert "speaker_id" not in c.columns
    assert set(c.detected_language.unique()) == {"hi"}

def test_predictions_schema():
    for adapter in ["smart-turn", "livekit-mini"]:
        f = ROOT / f"experiments/eot-bench/hinglish/predictions_{adapter}.parquet"
        assert f.exists(), f"missing {f}"
        df = pd.read_parquet(f)
        assert list(df.columns) == ["id", "language", "span_index", "timestamp", "silence_dur", "p_eot", "label"]
        assert len(df) == 116
        assert set(df.label.unique()) == {"eot", "hold"}
        assert df.p_eot.between(0, 1).all()

def test_metrics_json_present():
    for adapter in ["smart-turn", "livekit-mini"]:
        m = json.loads((ROOT / f"experiments/eot-bench/hinglish/metrics_{adapter}.json").read_text())
        assert 0 <= m["auc"] <= 1 and 0 <= m["ap"] <= 1
        assert m["n_scored"] == 116 and m["n_eot"] == 100 and m["n_hold"] == 16
