#!/usr/bin/env python
"""Phase E: Score the 116 reviewed Hinglish pause candidates with existing adapters.

Usage:
  ~/miniforge3-arm64/envs/turnsense/bin/python tools/score_hinglish.py --adapter smart-turn
  ~/miniforge3-arm64/envs/turnsense/bin/python tools/score_hinglish.py --adapter livekit-mini

What it does:
  For each scored pause (pauses.csv, scored=True, boundary in {EOT,HOLD}):
    - Loads the source WAV, slices the causal prefix [0, pause.start_time]
      (audio array + sample rate), builds EoT-Bench messages the same way
      io.build_messages does (words visible up to t - 0.5s lag from ASR).
    - Calls adapter.predict_batch (EXISTING vendored adapters, unchanged).
    - Stores one p_eot per pause in predictions form identical to
      experiments/eot-bench/.../predictions.parquet:
        {id=turn_id? -> candidate-based id, language, span_index,
         timestamp, silence_dur=0.0, p_eot, label='eot'|'hold'}
      For Hinglish, `id` is the pause's candidate_id (stable and unique),
      `span_index` is 0 (one scored pause per row), timestamp = pause start,
      silence_dur = 0.0 (we score CAUSALLY AT PAUSE ONSET — the model sees
      zero silence yet; see protocol §4). This is the causally cleanest
      single-point protocol and must be documented as such.
  Then computes metrics via the harness's own metrics functions
  (official_score_table / compute_scalar_metrics) with score_point_s=0.0
  — but since we score at onset, we do NOT use the harness's score_point
  machinery for AUC; we use compute_classification_metrics directly on
  the one-score-per-pause table, and compute the policy sweep manually
  for frontier latency (action_delay/timeout). See metrics section.

IMPORTANT — protocol decision (causal onset scoring):
  The Hindi benchmark scores model p(eot) at silence_dur == 0.2s INTO each
  span (score_point_s=0.2). Our protocol scores at pause ONSET
  (silence_dur=0.0), i.e. the instant the speaker stops, BEFORE any silence
  is observed. This is stricter and more operationally honest for
  conversational EOT (a deployed system must decide from speech + prosody,
  not from 200ms of ensuing silence). CONSEQUENCE: Hinglish AUC/AP are
  NOT directly comparable to Hindi AUC/AP within ~0.05 due to the score-point
  offset alone. All cross-dataset comparisons in Phase F must carry this caveat.

Adapters are EXISTING vendored ones (no retraining, no quota retries):
  smart-turn: eot_harness.smart_turn_adapter:SmartTurnAudioAdapter
             (pipecat-ai/smart-turn-v3.2-gpu.onnx, 8s window, 16kHz)
  livekit-mini: eot_harness.livekit_turn_detector_mini_adapter:
             LiveKitTurnDetectorMiniAdapter (local EOT, 1.2s window)
  (Silero baseline is NOT run here — see note: it is a silence-duration
   policy, not a learned model; its Hindi result is retained as context.
   Running it on Hinglish as a silence baseline is optional Phase F work.)

Output:
  experiments/eot-bench/hinglish/predictions_<adapter>.parquet
  experiments/eot-bench/hinglish/predictions_<adapter>.csv
  experiments/eot-bench/hinglish/metrics_<adapter>.json
  experiments/eot-bench/hinglish/scored_pauses_<adapter>.csv
    (candidate_id, turn_id, speaker_id, start/end, label, p_eot, ...joined)
"""

import argparse
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DEFAULT_TRANSCRIPT_LAG = 0.5
EOT_HOLD_LABELS = {"EOT": "eot", "HOLD": "hold"}

PAUSE_CSV = Path("data/annotations-hinglish/pauses.csv")
PROCESSED_WAVS = {
    "hinglish_001_general": "data/processed/hinglish_conversation_001_general_16k.wav",
    "hinglish_002_cricket": "data/processed/hinglish_conversation_002_cricket_16k.wav",
    "hinglish_003_movies": "data/processed/hinglish_conversation_003_movies_16k.wav",
}


def load_transcript_words(recording_id: str) -> list[dict]:
    """All words from all segments, sorted by start_time."""
    p = Path("data/transcripts") / f"{recording_id}.json"
    with open(p) as f:
        data = json.load(f)
    words: list[dict] = []
    for seg in data["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"], "start_time": w["start_time"], "end_time": w["end_time"]})
    words.sort(key=lambda w: w["start_time"])
    return words


def build_messages(words: list[dict], timestamp: float, *, lag: float = DEFAULT_TRANSCRIPT_LAG) -> list[dict]:
    """Mirror eot_harness.io.build_messages: words visible up to t - lag."""
    effective = timestamp - lag
    visible = [str(w["word"]).strip() for w in words if w["end_time"] <= effective + 1e-6]
    visible = [w for w in visible if w]
    if not visible:
        return []
    return [{"role": "user", "content": " ".join(visible)}]


def score_adapter(adapter_name: str):
    replay = False
    if adapter_name == "smart-turn":
        from eot_harness.smart_turn_adapter import SmartTurnAudioAdapter

        return SmartTurnAudioAdapter(), "smart_turn_audio_adapter"
    elif adapter_name == "livekit-mini":
        from eot_harness.livekit_turn_detector_mini_adapter import LiveKitTurnDetectorMiniAdapter

        return LiveKitTurnDetectorMiniAdapter(), "live_kit_turn_detector_mini_adapter"
    else:
        raise ValueError(f"unknown adapter {adapter_name!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", required=True, choices=["smart-turn", "livekit-mini"])
    parser.add_argument("--out-dir", default="experiments/eot-bench/hinglish")
    parser.add_argument("--transcript-lag", type=float, default=DEFAULT_TRANSCRIPT_LAG)
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    pauses = pd.read_csv(PAUSE_CSV, keep_default_na=False)
    scored = pauses[pauses["scored"] == True].copy()  # noqa: E712
    assert len(scored) == 116, f"expected 116 scored pauses, got {len(scored)}"
    assert set(scored["boundary_type"].unique()) == {"EOT", "HOLD"}

    # Pre-load audio per recording (full buffers) + words.
    audio_cache: dict[str, tuple[np.ndarray, int]] = {}
    words_cache: dict[str, list[dict]] = {}
    for rid, path in PROCESSED_WAVS.items():
        arr, sr = sf.read(path, dtype="float32")
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        audio_cache[rid] = (arr, int(sr))
        words_cache[rid] = load_transcript_words(rid)
        print(f"[load] {rid}: {len(arr)} samples @ {sr}Hz, {len(words_cache[rid])} words, {len(arr)/sr:.1f}s")

    adapter = None
    try:
        adapter, adapter_slug = score_adapter(args.adapter)
        print(f"[adapter] {adapter_slug}: {adapter.display_name} (id={adapter.adapter_id})")

        batch_inputs, batch_meta = [], []
        for idx, row in scored.reset_index(drop=True).iterrows():
            rid = row["recording_id"]
            arr, sr = audio_cache[rid]
            t = float(row["start_time"])
            end_sample = int(round(t * sr))
            prefix = arr[:end_sample].copy()
            messages = build_messages(words_cache[rid], t, lag=args.transcript_lag)
            batch_inputs.append(
                {
                    "language": "hi",
                    "audio": {"array": prefix, "sampling_rate": sr},
                    "messages": messages,
                }
            )
            batch_meta.append(
                {
                    "candidate_id": row["candidate_id"],
                    "recording_id": rid,
                    "turn_id": row["turn_id"],
                    "speaker_id": row["speaker_id"],
                    "start_time": t,
                    "end_time": float(row["end_time"]),
                    "label": EOT_HOLD_LABELS[row["boundary_type"]],
                    "preceding_text": row["preceding_text"],
                    "following_text": row["following_text"],
                    "preceding_language": row["preceding_language"],
                    "following_language": row["following_language"],
                    "code_switch_around_pause": row["code_switch_around_pause"],
                }
            )
            if idx % 20 == 19:
                print(f"[build] {idx+1}/{len(scored)}", flush=True)
        print(f"[build] all {len(batch_inputs)} inputs built", flush=True)

        t0 = time.perf_counter()
        p_eots = adapter.predict_batch(batch_inputs)
        elapsed = time.perf_counter() - t0
        print(f"[predict] scored {len(p_eots)} in {elapsed:.1f}s ({elapsed/len(p_eots)*1000:.1f} ms/pause)")

        scoring = pd.DataFrame(batch_meta)
        scoring["p_eot"] = p_eots

        # --- Benchmark-compatible predictions table (one row per scored pause). ---
        preds = pd.DataFrame(
            {
                "id": scoring["candidate_id"],
                "language": "hi",
                "span_index": 0,
                "timestamp": scoring["start_time"],
                "silence_dur": 0.0,
                "p_eot": scoring["p_eot"],
                "label": scoring["label"],
            }
        )
        preds_path = outdir / f"predictions_{args.adapter}.parquet"
        preds.to_parquet(preds_path, index=False)
        preds.to_csv(outdir / f"predictions_{args.adapter}.csv", index=False)
        scoring.to_csv(outdir / f"scored_pauses_{args.adapter}.csv", index=False)
        print(f"[write] {preds_path} ({len(preds)} rows)")

        # --- Metrics (Phase E core): use harness functions where they apply directly. ---
        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true = (scoring["label"] == "eot").astype(int).to_numpy()
        y_score = scoring["p_eot"].to_numpy(dtype=float)
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))

        # Policy sweep at causal onset: threshold grid, action_delay == 0.0
        # (decide immediately at pause onset), timeouts as in harness.
        # We reuse metrics.compute_joint_policy_sweep machinery via a
        # score_point-shaped table: each pause is an EOT/HOLD "span" with
        # p_eot scored at silence_dur 0.0, so fire time is 0 or inf.
        from eot_harness.metrics import compute_classification_metrics

        span_df = scoring.rename(columns={"p_eot": "p_eot"})
        # Build the minimal table compute_classification_metrics expects.
        cm = compute_classification_metrics(
            pd.DataFrame(
                {
                    "label": scoring["label"],
                    "p_eot": scoring["p_eot"],
                }
            )
        )
        assert abs(cm["auc"] - auc) < 1e-9 and abs(cm["ap"] - ap) < 1e-9, "metric mismatch"
        print(f"[metrics] AUC={auc:.4f} AP={ap:.4f} (n={len(scoring)}: {cm['n_eot_spans']} EOT, {cm['n_hold_spans']} HOLD)")

        # Frontier latency at causal onset: with action_delay=0, model fire is
        # instantaneous at onset; latency is therefore the onset itself. The
        # meaningful latency comparison is via timeout (how long until forced
        # action) — report cutoff/detect rates by timeout, threshold swept.
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        operating = []
        for thr in thresholds:
            cutoff_rate = float((scoring.loc[scoring["label"] == "hold", "p_eot"] > thr).mean())
            detect_rate = float((scoring.loc[scoring["label"] == "eot", "p_eot"] > thr).mean())
            operating.append({"threshold": thr, "cutoff_rate": cutoff_rate, "detect_rate": detect_rate})
            print(f"  thr={thr:.1f} cutoff={cutoff_rate:.3f} detect={detect_rate:.3f}")

        # Throughput / inference latency (separate from policy latency).
        rt = elapsed / sum(len(audio_cache[r][0]) / audio_cache[r][1] for r in audio_cache)  # RTF
        metrics_obj = {
            "adapter": adapter_slug,
            "display_name": adapter.display_name,
            "protocol": "causal onset (audio prefix [0, pause.start], silence_dur=0), transcript lag 0.5s words",
            "n_scored": int(len(scoring)),
            "n_eot": int(cm["n_eot_spans"]),
            "n_hold": int(cm["n_hold_spans"]),
            "auc": auc,
            "ap": ap,
            "operating_points_onset": operating,
            "inference": {
                "total_seconds": round(elapsed, 3),
                "ms_per_pause": round(elapsed / len(scoring) * 1000, 1),
                "realtime_factor": round(elapsed / sum(len(audio_cache[r][0]) / audio_cache[r][1] for r in audio_cache), 4),
            },
            "score_point_caveat": (
                "Hinglish scored at pause onset (silence_dur=0.0); Hindi benchmark scores at "
                "0.2s INTO silence (score_point_s=0.2). Direct AUC comparison carries a "
                "~0.05 protocol offset caveat; see evaluation_protocol.md."
            ),
            "env": {
                "python": platform.python_version(),
                "platform": platform.platform(),
            },
        }
        metrics_path = outdir / f"metrics_{args.adapter}.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics_obj, f, indent=2)
        print(f"[write] {metrics_path}")
    finally:
        close = getattr(adapter, "close", None) if adapter is not None else None
        if callable(close):
            close()


if __name__ == "__main__":
    main()
