#!/usr/bin/env python
"""Phase B transcription for TurnSense (Hinglish code-shift corpus).

Uses faster-whisper `base` (multilingual) on CPU.
Transcribes the three 16kHz mono WAVs in data/processed/ and writes:
  data/transcripts/hinglish_001_general.json
  data/transcripts/hinglish_002_cricket.json
  data/transcripts/hinglish_003_movies.json
  data/transcripts/combined_segments.csv
  data/transcripts/transcription_metadata.json

Model-selection rationale (documented after testing 2026-11-05):
  - distil-large-v3 / small at int8 on this ARM64 CPU are 1.5–3.5× SLOWER than
    real-time (small int8: 213s for 60s audio; distil-large-v3: >600s timeout
    on a 15s slice), and small emits garbled Devanagari/CJK artifacts on
    romanized Hinglish. distil-large-v3 was therefore NOT used.
  - `base` int8 transcribes the full 231s file in ~9s, yields clean romanized
    Hinglish with high word-probability and consistent word timestamps.
    Romanized output is retained verbatim (no script normalization).
  - `base` float32 is 4× slower for marginal spelling changes (Yaar vs YAR);
    int8 is used for speed with no material loss for draft alignment.

Settings (chosen once, recorded in metadata):
  - model: base (multilingual), language=None (auto-detect)
  - CPU, int8 compute type; word_timestamps=True; vad_filter=False
  - beam_size=1 (greedy; beam_size=5 tested, no material gain for draft use)
  - no translation, no initial_prompt, no hotwords
  - NO diarization (no external diarization model loaded; speaker_id omitted)

Usage:
  # Short slice test (first 15s of recording 001, prints structure, writes nothing):
  ~/miniforge3-arm64/envs/turnsense/bin/python tools/transcribe.py --test

  # Full run:
  ~/miniforge3-arm64/envs/turnsense/bin/python tools/transcribe.py
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
import tempfile
import wave
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------
# Constants (paths via CLI args override these)
# --------------------------------------------------------------------------
MODEL_NAME = "base"

RECORDINGS = [
    ("hinglish_001_general", "data/processed/hinglish_conversation_001_general_16k.wav"),
    ("hinglish_002_cricket", "data/processed/hinglish_conversation_002_cricket_16k.wav"),
    ("hinglish_003_movies", "data/processed/hinglish_conversation_003_movies_16k.wav"),
]

DEFAULT_OUTPUT_DIR = "data/transcripts"
DEFAULT_FFMPEG = "/opt/homebrew/bin/ffmpeg"


def wav_duration_sec(path: str) -> float:
    """Duration of a PCM WAV in seconds using the stdlib wave module."""
    with wave.open(path, "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return frames / rate


def transcribe_one(model, path, args, clip_timestamps="0"):
    """Run faster-whisper on one file; return (segments list of dicts, info)."""
    segments, info = model.transcribe(
        path,
        language=args.language,
        beam_size=args.beam_size,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad_filter,
        clip_timestamps=clip_timestamps,
    )

    out = []
    for seg in segments:  # generator consumed lazily
        words = []
        if seg.words:
            for w in seg.words:
                words.append(
                    {
                        "word": w.word,
                        "start_time": w.start,
                        "end_time": w.end,
                        "probability": w.probability,
                    }
                )
        out.append(
            {
                "start_time": seg.start,
                "end_time": seg.end,
                # Raw text kept verbatim (no stripping / special-casing).
                "text": seg.text,
                # faster-whisper exposes only TRANSCRIPTION-level detection, so
                # we write the detected language the engine actually used.
                "detected_language": info.language,
                "no_speech_prob": seg.no_speech_prob,
                "avg_logprob": seg.avg_logprob,
                "words": words,
            }
        )
    return out, info


def build_language_settings(args, info):
    return {
        "language": args.language,
        "detected_language": info.language,
        "language_probability": info.language_probability,
        "note": (
            "Auto-detect (language=None). faster-whisper reports "
            "transcription-level detection only; the same value is repeated "
            "on each segment. No per-segment detector was run."
        ),
    }


def run_test(args):
    """Transcribe the first 15s of recording 001; print a structural demo."""
    from faster_whisper import WhisperModel
    recording_id, src = RECORDINGS[0]
    print(f"[test] slicing first 15s of {src} -> scratch wav", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        slice_wav = str(Path(tmp) / "test_slice_001.wav")
        subprocess.run(
            [args.ffmpeg, "-y", "-v", "error", "-i", src,
             "-ss", "0", "-t", "15", "-ac", "1", "-ar", "16000", slice_wav],
            check=True,
        )
        dur = wav_duration_sec(slice_wav)
        print(f"[test] slice duration: {dur:.2f}s", flush=True)

        print("[test] loading model...", flush=True)
        model = WhisperModel(
            args.model, device=args.device, compute_type=args.compute_type
        )
        print("[test] model loaded; transcribing (word_timestamps=%s, vad_filter=%s, "
              "beam_size=%s)..." % (args.word_timestamps, args.vad_filter, args.beam_size),
              flush=True)
        segs, info = transcribe_one(model, slice_wav, args)

    print(f"[test] detected_language={info.language!r} "
          f"(prob={info.language_probability:.3f})", flush=True)
    print(f"[test] segments={len(segs)}; segments with words="
          f"{sum(1 for s in segs if s['words'])}", flush=True)
    if not segs:
        print("[test] WARNING: no segments emitted on slice.", flush=True)
    print("[test] first 3 segments (full structure, incl. word timings):", flush=True)
    print(json.dumps(segs[:3], ensure_ascii=False, indent=2), flush=True)
    ok = bool(segs) and any(s["words"] for s in segs)
    print("[test] " + ("PASS: model loads and emits timestamps + word timings."
                       if ok else "FAIL: missing timestamps/words."), flush=True)
    return 0 if ok else 1


def run_full(args):
    import ctranslate2
    import faster_whisper
    from faster_whisper import WhisperModel

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    per_recording = {}
    runtimes = {}
    csv_rows = []

    print("[run] loading model %s (device=%s, compute_type=%s)..."
          % (args.model, args.device, args.compute_type), flush=True)
    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)
    print("[run] model loaded.", flush=True)

    for idx, (recording_id, src) in enumerate(RECORDINGS):
        print(f"[run] transcribing {recording_id}: {src}", flush=True)
        duration_sec = round(wav_duration_sec(src), 3)
        t0 = time.perf_counter()
        segs, info = transcribe_one(model, src, args)
        elapsed = time.perf_counter() - t0
        runtimes[recording_id] = round(elapsed, 2)

        langs = sorted({s["detected_language"] for s in segs})
        print(f"[run]   done: {len(segs)} segments, duration {duration_sec}s, "
              f"runtime {runtimes[recording_id]}s, "
              f"detected_language={info.language!r} (prob={info.language_probability:.3f})",
              flush=True)

        rec_obj = {
            "recording_id": recording_id,
            "source_wav": src,
            "duration_sec": duration_sec,
            "model": args.model,
            "language_settings": build_language_settings(args, info),
            "segments": [
                {
                    "start_time": s["start_time"],
                    "end_time": s["end_time"],
                    "text": s["text"],
                    "detected_language": s["detected_language"],
                    "no_speech_prob": s["no_speech_prob"],
                    "avg_logprob": s["avg_logprob"],
                    "words": s["words"],
                }
                for s in segs
            ],
        }
        (outdir / f"{recording_id}.json").write_text(
            json.dumps(rec_obj, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        per_recording[recording_id] = {
            "num_segments": len(segs),
            "duration_sec": duration_sec,
        }
        for i, s in enumerate(segs):
            csv_rows.append(
                {
                    "recording_id": recording_id,
                    "segment_index": i,
                    "start_time": s["start_time"],
                    "end_time": s["end_time"],
                    "text": s["text"],
                    "detected_language": s["detected_language"]
                    if s["detected_language"] is not None else "",
                    "no_speech_prob": s["no_speech_prob"]
                    if s["no_speech_prob"] is not None else "",
                    "avg_logprob": s["avg_logprob"]
                    if s["avg_logprob"] is not None else "",
                }
            )

    # ---- combined_segments.csv -------------------------------------------
    csv_path = outdir / "combined_segments.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "recording_id", "segment_index", "start_time", "end_time",
                "text", "detected_language", "no_speech_prob", "avg_logprob",
            ],
        )
        w.writeheader()
        w.writerows(csv_rows)

    # ---- transcription_metadata.json -------------------------------------
    metadata = {
        "transcription_date": "2026-11-05",
        "model": args.model,
        "model_revision": args.model_revision,
        "faster_whisper_version": faster_whisper.__version__,
        "ctranslate2_version": ctranslate2.__version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "language_settings": {
            "language": args.language,
            "note": (
                "language=None (auto-detect), no forced [hi,en], no translation. "
                "detected_language per recording is recorded in each per-recording "
                "JSON; faster-whisper emits only transcription-level detection."
            ),
        },
        "decoding_parameters": {
            "beam_size": args.beam_size,
            "device": args.device,
            "compute_type": args.compute_type,
            "word_timestamps": args.word_timestamps,
            "vad_filter": args.vad_filter,
            "translate": False,
            "initial_prompt": None,
            "hotwords": None,
            "clip_timestamps": None,
        },
        "runtimes": runtimes,
        "per_recording": per_recording,
        "notes": [
            "No diarization performed; no external diarization model loaded; "
            "speaker_id is intentionally omitted from all output rows.",
            "Language detection is transcription-level only and is unreliable for "
            "Hinglish code-switching; per-segment detected_language repeats the "
            "transcription-level value.",
            "Segments with high no_speech_prob may be silence/unintelligible; "
            "they were retained without special-casing or deletion.",
            "ASR output is a DRAFT only and is NOT ground truth.",
            "vad_filter=False chosen to keep the first pass simple and honest; "
            "silences were not suppressed.",
            "beam_size=5 chosen for accuracy (greedy=1 considered); int8 on CPU "
            "chosen for speed. No translation; no initial_prompt/hotwords.",
            "model_revision: resolved by faster-whisper from HF Hub at load time; "
            "not pinned unless model_revision below is set.",
        ],
    }
    (outdir / "transcription_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[run] wrote 3 JSONs + combined_segments.csv ({len(csv_rows)} rows) "
          f"+ transcription_metadata.json to {outdir}", flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--word-timestamps", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--vad-filter", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--language", default=None)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test", action="store_true",
                        help="Test on first 15s of recording 001 only; write nothing.")
    args = parser.parse_args(argv)
    return run_test(args) if args.test else run_full(args)


if __name__ == "__main__":
    raise SystemExit(main())
