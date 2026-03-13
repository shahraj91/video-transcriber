#!/usr/bin/env python3
"""
Video Transcription Pipeline
video → ffmpeg (audio extraction) → Whisper (STT) → Llama (refinement) → .txt + .srt
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_API_URL = "http://localhost:11434/api/generate"   # Ollama default
LLAMA_MODEL   = "llama3:8b"                              # change to your model name
WHISPER_MODEL = "base"                                   # tiny | base | small | medium | large


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video using ffmpeg."""
    print(f"[1/3] Extracting audio from: {video_path.name}")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",                      # no video
        "-acodec", "pcm_s16le",     # WAV 16-bit PCM (Whisper-friendly)
        "-ar", "16000",             # 16 kHz sample rate
        "-ac", "1",                 # mono
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ffmpeg error:\n{result.stderr}")
        sys.exit(1)
    print(f"  Audio saved to: {audio_path.name}")


def transcribe_with_whisper(audio_path: Path, whisper_model: str) -> list[dict]:
    """
    Transcribe audio using openai-whisper.
    Returns list of segments: [{start, end, text}, ...]
    """
    print(f"[2/3] Transcribing with Whisper ({whisper_model} model)…")
    try:
        import whisper
    except ImportError:
        print("  whisper not found. Install with:  pip install openai-whisper")
        sys.exit(1)

    model = whisper.load_model(whisper_model)
    result = model.transcribe(str(audio_path), verbose=False)
    segments = [
        {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
        for seg in result["segments"]
    ]
    print(f"  Got {len(segments)} segments.")
    return segments


def refine_with_llama(raw_text: str) -> str:
    """
    Send raw transcript to local Llama via Ollama and return refined text.
    Falls back to raw text if Llama is unreachable.
    """
    print(f"[2.5/3] Refining transcript with Llama ({LLAMA_MODEL})…")
    try:
        import urllib.request
        prompt = (
            "You are a transcript editor. Clean up the following raw speech transcript: "
            "fix punctuation, capitalisation, and remove filler words (um, uh, like). "
            "Return ONLY the cleaned transcript text, no commentary.\n\n"
            f"TRANSCRIPT:\n{raw_text}"
        )
        payload = json.dumps({"model": LLAMA_MODEL, "prompt": prompt, "stream": False}).encode()
        req = urllib.request.Request(
            LLAMA_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            refined = data.get("response", "").strip()
            if refined:
                print("  Refinement complete.")
                return refined
    except Exception as e:
        print(f"  Llama unreachable ({e}). Using raw Whisper transcript.")
    return raw_text


def format_srt_time(seconds: float) -> str:
    """Convert float seconds → SRT timestamp  HH:MM:SS,mmm"""
    ms  = int((seconds % 1) * 1000)
    s   = int(seconds) % 60
    m   = (int(seconds) // 60) % 60
    h   = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def save_txt(segments: list[dict], refined_text: str, out_path: Path) -> None:
    """Save plain-text transcript (Llama-refined full text)."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(refined_text)
    print(f"  .txt saved → {out_path}")


def save_srt(segments: list[dict], out_path: Path) -> None:
    """Save SRT subtitle file from Whisper segments (timing preserved)."""
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_srt_time(seg['start'])} --> {format_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  .srt saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global LLAMA_MODEL
    parser = argparse.ArgumentParser(description="Transcribe a video to .txt and .srt")
    parser.add_argument("video", help="Path to the input video file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save outputs (default: same as video)")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--llama-model", default=LLAMA_MODEL,
                        help=f"Ollama model name (default: {LLAMA_MODEL})")
    parser.add_argument("--no-llama", action="store_true",
                        help="Skip Llama refinement, use raw Whisper output")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve() if args.output_dir else video_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    stem        = video_path.stem
    audio_path  = out_dir / f"{stem}_audio.wav"
    txt_path    = out_dir / f"{stem}_transcript.txt"
    srt_path    = out_dir / f"{stem}_transcript.srt"

    LLAMA_MODEL = args.llama_model

    # Step 1 – extract audio
    extract_audio(video_path, audio_path)

    # Step 2 – Whisper transcription
    segments = transcribe_with_whisper(audio_path, args.whisper_model)
    raw_text = " ".join(seg["text"] for seg in segments)

    # Step 3 – optional Llama refinement
    if args.no_llama:
        refined_text = raw_text
    else:
        refined_text = refine_with_llama(raw_text)

    # Step 4 – save outputs
    print("[3/3] Saving transcript files…")
    save_txt(segments, refined_text, txt_path)
    save_srt(segments, srt_path)

    # Clean up temp audio
    audio_path.unlink(missing_ok=True)

    print("\n✅ Done!")
    print(f"   TXT  → {txt_path}")
    print(f"   SRT  → {srt_path}")


if __name__ == "__main__":
    main()
