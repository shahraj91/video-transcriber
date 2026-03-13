#!/usr/bin/env python3
"""
Video Transcription Pipeline
video → ffmpeg (audio extraction) → Whisper (STT) → Llama (refinement, speakers,
        summary, action items, translation) → .txt + .srt + extras
"""

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path


# ── Config ────────────────────────────────────────────────────────────────────

LLAMA_API_URL = "http://localhost:11434/api/generate"   # Ollama default
LLAMA_MODEL   = "llama3:8b"                              # change to your model name
WHISPER_MODEL = "base"                                   # tiny | base | small | medium | large


# ── Core: ffmpeg + Whisper ────────────────────────────────────────────────────

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
    print(f"[2/3] Transcribing with Whisper ({whisper_model} model)...")
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


# ── Llama helpers ─────────────────────────────────────────────────────────────

def call_llama(prompt: str, label: str) -> str | None:
    """
    Send a prompt to local Llama via Ollama REST API.
    Returns the response string, or None if Llama is unreachable.
    """
    try:
        payload = json.dumps({
            "model": LLAMA_MODEL,
            "prompt": prompt,
            "stream": False
        }).encode()
        req = urllib.request.Request(
            LLAMA_API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
            response = data.get("response", "").strip()
            if response:
                print(f"  {label} complete.")
                return response
    except Exception as e:
        print(f"  Llama unreachable for {label} ({e}). Skipping.")
    return None


# ── Feature 1: Refinement ─────────────────────────────────────────────────────

def refine_transcript(raw_text: str) -> str:
    """Clean up raw Whisper transcript — fix punctuation, remove filler words."""
    print(f"  -> Refining transcript...")
    prompt = (
        "You are a transcript editor. Clean up the following raw speech transcript: "
        "fix punctuation, capitalisation, and remove filler words (um, uh, like). "
        "Return ONLY the cleaned transcript text, no commentary.\n\n"
        f"TRANSCRIPT:\n{raw_text}"
    )
    return call_llama(prompt, "Refinement") or raw_text


# ── Feature 2: Speaker Detection ─────────────────────────────────────────────

def detect_speakers(refined_text: str) -> str:
    """
    Ask Llama to identify speaker changes and label them.
    Returns transcript with Speaker 1:, Speaker 2: etc. labels.
    """
    print(f"  -> Detecting speakers...")
    prompt = (
        "You are a transcript editor. Read the following transcript and identify where "
        "the speaker changes based on context, tone shifts, or conversational cues. "
        "Label each speaker's turn as 'Speaker 1:', 'Speaker 2:', etc. "
        "If it appears to be a single speaker throughout, label all lines as 'Speaker 1:'. "
        "Return ONLY the labelled transcript, no commentary or explanation.\n\n"
        f"TRANSCRIPT:\n{refined_text}"
    )
    result = call_llama(prompt, "Speaker detection")
    return result or refined_text


# ── Feature 3: Summary ────────────────────────────────────────────────────────

def summarise_transcript(refined_text: str) -> str:
    """Generate a concise 3-5 sentence summary of the transcript."""
    print(f"  -> Generating summary...")
    prompt = (
        "You are a transcript summariser. Read the following transcript and write a "
        "clear, concise summary in 3 to 5 sentences covering the main topics discussed. "
        "Return ONLY the summary, no preamble or commentary.\n\n"
        f"TRANSCRIPT:\n{refined_text}"
    )
    return call_llama(prompt, "Summary") or "Summary could not be generated."


# ── Feature 4: Action Items ───────────────────────────────────────────────────

def extract_action_items(refined_text: str) -> str:
    """
    Extract tasks, decisions, and follow-ups from the transcript.
    Useful for meeting recordings.
    """
    print(f"  -> Extracting action items...")
    prompt = (
        "You are a meeting notes assistant. Read the following transcript and extract "
        "all action items, tasks, decisions, and follow-ups that were mentioned. "
        "Format each item as a bullet point starting with '- '. "
        "If no clear action items are found, write: 'No action items identified.' "
        "Return ONLY the bullet list, no preamble or commentary.\n\n"
        f"TRANSCRIPT:\n{refined_text}"
    )
    return call_llama(prompt, "Action items") or "Action items could not be extracted."


# ── Feature 5: Translation ────────────────────────────────────────────────────

def translate_transcript(refined_text: str, language: str) -> str:
    """Translate the refined transcript into the specified language."""
    print(f"  -> Translating to {language}...")
    prompt = (
        f"You are a professional translator. Translate the following transcript into {language}. "
        "Preserve the meaning and natural flow of speech. "
        "Return ONLY the translated text, no commentary or explanation.\n\n"
        f"TRANSCRIPT:\n{refined_text}"
    )
    return call_llama(prompt, f"Translation ({language})") or "Translation could not be generated."


# ── SRT helpers ───────────────────────────────────────────────────────────────

def format_srt_time(seconds: float) -> str:
    """Convert float seconds to SRT timestamp HH:MM:SS,mmm"""
    ms = int((seconds % 1) * 1000)
    s  = int(seconds) % 60
    m  = (int(seconds) // 60) % 60
    h  = int(seconds) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


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
    print(f"  .srt saved       -> {out_path.name}")


def save_text(content: str, out_path: Path, label: str) -> None:
    """Generic text file saver."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  {label} saved -> {out_path.name}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global LLAMA_MODEL

    parser = argparse.ArgumentParser(
        description="Transcribe a video to .txt and .srt, with optional Llama enhancements"
    )
    parser.add_argument("video",
                        help="Path to the input video file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Directory to save outputs (default: same as video)")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--llama-model", default=LLAMA_MODEL,
                        help=f"Ollama model name (default: {LLAMA_MODEL})")
    parser.add_argument("--language", default=None,
                        help="Translate transcript to this language e.g. Hindi, Spanish, French")

    # Flags to skip individual Llama features
    parser.add_argument("--no-llama",    action="store_true", help="Skip ALL Llama processing")
    parser.add_argument("--no-speakers", action="store_true", help="Skip speaker detection")
    parser.add_argument("--no-summary",  action="store_true", help="Skip summary generation")
    parser.add_argument("--no-actions",  action="store_true", help="Skip action items extraction")

    args = parser.parse_args()

    # ── Validate input ────────────────────────────────────────────────────────
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir  = Path(args.output_dir).resolve() if args.output_dir else video_path.parent
    out_dir   = base_dir / f"{video_path.stem}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output folder: {out_dir}")

    stem = video_path.stem
    LLAMA_MODEL = args.llama_model

    # ── Output paths ──────────────────────────────────────────────────────────
    audio_path       = out_dir / f"{stem}_audio.wav"
    txt_path         = out_dir / f"{stem}_transcript.txt"
    srt_path         = out_dir / f"{stem}_transcript.srt"
    speakers_path    = out_dir / f"{stem}_speakers.txt"
    summary_path     = out_dir / f"{stem}_summary.txt"
    actions_path     = out_dir / f"{stem}_actions.txt"
    translation_path = out_dir / f"{stem}_translation_{args.language}.txt" if args.language else None

    # ── Step 1: Extract audio ─────────────────────────────────────────────────
    extract_audio(video_path, audio_path)

    # ── Step 2: Whisper transcription ─────────────────────────────────────────
    segments = transcribe_with_whisper(audio_path, args.whisper_model)
    raw_text = " ".join(seg["text"] for seg in segments)

    # ── Step 3: Llama enhancements ────────────────────────────────────────────
    print(f"[3/3] Running Llama enhancements ({LLAMA_MODEL})...")

    if args.no_llama:
        refined_text = raw_text
        print("  Skipping all Llama features (--no-llama).")
    else:
        # Feature 1: Refinement (always runs unless --no-llama)
        refined_text = refine_transcript(raw_text)

        # Feature 2: Speaker detection
        if not args.no_speakers:
            speakers_text = detect_speakers(refined_text)
            save_text(speakers_text, speakers_path, "Speakers  ")

        # Feature 3: Summary
        if not args.no_summary:
            summary_text = summarise_transcript(refined_text)
            save_text(summary_text, summary_path, "Summary   ")

        # Feature 4: Action items
        if not args.no_actions:
            actions_text = extract_action_items(refined_text)
            save_text(actions_text, actions_path, "Actions   ")

        # Feature 5: Translation (only if --language is passed)
        if args.language and translation_path:
            translation_text = translate_transcript(refined_text, args.language)
            save_text(translation_text, translation_path, f"Translation ({args.language})")

    # ── Step 4: Save core outputs ─────────────────────────────────────────────
    save_text(refined_text, txt_path, "Transcript")
    save_srt(segments, srt_path)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    audio_path.unlink(missing_ok=True)

    # ── Print output summary ──────────────────────────────────────────────────
    print("\n Done!")
    print(f"   Transcript  -> {txt_path}")
    print(f"   Subtitles   -> {srt_path}")
    if not args.no_llama:
        if not args.no_speakers: print(f"   Speakers    -> {speakers_path}")
        if not args.no_summary:  print(f"   Summary     -> {summary_path}")
        if not args.no_actions:  print(f"   Actions     -> {actions_path}")
        if args.language:        print(f"   Translation -> {translation_path}")


if __name__ == "__main__":
    main()
