#!/usr/bin/env python3
"""
transcribe.py — Entry point for the Video Transcription Pipeline.

Usage:
    python3 transcribe.py video.mp4
    python3 transcribe.py video.mp4 --whisper-model medium
    python3 transcribe.py video.mp4 --language Hindi
    python3 transcribe.py video.mp4 --no-summary --no-actions
    python3 transcribe.py video.mp4 --no-llama
"""

import argparse
import sys
from pathlib import Path

from config import LLAMA_MODEL, WHISPER_MODEL
from pipeline.audio       import AudioExtractor
from pipeline.transcriber import WhisperTranscriber
from pipeline.llama       import LlamaClient
from pipeline.enhancers   import (
    TranscriptRefiner,
    SpeakerDetector,
    TranscriptSummariser,
    ActionItemExtractor,
    TranscriptTranslator,
)
from pipeline.output import OutputManager


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe a video to .txt and .srt, with optional Llama enhancements"
    )
    parser.add_argument("video",
                        help="Path to the input video file")
    parser.add_argument("--output-dir", "-o", default=None,
                        help="Base directory for output (a timestamped subfolder is created inside)")
    parser.add_argument("--whisper-model", default=WHISPER_MODEL,
                        choices=["tiny", "base", "small", "medium", "large"],
                        help=f"Whisper model size (default: {WHISPER_MODEL})")
    parser.add_argument("--llama-model", default=LLAMA_MODEL,
                        help=f"Ollama model name (default: {LLAMA_MODEL})")
    parser.add_argument("--language", default=None,
                        help="Translate transcript to this language e.g. Hindi, Spanish, French")

    # Skip flags for individual Llama features
    parser.add_argument("--no-llama",    action="store_true", help="Skip ALL Llama processing")
    parser.add_argument("--no-speakers", action="store_true", help="Skip speaker detection")
    parser.add_argument("--no-summary",  action="store_true", help="Skip summary generation")
    parser.add_argument("--no-actions",  action="store_true", help="Skip action items extraction")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Validate input video ──────────────────────────────────────────────────
    video_path = Path(args.video).resolve()
    if not video_path.exists():
        print(f"Error: video file not found: {video_path}")
        sys.exit(1)

    base_dir = Path(args.output_dir).resolve() if args.output_dir else None

    # ── Set up output folder ──────────────────────────────────────────────────
    output = OutputManager(video_path, base_dir)
    audio_path = output.path("_audio.wav")

    # ── Step 1: Extract audio ─────────────────────────────────────────────────
    AudioExtractor().extract(video_path, audio_path)

    # ── Step 2: Transcribe with Whisper ───────────────────────────────────────
    transcriber = WhisperTranscriber(model_size=args.whisper_model)
    segments    = transcriber.transcribe(audio_path)
    raw_text    = WhisperTranscriber.segments_to_text(segments)

    # ── Step 3: Llama enhancements ────────────────────────────────────────────
    print(f"[3/3] Running Llama enhancements ({args.llama_model})...")

    if args.no_llama:
        refined_text = raw_text
        print("  Skipping all Llama features (--no-llama).")
    else:
        client = LlamaClient(model=args.llama_model)

        # Feature 1: Refinement — always runs unless --no-llama
        refined_text = TranscriptRefiner(client).run(raw_text)

        # Feature 2: Speaker detection
        if not args.no_speakers:
            speakers_text = SpeakerDetector(client).run(refined_text)
            output.save_text(speakers_text, "_speakers.txt", "Speakers")

        # Feature 3: Summary
        if not args.no_summary:
            summary_text = TranscriptSummariser(client).run(refined_text)
            output.save_text(summary_text, "_summary.txt", "Summary")

        # Feature 4: Action items
        if not args.no_actions:
            actions_text = ActionItemExtractor(client).run(refined_text)
            output.save_text(actions_text, "_actions.txt", "Action items")

        # Feature 5: Translation (only if --language is passed)
        if args.language:
            translation_text = TranscriptTranslator(client).run(refined_text, args.language)
            output.save_text(translation_text, f"_translation_{args.language}.txt",
                             f"Translation ({args.language})")

    # ── Step 4: Save core outputs ─────────────────────────────────────────────
    output.save_text(refined_text, "_transcript.txt", "Transcript")
    output.save_srt(segments)

    # ── Step 5: Cleanup temp audio ────────────────────────────────────────────
    audio_path.unlink(missing_ok=True)

    # ── Done ──────────────────────────────────────────────────────────────────
    print(f"\n✅ Done! All outputs saved to:\n   {output.out_dir}")


if __name__ == "__main__":
    main()
