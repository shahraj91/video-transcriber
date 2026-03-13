# ── pipeline/audio.py ────────────────────────────────────────────────────────
# Responsible for extracting audio from a video file using ffmpeg.

import subprocess
import sys
from pathlib import Path


class AudioExtractor:
    """Extracts audio from a video file and saves it as a WAV file."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate  # 16kHz — Whisper's native format
        self.channels    = channels     # mono — Whisper doesn't use stereo

    def extract(self, video_path: Path, audio_path: Path) -> None:
        """
        Extract audio from video_path and save to audio_path as WAV.
        Exits the program if ffmpeg fails.
        """
        print(f"[1/3] Extracting audio from: {video_path.name}")
        cmd = [
            "ffmpeg", "-y",                         # overwrite output without prompt
            "-i", str(video_path),                  # input video
            "-vn",                                  # strip video track
            "-acodec", "pcm_s16le",                 # 16-bit PCM WAV (lossless)
            "-ar", str(self.sample_rate),           # sample rate
            "-ac", str(self.channels),              # channels
            str(audio_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ffmpeg error:\n{result.stderr}")
            sys.exit(1)
        print(f"  Audio saved to: {audio_path.name}")
