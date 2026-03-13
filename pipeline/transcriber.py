# ── pipeline/transcriber.py ──────────────────────────────────────────────────
# Responsible for speech-to-text transcription using OpenAI Whisper (local).

import sys
from pathlib import Path


class WhisperTranscriber:
    """
    Transcribes audio to text using a locally running Whisper model.
    Returns timestamped segments: [{start, end, text}, ...]
    """

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model     = None          # lazy-loaded on first transcribe() call

    def _load_model(self):
        """Load Whisper model (downloaded and cached on first run)."""
        if self._model is None:
            try:
                import whisper
            except ImportError:
                print("  whisper not found. Install with:  pip install openai-whisper")
                sys.exit(1)
            print(f"  Loading Whisper model ({self.model_size})...")
            self._model = whisper.load_model(self.model_size)
        return self._model

    def transcribe(self, audio_path: Path) -> list[dict]:
        """
        Transcribe the audio file at audio_path.
        Returns a list of segments: [{start: float, end: float, text: str}]
        """
        print(f"[2/3] Transcribing with Whisper ({self.model_size} model)...")
        model  = self._load_model()
        result = model.transcribe(str(audio_path), verbose=False)
        segments = [
            {
                "start": seg["start"],
                "end":   seg["end"],
                "text":  seg["text"].strip(),
            }
            for seg in result["segments"]
        ]
        print(f"  Got {len(segments)} segments.")
        return segments

    @staticmethod
    def segments_to_text(segments: list[dict]) -> str:
        """Join all segment texts into a single string."""
        return " ".join(seg["text"] for seg in segments)
