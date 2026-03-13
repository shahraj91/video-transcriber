# ── pipeline/output.py ───────────────────────────────────────────────────────
# Responsible for all file I/O — saving transcripts, SRT files, and creating
# the timestamped output folder for each run.

from datetime import datetime
from pathlib import Path


class OutputManager:
    """
    Creates a timestamped output folder and saves all pipeline outputs.
    Folder name format: {video_stem}_{YYYYMMDD_HHMMSS}
    """

    def __init__(self, video_path: Path, base_dir: Path | None = None):
        self.video_path = video_path
        self.stem       = video_path.stem
        timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        base            = base_dir or video_path.parent
        self.out_dir    = base / f"{self.stem}_{timestamp}"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output folder: {self.out_dir}")

    # ── Path helpers ──────────────────────────────────────────────────────────

    def path(self, suffix: str) -> Path:
        """Return a full output path for a given suffix e.g. '_transcript.txt'"""
        return self.out_dir / f"{self.stem}{suffix}"

    # ── Savers ────────────────────────────────────────────────────────────────

    def save_text(self, content: str, suffix: str, label: str) -> Path:
        """Save a plain text file. Returns the path it was saved to."""
        out_path = self.path(suffix)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  {label:<20} -> {out_path.name}")
        return out_path

    def save_srt(self, segments: list[dict]) -> Path:
        """Save an SRT subtitle file from Whisper segments."""
        out_path = self.path("_transcript.srt")
        lines = []
        for i, seg in enumerate(segments, start=1):
            lines.append(str(i))
            lines.append(f"{self._srt_time(seg['start'])} --> {self._srt_time(seg['end'])}")
            lines.append(seg["text"])
            lines.append("")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  {'Subtitles':<20} -> {out_path.name}")
        return out_path

    # ── SRT time formatter ────────────────────────────────────────────────────

    @staticmethod
    def _srt_time(seconds: float) -> str:
        """Convert float seconds to SRT timestamp format HH:MM:SS,mmm"""
        ms = int((seconds % 1) * 1000)
        s  = int(seconds) % 60
        m  = (int(seconds) // 60) % 60
        h  = int(seconds) // 3600
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
