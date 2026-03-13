# tests/test_output.py
# Unit tests for pipeline/output.py — OutputManager class.

import pytest
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.output import OutputManager


class TestOutputManagerInit:
    """Tests for OutputManager initialisation and folder creation."""

    def test_creates_output_folder(self, tmp_video):
        manager = OutputManager(tmp_video)
        assert manager.out_dir.exists()

    def test_folder_name_contains_video_stem(self, tmp_video):
        manager = OutputManager(tmp_video)
        assert "test_video" in manager.out_dir.name

    def test_folder_name_contains_timestamp(self, tmp_video):
        """Folder name should include a YYYYMMDD_HHMMSS timestamp."""
        manager = OutputManager(tmp_video)
        # timestamp part: last 15 chars e.g. 20260313_142305
        name_parts = manager.out_dir.name.split("_")
        assert len(name_parts) >= 3  # stem + date + time

    def test_uses_video_parent_as_default_base(self, tmp_video):
        manager = OutputManager(tmp_video)
        assert manager.out_dir.parent == tmp_video.parent

    def test_uses_custom_base_dir(self, tmp_video, tmp_path):
        custom_dir = tmp_path / "custom_output"
        manager = OutputManager(tmp_video, base_dir=custom_dir)
        assert manager.out_dir.parent == custom_dir

    def test_two_runs_create_different_folders(self, tmp_video):
        """Each OutputManager instance should create a unique folder."""
        import time
        m1 = OutputManager(tmp_video)
        time.sleep(1)
        m2 = OutputManager(tmp_video)
        assert m1.out_dir != m2.out_dir


class TestOutputManagerSaveText:
    """Tests for OutputManager.save_text()."""

    def test_saves_file_to_output_dir(self, tmp_video):
        manager = OutputManager(tmp_video)
        manager.save_text("hello world", "_transcript.txt", "Transcript")
        expected = manager.out_dir / "test_video_transcript.txt"
        assert expected.exists()

    def test_file_content_matches(self, tmp_video):
        manager = OutputManager(tmp_video)
        manager.save_text("hello world", "_transcript.txt", "Transcript")
        content = (manager.out_dir / "test_video_transcript.txt").read_text()
        assert content == "hello world"

    def test_returns_path(self, tmp_video):
        manager = OutputManager(tmp_video)
        result = manager.save_text("content", "_summary.txt", "Summary")
        assert isinstance(result, Path)

    def test_saves_with_correct_suffix(self, tmp_video):
        manager = OutputManager(tmp_video)
        manager.save_text("data", "_speakers.txt", "Speakers")
        assert (manager.out_dir / "test_video_speakers.txt").exists()

    def test_utf8_encoding(self, tmp_video):
        """save_text() should correctly save non-ASCII characters (e.g. Hindi)."""
        manager = OutputManager(tmp_video)
        hindi_text = "नमस्ते दुनिया"
        manager.save_text(hindi_text, "_translation_Hindi.txt", "Translation")
        content = (manager.out_dir / "test_video_translation_Hindi.txt").read_text(encoding="utf-8")
        assert content == hindi_text


class TestOutputManagerSaveSrt:
    """Tests for OutputManager.save_srt()."""

    def test_creates_srt_file(self, tmp_video, sample_segments):
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        assert (manager.out_dir / "test_video_transcript.srt").exists()

    def test_srt_contains_index_numbers(self, tmp_video, sample_segments):
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        assert "1\n" in content
        assert "2\n" in content

    def test_srt_contains_timestamps(self, tmp_video, sample_segments):
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        assert "-->" in content

    def test_srt_contains_text(self, tmp_video, sample_segments):
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        assert "hello everyone" in content

    def test_srt_empty_segments(self, tmp_video):
        """save_srt() should handle empty segments without crashing."""
        manager = OutputManager(tmp_video)
        manager.save_srt([])
        assert (manager.out_dir / "test_video_transcript.srt").exists()


class TestSrtTimeFormat:
    """Tests for OutputManager._srt_time() static method."""

    def test_zero_seconds(self):
        assert OutputManager._srt_time(0.0) == "00:00:00,000"

    def test_milliseconds(self):
        assert OutputManager._srt_time(0.5) == "00:00:00,500"

    def test_seconds(self):
        assert OutputManager._srt_time(5.0) == "00:00:05,000"

    def test_minutes(self):
        assert OutputManager._srt_time(90.0) == "00:01:30,000"

    def test_hours(self):
        assert OutputManager._srt_time(3661.0) == "01:01:01,000"

    def test_comma_separator(self):
        """SRT format requires comma before milliseconds — not a dot."""
        result = OutputManager._srt_time(1.5)
        assert "," in result
        assert "." not in result

    def test_zero_padded(self):
        """All fields should be zero-padded to their correct widths."""
        result = OutputManager._srt_time(1.0)
        assert result == "00:00:01,000"
