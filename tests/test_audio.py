# tests/test_audio.py
# Unit tests for pipeline/audio.py — AudioExtractor class.
# All ffmpeg calls are mocked — no real audio processing happens.

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.audio import AudioExtractor


class TestAudioExtractorInit:
    """Tests for AudioExtractor initialisation."""

    def test_default_sample_rate(self):
        extractor = AudioExtractor()
        assert extractor.sample_rate == 16000

    def test_default_channels(self):
        extractor = AudioExtractor()
        assert extractor.channels == 1

    def test_custom_sample_rate(self):
        extractor = AudioExtractor(sample_rate=44100)
        assert extractor.sample_rate == 44100

    def test_custom_channels(self):
        extractor = AudioExtractor(channels=2)
        assert extractor.channels == 2


class TestAudioExtractorExtract:
    """Tests for AudioExtractor.extract()."""

    @patch("pipeline.audio.subprocess.run")
    def test_extract_calls_ffmpeg(self, mock_run, tmp_video, tmp_audio):
        """extract() should call subprocess.run with ffmpeg."""
        mock_run.return_value = MagicMock(returncode=0)
        AudioExtractor().extract(tmp_video, tmp_audio)
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"

    @patch("pipeline.audio.subprocess.run")
    def test_extract_uses_correct_sample_rate(self, mock_run, tmp_video, tmp_audio):
        """extract() should pass the configured sample rate to ffmpeg."""
        mock_run.return_value = MagicMock(returncode=0)
        AudioExtractor(sample_rate=22050).extract(tmp_video, tmp_audio)
        cmd = mock_run.call_args[0][0]
        assert "22050" in cmd

    @patch("pipeline.audio.subprocess.run")
    def test_extract_includes_no_video_flag(self, mock_run, tmp_video, tmp_audio):
        """extract() should pass -vn flag to strip video track."""
        mock_run.return_value = MagicMock(returncode=0)
        AudioExtractor().extract(tmp_video, tmp_audio)
        cmd = mock_run.call_args[0][0]
        assert "-vn" in cmd

    @patch("pipeline.audio.subprocess.run")
    def test_extract_includes_mono_flag(self, mock_run, tmp_video, tmp_audio):
        """-ac 1 should be in the ffmpeg command for mono output."""
        mock_run.return_value = MagicMock(returncode=0)
        AudioExtractor().extract(tmp_video, tmp_audio)
        cmd = mock_run.call_args[0][0]
        assert "-ac" in cmd
        assert "1" in cmd

    @patch("pipeline.audio.subprocess.run")
    def test_extract_exits_on_ffmpeg_failure(self, mock_run, tmp_video, tmp_audio):
        """extract() should call sys.exit(1) if ffmpeg returns non-zero."""
        mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg error")
        with pytest.raises(SystemExit) as exc:
            AudioExtractor().extract(tmp_video, tmp_audio)
        assert exc.value.code == 1

    @patch("pipeline.audio.subprocess.run")
    def test_extract_passes_input_path(self, mock_run, tmp_video, tmp_audio):
        """extract() should pass the video path as -i argument."""
        mock_run.return_value = MagicMock(returncode=0)
        AudioExtractor().extract(tmp_video, tmp_audio)
        cmd = mock_run.call_args[0][0]
        assert str(tmp_video) in cmd

    @patch("pipeline.audio.subprocess.run")
    def test_extract_passes_output_path(self, mock_run, tmp_video, tmp_audio):
        """extract() should pass the audio output path to ffmpeg."""
        mock_run.return_value = MagicMock(returncode=0)
        AudioExtractor().extract(tmp_video, tmp_audio)
        cmd = mock_run.call_args[0][0]
        assert str(tmp_audio) in cmd
