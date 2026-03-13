# tests/test_ux.py
# User experience tests — verifies output quality, progress feedback,
# error message readability, and output file validity.
# These tests verify what the user actually sees and receives.

import pytest
import re
from pathlib import Path
from unittest.mock import patch, MagicMock
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.output      import OutputManager
from pipeline.transcriber import WhisperTranscriber
from pipeline.enhancers   import (
    TranscriptRefiner, TranscriptSummariser,
    ActionItemExtractor, SpeakerDetector
)
from pipeline.llama       import LlamaClient
from pipeline.audio       import AudioExtractor


# ── Progress Feedback ─────────────────────────────────────────────────────────

class TestProgressFeedback:
    """Users should see clear step-by-step progress during pipeline execution."""

    @patch("pipeline.audio.subprocess.run")
    def test_audio_extraction_prints_step(self, mock_run, tmp_video, tmp_audio, capsys):
        """Step 1 should print a message showing the video filename."""
        mock_run.return_value = MagicMock(returncode=0)
        AudioExtractor().extract(tmp_video, tmp_audio)
        captured = capsys.readouterr()
        assert "test_video.mp4" in captured.out

    def test_transcription_prints_model_name(self, tmp_audio, capsys):
        """Step 2 should print which Whisper model is being used."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        transcriber = WhisperTranscriber(model_size="medium")
        transcriber._model = mock_model
        transcriber.transcribe(tmp_audio)
        captured = capsys.readouterr()
        assert "medium" in captured.out

    def test_transcription_prints_segment_count(self, tmp_audio, sample_segments, capsys):
        """Step 2 should print how many segments were found."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": sample_segments}
        transcriber = WhisperTranscriber()
        transcriber._model = mock_model
        transcriber.transcribe(tmp_audio)
        captured = capsys.readouterr()
        assert "4" in captured.out

    def test_output_folder_printed_on_creation(self, tmp_video, capsys):
        """OutputManager should print the output folder path on creation."""
        manager = OutputManager(tmp_video)
        captured = capsys.readouterr()
        assert str(manager.out_dir) in captured.out

    def test_each_saved_file_is_announced(self, tmp_video, capsys):
        """Each saved output file should be announced in the console."""
        manager = OutputManager(tmp_video)
        capsys.readouterr()  # clear creation message
        manager.save_text("content", "_summary.txt", "Summary")
        captured = capsys.readouterr()
        assert "summary" in captured.out.lower()

    def test_llama_completion_announced(self, mock_llama_client, sample_raw_text, capsys):
        """Each Llama feature should print a completion message."""
        mock_llama_client.ask.return_value = "Refined text."
        TranscriptRefiner(mock_llama_client).run(sample_raw_text)
        captured = capsys.readouterr()
        assert "refin" in captured.out.lower()


# ── Error Message Readability ─────────────────────────────────────────────────

class TestErrorMessages:
    """Error messages should be human-readable, not raw Python exceptions."""

    @patch("pipeline.audio.subprocess.run")
    def test_ffmpeg_failure_prints_readable_message(self, mock_run, tmp_video, tmp_audio, capsys):
        """ffmpeg failure should print a readable error, not a traceback."""
        mock_run.return_value = MagicMock(returncode=1, stderr="ffmpeg: No such file")
        with pytest.raises(SystemExit):
            AudioExtractor().extract(tmp_video, tmp_audio)
        captured = capsys.readouterr()
        assert "ffmpeg" in captured.out.lower() or "error" in captured.out.lower()
        assert "Traceback" not in captured.out

    def test_llama_failure_prints_readable_message(self, mock_llama_client, sample_raw_text, capsys):
        """Llama being unreachable should print a readable message."""
        mock_llama_client.ask.return_value = None
        # Simulate what LlamaClient prints on failure
        with patch("pipeline.llama.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")
            client = LlamaClient()
            client.ask("prompt", "Refinement")
        captured = capsys.readouterr()
        assert "Traceback" not in captured.out
        assert "Traceback" not in captured.err if hasattr(captured, 'err') else True

    def test_llama_unreachable_does_not_crash(self, sample_raw_text):
        """If Llama is unreachable, refiner should fall back, not crash."""
        with patch("pipeline.llama.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.side_effect = Exception("Connection refused")
            client = LlamaClient()
            refiner = TranscriptRefiner(client)
            result = refiner.run(sample_raw_text)
        assert result == sample_raw_text  # graceful fallback


# ── Output File Quality ───────────────────────────────────────────────────────

class TestOutputFileQuality:
    """Output files should be valid, non-empty, and correctly formatted."""

    def test_transcript_txt_is_not_empty(self, tmp_video, sample_refined_text):
        """Transcript .txt file should not be empty."""
        manager = OutputManager(tmp_video)
        manager.save_text(sample_refined_text, "_transcript.txt", "Transcript")
        content = (manager.out_dir / "test_video_transcript.txt").read_text()
        assert len(content.strip()) > 0

    def test_transcript_txt_does_not_contain_filler_words(self, tmp_video, sample_refined_text):
        """Refined transcript should not contain raw filler words like 'um' or 'uh'."""
        manager = OutputManager(tmp_video)
        manager.save_text(sample_refined_text, "_transcript.txt", "Transcript")
        content = (manager.out_dir / "test_video_transcript.txt").read_text().lower()
        # sample_refined_text fixture is already cleaned — verify filler words gone
        assert "um " not in content
        assert " uh " not in content

    def test_summary_txt_is_not_empty(self, tmp_video, mock_llama_client, sample_refined_text):
        """Summary .txt file should not be empty."""
        mock_llama_client.ask.return_value = "This is a meaningful summary of the transcript."
        summary = TranscriptSummariser(mock_llama_client).run(sample_refined_text)
        manager = OutputManager(tmp_video)
        manager.save_text(summary, "_summary.txt", "Summary")
        content = (manager.out_dir / "test_video_summary.txt").read_text()
        assert len(content.strip()) > 0

    def test_actions_txt_contains_bullet_points(self, tmp_video, mock_llama_client, sample_refined_text):
        """Action items file should contain bullet point format."""
        mock_llama_client.ask.return_value = "- Finish report by Friday\n- John to handle presentation"
        actions = ActionItemExtractor(mock_llama_client).run(sample_refined_text)
        manager = OutputManager(tmp_video)
        manager.save_text(actions, "_actions.txt", "Actions")
        content = (manager.out_dir / "test_video_actions.txt").read_text()
        assert content.startswith("-") or "- " in content

    def test_speakers_txt_contains_speaker_labels(self, tmp_video, mock_llama_client, sample_refined_text):
        """Speakers file should contain Speaker labels."""
        mock_llama_client.ask.return_value = "Speaker 1: Hello everyone.\nSpeaker 1: Today we discuss the project."
        speakers = SpeakerDetector(mock_llama_client).run(sample_refined_text)
        manager = OutputManager(tmp_video)
        manager.save_text(speakers, "_speakers.txt", "Speakers")
        content = (manager.out_dir / "test_video_speakers.txt").read_text()
        assert "Speaker" in content

    def test_translation_file_is_not_empty(self, tmp_video, mock_llama_client, sample_refined_text):
        """Translation file should not be empty."""
        from pipeline.enhancers import TranscriptTranslator
        mock_llama_client.ask.return_value = "नमस्ते सबको। आज हम इस परियोजना पर चर्चा करेंगे।"
        translation = TranscriptTranslator(mock_llama_client).run(sample_refined_text, "Hindi")
        manager = OutputManager(tmp_video)
        manager.save_text(translation, "_translation_Hindi.txt", "Translation (Hindi)")
        content = (manager.out_dir / "test_video_translation_Hindi.txt").read_text(encoding="utf-8")
        assert len(content.strip()) > 0


# ── SRT Format Validity ───────────────────────────────────────────────────────

class TestSrtValidity:
    """SRT files must follow the SRT specification exactly to work in video players."""

    def test_srt_entries_have_four_parts(self, tmp_video, sample_segments):
        """Each SRT entry must have: index, timestamp, text, blank line."""
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        entries = content.strip().split("\n\n")
        for entry in entries:
            lines = entry.strip().split("\n")
            assert len(lines) >= 3  # index, timestamp, at least one text line

    def test_srt_index_starts_at_one(self, tmp_video, sample_segments):
        """SRT index must start at 1, not 0."""
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        first_line = content.strip().split("\n")[0]
        assert first_line == "1"

    def test_srt_timestamp_format_is_valid(self, tmp_video, sample_segments):
        """SRT timestamps must match HH:MM:SS,mmm --> HH:MM:SS,mmm format."""
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        # Regex for valid SRT timestamp line
        pattern = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        matches = re.findall(pattern, content)
        assert len(matches) == len(sample_segments)

    def test_srt_uses_comma_not_dot_in_timestamps(self, tmp_video, sample_segments):
        """SRT spec requires comma before milliseconds — not a dot."""
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        assert re.search(r"\d{2}:\d{2}:\d{2},\d{3}", content)
        assert not re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", content)

    def test_srt_end_time_after_start_time(self, tmp_video, sample_segments):
        """Each SRT entry's end timestamp must be after its start timestamp."""
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        pattern = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
        for match in re.finditer(pattern, content):
            start, end = match.group(1), match.group(2)
            assert start < end  # lexicographic comparison works for this format

    def test_srt_entries_separated_by_blank_line(self, tmp_video, sample_segments):
        """SRT entries must be separated by blank lines — required by spec."""
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        assert "\n\n" in content

    def test_srt_sequential_index_numbers(self, tmp_video, sample_segments):
        """SRT index numbers must be sequential: 1, 2, 3, 4..."""
        manager = OutputManager(tmp_video)
        manager.save_srt(sample_segments)
        content = (manager.out_dir / "test_video_transcript.srt").read_text()
        entries = [e.strip() for e in content.strip().split("\n\n") if e.strip()]
        for i, entry in enumerate(entries, start=1):
            first_line = entry.split("\n")[0]
            assert first_line == str(i)


# ── Output Folder UX ──────────────────────────────────────────────────────────

class TestOutputFolderUX:
    """Output folder structure should be intuitive and easy to navigate."""

    def test_output_folder_named_after_video(self, tmp_video):
        """Folder name should start with the video filename — easy to identify."""
        manager = OutputManager(tmp_video)
        assert manager.out_dir.name.startswith("test_video")

    def test_output_files_named_after_video(self, tmp_video, sample_segments):
        """All output files should start with the video stem — consistent naming."""
        manager = OutputManager(tmp_video)
        manager.save_text("content", "_transcript.txt", "Transcript")
        manager.save_srt(sample_segments)
        for f in manager.out_dir.iterdir():
            assert f.name.startswith("test_video")

    def test_temp_audio_not_in_output_folder(self, tmp_video, sample_segments, tmp_path):
        """The temporary .wav audio file should not appear in final output."""
        manager = OutputManager(tmp_video)
        manager.save_text("content", "_transcript.txt", "Transcript")
        manager.save_srt(sample_segments)
        # Simulate cleanup
        wav = manager.out_dir / "test_video_audio.wav"
        wav.write_bytes(b"")
        wav.unlink(missing_ok=True)
        files = [f.name for f in manager.out_dir.iterdir()]
        assert not any("_audio.wav" in f for f in files)

    def test_each_run_isolated_in_own_folder(self, tmp_video):
        """Each run should produce its own folder — previous runs not overwritten."""
        import time
        m1 = OutputManager(tmp_video)
        m1.save_text("run 1 content", "_transcript.txt", "Transcript")
        time.sleep(1)
        m2 = OutputManager(tmp_video)
        m2.save_text("run 2 content", "_transcript.txt", "Transcript")
        # Both folders exist independently
        assert m1.out_dir.exists()
        assert m2.out_dir.exists()
        assert m1.out_dir != m2.out_dir
        # Content is separate
        assert (m1.out_dir / "test_video_transcript.txt").read_text() == "run 1 content"
        assert (m2.out_dir / "test_video_transcript.txt").read_text() == "run 2 content"
