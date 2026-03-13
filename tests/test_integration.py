# tests/test_integration.py
# Integration tests — classes working together.
# End-to-end test — full pipeline run with all external calls mocked.

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.audio       import AudioExtractor
from pipeline.transcriber import WhisperTranscriber
from pipeline.llama       import LlamaClient
from pipeline.enhancers   import TranscriptRefiner, SpeakerDetector, TranscriptSummariser, ActionItemExtractor
from pipeline.output      import OutputManager


# ── Integration: Transcriber + OutputManager ──────────────────────────────────

class TestTranscriberOutputIntegration:
    """WhisperTranscriber segments flow correctly into OutputManager.save_srt()."""

    def test_segments_from_transcriber_saved_to_srt(self, tmp_video, tmp_audio, sample_segments):
        """Segments returned by transcriber should produce a valid SRT file."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": sample_segments}

        transcriber = WhisperTranscriber()
        transcriber._model = mock_model
        segments = transcriber.transcribe(tmp_audio)

        manager = OutputManager(tmp_video)
        manager.save_srt(segments)

        srt_path = manager.out_dir / "test_video_transcript.srt"
        content  = srt_path.read_text()

        assert "hello everyone" in content
        assert "-->" in content
        assert "1\n" in content

    def test_segments_to_text_flows_into_refiner(self, mock_llama_client, sample_segments):
        """segments_to_text() output should be passed correctly to TranscriptRefiner."""
        mock_llama_client.ask.return_value = "Refined text."
        raw_text = WhisperTranscriber.segments_to_text(sample_segments)
        result   = TranscriptRefiner(mock_llama_client).run(raw_text)
        assert result == "Refined text."
        prompt = mock_llama_client.ask.call_args[0][0]
        assert "hello everyone" in prompt


# ── Integration: All Enhancers share one LlamaClient ─────────────────────────

class TestSharedLlamaClientIntegration:
    """All enhancers should use the same LlamaClient instance (dependency injection)."""

    def test_all_enhancers_use_same_client(self, mock_llama_client, sample_refined_text):
        """Each enhancer's ask() call should go through the shared client."""
        mock_llama_client.ask.return_value = "response"

        TranscriptRefiner(mock_llama_client).run(sample_refined_text)
        SpeakerDetector(mock_llama_client).run(sample_refined_text)
        TranscriptSummariser(mock_llama_client).run(sample_refined_text)
        ActionItemExtractor(mock_llama_client).run(sample_refined_text)

        assert mock_llama_client.ask.call_count == 4

    def test_enhancers_receive_refined_text(self, mock_llama_client, sample_raw_text, sample_refined_text):
        """Downstream enhancers should receive refined text, not raw text."""
        mock_llama_client.ask.return_value = sample_refined_text

        refined = TranscriptRefiner(mock_llama_client).run(sample_raw_text)

        # reset call count, now test that downstream enhancers get refined text
        mock_llama_client.ask.reset_mock()
        mock_llama_client.ask.return_value = "summary"
        TranscriptSummariser(mock_llama_client).run(refined)

        prompt = mock_llama_client.ask.call_args[0][0]
        assert sample_refined_text in prompt


# ── Integration: OutputManager saves all feature outputs ─────────────────────

class TestOutputManagerAllFiles:
    """All pipeline outputs are saved correctly into the same timestamped folder."""

    def test_all_output_files_in_same_folder(self, tmp_video, sample_segments, sample_refined_text):
        manager = OutputManager(tmp_video)
        manager.save_text(sample_refined_text, "_transcript.txt", "Transcript")
        manager.save_text("Speaker 1: Hello.", "_speakers.txt",   "Speakers")
        manager.save_text("A brief summary.", "_summary.txt",     "Summary")
        manager.save_text("- Finish report", "_actions.txt",      "Actions")
        manager.save_srt(sample_segments)

        files = list(manager.out_dir.iterdir())
        names = [f.name for f in files]

        assert any("transcript.txt" in n for n in names)
        assert any("speakers"       in n for n in names)
        assert any("summary"        in n for n in names)
        assert any("actions"        in n for n in names)
        assert any("transcript.srt" in n for n in names)
        assert len(files) == 5


# ── End-to-End: Full pipeline with all external calls mocked ──────────────────

class TestEndToEnd:
    """
    Full pipeline run — mocks ffmpeg, Whisper, and Ollama.
    Tests that all components wire together correctly without real I/O.
    """

    def _mock_ollama_response(self, text):
        payload = json.dumps({"response": text}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("pipeline.llama.urllib.request.urlopen")
    @patch("pipeline.audio.subprocess.run")
    def test_full_pipeline_creates_all_output_files(
        self, mock_subprocess, mock_urlopen, tmp_video, tmp_path, sample_segments
    ):
        """
        End-to-end: video → audio extraction → transcription → all Llama
        enhancements → all output files created in timestamped folder.
        """
        # Mock ffmpeg success
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Mock Ollama responses for all 4 Llama calls
        mock_urlopen.return_value = self._mock_ollama_response("Mocked Llama output.")

        # Mock Whisper model
        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe.return_value = {"segments": sample_segments}

        # Run pipeline components in order (mirrors transcribe.py main())
        audio_path = tmp_path / "test_audio.wav"
        audio_path.write_bytes(b"")          # create fake wav

        output = OutputManager(tmp_video, base_dir=tmp_path)

        AudioExtractor().extract(tmp_video, audio_path)

        transcriber = WhisperTranscriber(model_size="base")
        transcriber._model = mock_whisper_model   # inject mock directly
        segments    = transcriber.transcribe(audio_path)
        raw_text    = WhisperTranscriber.segments_to_text(segments)

        client = LlamaClient()

        refined_text  = TranscriptRefiner(client).run(raw_text)
        speakers_text = SpeakerDetector(client).run(refined_text)
        summary_text  = TranscriptSummariser(client).run(refined_text)
        actions_text  = ActionItemExtractor(client).run(refined_text)

        output.save_text(refined_text,  "_transcript.txt", "Transcript")
        output.save_text(speakers_text, "_speakers.txt",   "Speakers")
        output.save_text(summary_text,  "_summary.txt",    "Summary")
        output.save_text(actions_text,  "_actions.txt",    "Actions")
        output.save_srt(segments)

        # Verify all output files exist
        files = [f.name for f in output.out_dir.iterdir()]
        assert any("transcript.txt" in f for f in files)
        assert any("speakers"       in f for f in files)
        assert any("summary"        in f for f in files)
        assert any("actions"        in f for f in files)
        assert any("transcript.srt" in f for f in files)

    @patch("pipeline.llama.urllib.request.urlopen")
    @patch("pipeline.audio.subprocess.run")
    def test_pipeline_with_llama_unreachable(
        self, mock_subprocess, mock_urlopen, tmp_video, tmp_path, sample_segments
    ):
        """
        End-to-end: if Llama is unreachable, pipeline should still complete
        with raw Whisper text as fallback — no crash.
        """
        mock_subprocess.return_value = MagicMock(returncode=0)
        mock_urlopen.side_effect = Exception("Connection refused")

        mock_whisper_model = MagicMock()
        mock_whisper_model.transcribe.return_value = {"segments": sample_segments}

        audio_path = tmp_path / "test_audio.wav"
        audio_path.write_bytes(b"")

        output      = OutputManager(tmp_video, base_dir=tmp_path)
        transcriber = WhisperTranscriber()
        transcriber._model = mock_whisper_model   # inject mock directly
        segments    = transcriber.transcribe(audio_path)
        raw_text    = WhisperTranscriber.segments_to_text(segments)
        client      = LlamaClient()

        # All enhancers should fall back gracefully
        refined_text = TranscriptRefiner(client).run(raw_text)
        assert refined_text == raw_text      # fallback to raw

        summary = TranscriptSummariser(client).run(refined_text)
        assert "could not" in summary.lower()  # fallback message

        output.save_text(refined_text, "_transcript.txt", "Transcript")
        output.save_srt(segments)

        assert (output.out_dir / "test_video_transcript.txt").exists()
        assert (output.out_dir / "test_video_transcript.srt").exists()

    @patch("pipeline.audio.subprocess.run")
    def test_pipeline_exits_if_video_not_found(self, mock_subprocess, tmp_path):
        """Pipeline should exit cleanly if ffmpeg fails (bad video path)."""
        mock_subprocess.return_value = MagicMock(returncode=1, stderr="No such file")
        fake_video = tmp_path / "nonexistent.mp4"
        fake_video.write_bytes(b"")
        audio_path = tmp_path / "out.wav"

        with pytest.raises(SystemExit) as exc:
            AudioExtractor().extract(fake_video, audio_path)
        assert exc.value.code == 1
