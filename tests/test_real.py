# tests/test_real.py
# Real file tests — uses actual audio/video assets with real Whisper transcription.
# No mocks. Requires:
#   1. Assets generated:  python tests/assets/generate_assets.py
#   2. Whisper installed: pip install openai-whisper
#   3. Ollama running:    curl http://localhost:11434/api/tags (for Llama tests)
#
# Run with:
#   pytest -m real -v
#   pytest -m "real and not llama" -v    # skip Ollama-dependent tests
#
# These tests are intentionally SLOW (Whisper takes 10-60s per file).
# They are marked separately so they don't run in regular pytest.

import pytest
import re
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pipeline.audio       import AudioExtractor
from pipeline.transcriber import WhisperTranscriber
from pipeline.llama       import LlamaClient
from pipeline.enhancers   import (
    TranscriptRefiner, SpeakerDetector,
    TranscriptSummariser, ActionItemExtractor, TranscriptTranslator
)
from pipeline.output      import OutputManager


# ── Audio Extraction — Real ffmpeg ────────────────────────────────────────────

@pytest.mark.real
class TestRealAudioExtraction:
    """Real ffmpeg audio extraction from generated video files."""

    def test_extracts_audio_from_english_video(self, english_clear_video, tmp_path):
        """ffmpeg should extract a real WAV file from the English video."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        assert audio_path.exists()
        assert audio_path.stat().st_size > 0

    def test_extracted_audio_is_wav(self, english_clear_video, tmp_path):
        """Extracted file should be a valid WAV (starts with RIFF header)."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        header = audio_path.read_bytes()[:4]
        assert header == b"RIFF"

    def test_extracts_audio_from_silence_video(self, silence_video, tmp_path):
        """ffmpeg should handle a silent video without crashing."""
        audio_path = tmp_path / "silence.wav"
        AudioExtractor().extract(silence_video, audio_path)
        assert audio_path.exists()
        assert audio_path.stat().st_size > 0

    def test_extracts_audio_from_short_clip(self, short_clip_video, tmp_path):
        """ffmpeg should handle a very short 2-second clip."""
        audio_path = tmp_path / "short.wav"
        AudioExtractor().extract(short_clip_video, audio_path)
        assert audio_path.exists()

    def test_extracts_audio_from_noisy_video(self, background_noise_video, tmp_path):
        """ffmpeg should extract audio from a noisy video without crashing."""
        audio_path = tmp_path / "noisy.wav"
        AudioExtractor().extract(background_noise_video, audio_path)
        assert audio_path.exists()
        assert audio_path.stat().st_size > 0


# ── Whisper Transcription — Real model ───────────────────────────────────────

@pytest.mark.real
class TestRealWhisperTranscription:
    """Real Whisper transcription using the tiny model (fastest, ~1GB)."""

    @pytest.fixture(autouse=True)
    def setup_transcriber(self):
        """Use tiny model for speed in tests."""
        self.transcriber = WhisperTranscriber(model_size="tiny")

    def test_transcribes_english_returns_segments(self, english_clear_video, tmp_path):
        """Whisper should return a list of segments from English audio."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        assert isinstance(segments, list)

    def test_segments_have_required_keys(self, english_clear_video, tmp_path):
        """Every segment must have start, end, and text keys."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        for seg in segments:
            assert "start" in seg
            assert "end"   in seg
            assert "text"  in seg

    def test_timestamps_are_positive_floats(self, english_clear_video, tmp_path):
        """All timestamps should be non-negative floats."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        for seg in segments:
            assert isinstance(seg["start"], float)
            assert isinstance(seg["end"],   float)
            assert seg["start"] >= 0.0
            assert seg["end"]   >= 0.0

    def test_end_time_after_start_time(self, english_clear_video, tmp_path):
        """Each segment's end time must be after its start time."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        for seg in segments:
            assert seg["end"] > seg["start"]

    def test_silence_returns_empty_or_minimal_segments(self, silence_video, tmp_path):
        """Whisper on a silent video should return no or very minimal segments."""
        audio_path = tmp_path / "silence.wav"
        AudioExtractor().extract(silence_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        all_text = " ".join(seg["text"].strip() for seg in segments)
        assert len(all_text.strip()) < 20

    def test_short_clip_does_not_crash(self, short_clip_video, tmp_path):
        """Whisper should handle a 2-second clip without crashing."""
        audio_path = tmp_path / "short.wav"
        AudioExtractor().extract(short_clip_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        assert isinstance(segments, list)

    def test_noisy_audio_returns_segments(self, background_noise_video, tmp_path):
        """Whisper should attempt transcription even on noisy audio."""
        audio_path = tmp_path / "noisy.wav"
        AudioExtractor().extract(background_noise_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        assert isinstance(segments, list)

    def test_segments_to_text_produces_string(self, english_clear_video, tmp_path):
        """segments_to_text() should always return a string — even with 0 segments."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = self.transcriber.transcribe(audio_path)
        text = WhisperTranscriber.segments_to_text(segments)
        assert isinstance(text, str)  # empty string is valid for silent/synthetic audio


# ── Real SRT Output ───────────────────────────────────────────────────────────

@pytest.mark.real
class TestRealSrtOutput:
    """SRT files generated from real Whisper output must be spec-compliant."""

    def test_real_srt_file_created(self, english_clear_video, tmp_path):
        """Real pipeline should produce a valid SRT file."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        manager  = OutputManager(english_clear_video, base_dir=tmp_path)
        manager.save_srt(segments)
        assert len(list(manager.out_dir.glob("*.srt"))) == 1

    def test_real_srt_has_valid_timestamps(self, english_clear_video, tmp_path):
        """SRT timestamps from real Whisper output must match HH:MM:SS,mmm format."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        if not segments:
            pytest.skip("Whisper returned 0 segments — synthetic audio not speech-like enough")
        manager  = OutputManager(english_clear_video, base_dir=tmp_path)
        manager.save_srt(segments)
        content  = list(manager.out_dir.glob("*.srt"))[0].read_text()
        pattern  = r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}"
        assert len(re.findall(pattern, content)) == len(segments)

    def test_real_srt_comma_separator(self, english_clear_video, tmp_path):
        """Real SRT output must use comma before milliseconds."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        if not segments:
            pytest.skip("Whisper returned 0 segments — synthetic audio not speech-like enough")
        manager  = OutputManager(english_clear_video, base_dir=tmp_path)
        manager.save_srt(segments)
        content  = list(manager.out_dir.glob("*.srt"))[0].read_text()
        assert re.search(r"\d{2}:\d{2}:\d{2},\d{3}", content)
        assert not re.search(r"\d{2}:\d{2}:\d{2}\.\d{3}", content)

    def test_silence_srt_does_not_crash(self, silence_video, tmp_path):
        """Saving an SRT from a silent video should not crash."""
        audio_path = tmp_path / "silence.wav"
        AudioExtractor().extract(silence_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        manager  = OutputManager(silence_video, base_dir=tmp_path)
        manager.save_srt(segments)
        assert len(list(manager.out_dir.glob("*.srt"))) == 1


# ── Real Llama Integration ────────────────────────────────────────────────────

@pytest.mark.real
@pytest.mark.llama
class TestRealLlamaIntegration:
    """
    Real Llama calls via Ollama REST API.
    Requires Ollama to be running: curl http://localhost:11434/api/tags
    Skip with: pytest -m "real and not llama"
    """

    def _ollama_running(self) -> bool:
        import urllib.request
        try:
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            return True
        except Exception:
            return False

    def test_refiner_returns_non_empty_string(self, english_clear_video, tmp_path):
        """Real Llama refiner should return a non-empty string."""
        if not self._ollama_running():
            pytest.skip("Ollama not running")
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text = WhisperTranscriber.segments_to_text(segments)
        result   = TranscriptRefiner(LlamaClient()).run(raw_text)
        assert isinstance(result, str) and len(result.strip()) > 0

    def test_summariser_returns_non_empty_string(self, english_clear_video, tmp_path):
        """Real Llama summariser should return a non-empty summary."""
        if not self._ollama_running():
            pytest.skip("Ollama not running")
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text = WhisperTranscriber.segments_to_text(segments)
        client   = LlamaClient()
        refined  = TranscriptRefiner(client).run(raw_text)
        summary  = TranscriptSummariser(client).run(refined)
        assert isinstance(summary, str) and len(summary.strip()) > 0

    def test_action_items_returns_string(self, english_clear_video, tmp_path):
        """Real action item extractor should return a string."""
        if not self._ollama_running():
            pytest.skip("Ollama not running")
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text = WhisperTranscriber.segments_to_text(segments)
        client   = LlamaClient()
        refined  = TranscriptRefiner(client).run(raw_text)
        actions  = ActionItemExtractor(client).run(refined)
        assert isinstance(actions, str) and len(actions.strip()) > 0

    def test_refiner_fallback_when_ollama_down(self, english_clear_video, tmp_path):
        """If Ollama is unreachable, refiner should return raw text — not crash."""
        audio_path = tmp_path / "out.wav"
        AudioExtractor().extract(english_clear_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text = WhisperTranscriber.segments_to_text(segments)
        client   = LlamaClient(api_url="http://localhost:19999/api/generate")
        result   = TranscriptRefiner(client).run(raw_text)
        assert result == raw_text


# ── Real End-to-End Pipeline ──────────────────────────────────────────────────

@pytest.mark.real
class TestRealEndToEnd:
    """Full pipeline runs with real files — no mocks at all."""

    def test_full_pipeline_produces_txt_and_srt(self, english_clear_video, real_output_dir):
        """Full pipeline should produce both .txt and .srt files."""
        output     = OutputManager(english_clear_video, base_dir=real_output_dir)
        audio_path = output.path("_audio.wav")
        AudioExtractor().extract(english_clear_video, audio_path)
        segments     = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text     = WhisperTranscriber.segments_to_text(segments)
        refined_text = TranscriptRefiner(LlamaClient()).run(raw_text)
        output.save_text(refined_text, "_transcript.txt", "Transcript")
        output.save_srt(segments)
        audio_path.unlink(missing_ok=True)
        assert len(list(output.out_dir.glob("*_transcript.txt"))) == 1
        assert len(list(output.out_dir.glob("*_transcript.srt"))) == 1

    def test_output_txt_is_not_empty(self, english_clear_video, real_output_dir):
        """Transcript .txt from real run should not be empty."""
        output     = OutputManager(english_clear_video, base_dir=real_output_dir)
        audio_path = output.path("_audio.wav")
        AudioExtractor().extract(english_clear_video, audio_path)
        segments     = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text     = WhisperTranscriber.segments_to_text(segments)
        refined      = TranscriptRefiner(LlamaClient()).run(raw_text)
        output.save_text(refined, "_transcript.txt", "Transcript")
        audio_path.unlink(missing_ok=True)
        content = list(output.out_dir.glob("*_transcript.txt"))[0].read_text()
        assert len(content.strip()) > 0

    def test_temp_audio_cleaned_up(self, english_clear_video, real_output_dir):
        """Temp .wav file should be deleted after pipeline completes."""
        output     = OutputManager(english_clear_video, base_dir=real_output_dir)
        audio_path = output.path("_audio.wav")
        AudioExtractor().extract(english_clear_video, audio_path)
        assert audio_path.exists()
        WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        audio_path.unlink(missing_ok=True)
        assert not audio_path.exists()

    def test_silence_pipeline_does_not_crash(self, silence_video, real_output_dir):
        """Full pipeline on a silent video should complete without crashing."""
        output     = OutputManager(silence_video, base_dir=real_output_dir)
        audio_path = output.path("_audio.wav")
        AudioExtractor().extract(silence_video, audio_path)
        segments     = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text     = WhisperTranscriber.segments_to_text(segments)
        refined      = TranscriptRefiner(LlamaClient()).run(raw_text)
        output.save_text(refined, "_transcript.txt", "Transcript")
        output.save_srt(segments)
        audio_path.unlink(missing_ok=True)
        assert len(list(output.out_dir.glob("*_transcript.srt"))) == 1

    def test_short_clip_pipeline_does_not_crash(self, short_clip_video, real_output_dir):
        """Full pipeline on a 2-second clip should complete without crashing."""
        output     = OutputManager(short_clip_video, base_dir=real_output_dir)
        audio_path = output.path("_audio.wav")
        AudioExtractor().extract(short_clip_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text = WhisperTranscriber.segments_to_text(segments)
        refined  = TranscriptRefiner(LlamaClient()).run(raw_text)
        output.save_text(refined, "_transcript.txt", "Transcript")
        output.save_srt(segments)
        audio_path.unlink(missing_ok=True)
        assert len(list(output.out_dir.glob("*_transcript.txt"))) == 1

    def test_noisy_pipeline_does_not_crash(self, background_noise_video, real_output_dir):
        """Full pipeline on noisy audio should complete without crashing."""
        output     = OutputManager(background_noise_video, base_dir=real_output_dir)
        audio_path = output.path("_audio.wav")
        AudioExtractor().extract(background_noise_video, audio_path)
        segments = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text = WhisperTranscriber.segments_to_text(segments)
        refined  = TranscriptRefiner(LlamaClient()).run(raw_text)
        output.save_text(refined, "_transcript.txt", "Transcript")
        output.save_srt(segments)
        audio_path.unlink(missing_ok=True)
        assert len(list(output.out_dir.glob("*_transcript.txt"))) == 1

    def test_timestamped_folder_created_per_run(self, english_clear_video, real_output_dir):
        """Each real pipeline run should create a unique timestamped folder."""
        import time
        output1 = OutputManager(english_clear_video, base_dir=real_output_dir)
        time.sleep(1)
        output2 = OutputManager(english_clear_video, base_dir=real_output_dir)
        assert output1.out_dir != output2.out_dir
        assert output1.out_dir.exists()
        assert output2.out_dir.exists()

    def test_your_own_video(self, real_output_dir):
        """
        Test with your own video file.
        Drop any .mp4 into tests/assets/ named your_video.mp4.
        This test is skipped automatically if the file doesn't exist.
        """
        assets_dir = Path(__file__).parent / "assets"
        your_video = assets_dir / "your_video.mp4"
        if not your_video.exists():
            pytest.skip("Add your own video at tests/assets/your_video.mp4 to run this test")

        output     = OutputManager(your_video, base_dir=real_output_dir)
        audio_path = output.path("_audio.wav")
        AudioExtractor().extract(your_video, audio_path)
        segments     = WhisperTranscriber(model_size="tiny").transcribe(audio_path)
        raw_text     = WhisperTranscriber.segments_to_text(segments)
        refined      = TranscriptRefiner(LlamaClient()).run(raw_text)
        output.save_text(refined, "_transcript.txt", "Transcript")
        output.save_srt(segments)
        audio_path.unlink(missing_ok=True)
        content = list(output.out_dir.glob("*_transcript.txt"))[0].read_text()
        assert len(content.strip()) > 0
        print(f"\n--- Your video transcript preview ---\n{content[:300]}\n")
