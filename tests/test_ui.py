# tests/test_ui.py
# Unit tests for the run_pipeline() generator in ui.py.
# All external pipeline calls are mocked — no ffmpeg, Whisper, or Ollama needed.

import io
import zipfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ui import run_pipeline


# ── Helpers ───────────────────────────────────────────────────────────────────

SEGMENTS = [
    {"start": 0.0, "end": 3.0, "text": "Hello everyone"},
    {"start": 3.0, "end": 6.0, "text": "welcome to the meeting"},
]
RAW_TEXT      = "Hello everyone welcome to the meeting"
REFINED_TEXT  = "Hello everyone, welcome to the meeting."


def _settings(
    whisper_model="base",
    llama_model="llama3:8b",
    use_llama=True,
    use_speakers=True,
    use_summary=True,
    use_actions=True,
    language="",
):
    return dict(
        whisper_model=whisper_model,
        llama_model=llama_model,
        use_llama=use_llama,
        use_speakers=use_speakers,
        use_summary=use_summary,
        use_actions=use_actions,
        language=language,
    )


def _collect(gen):
    """Drain the generator and return (steps, results, out_dir)."""
    steps   = []
    results = {}
    out_dir = None
    for event, payload in gen:
        if event == "step":
            steps.append(payload)
        elif event == "done":
            results, out_dir = payload
    return steps, results, out_dir


@pytest.fixture
def mock_pipeline(tmp_path):
    """Patch every external class used inside run_pipeline."""
    out_dir = tmp_path / "test_video_20260101_120000"
    out_dir.mkdir()

    srt_path        = out_dir / "test_video_transcript.srt"
    transcript_path = out_dir / "test_video_transcript.txt"
    speakers_path   = out_dir / "test_video_speakers.txt"
    summary_path    = out_dir / "test_video_summary.txt"
    actions_path    = out_dir / "test_video_actions.txt"
    translation_path = out_dir / "test_video_translation_Hindi.txt"

    srt_path.write_text("1\n00:00:00,000 --> 00:00:03,000\nHello everyone\n")

    om = MagicMock()
    om.out_dir = out_dir
    om.path.side_effect = lambda s: out_dir / f"test_video{s}"

    def _save_text(content, suffix, label):
        mapping = {
            "_transcript.txt":     transcript_path,
            "_speakers.txt":       speakers_path,
            "_summary.txt":        summary_path,
            "_actions.txt":        actions_path,
            "_translation_Hindi.txt": translation_path,
        }
        return mapping.get(suffix, out_dir / f"test_video{suffix}")

    om.save_text.side_effect = _save_text
    om.save_srt.return_value = srt_path

    wt_instance = MagicMock()
    wt_instance.transcribe.return_value = SEGMENTS

    with patch("ui.OutputManager",       return_value=om)          as p_om, \
         patch("ui.AudioExtractor")                                 as p_ae, \
         patch("ui.WhisperTranscriber",  return_value=wt_instance) as p_wt, \
         patch("ui.LlamaClient")                                    as p_lc, \
         patch("ui.TranscriptRefiner")                              as p_ref, \
         patch("ui.SpeakerDetector")                                as p_spk, \
         patch("ui.TranscriptSummariser")                           as p_sum, \
         patch("ui.ActionItemExtractor")                            as p_act, \
         patch("ui.TranscriptTranslator")                           as p_tr:

        p_wt.segments_to_text = MagicMock(return_value=RAW_TEXT)
        p_ref.return_value.run.return_value        = REFINED_TEXT
        p_spk.return_value.run.return_value        = "Speaker 1: Hello everyone."
        p_sum.return_value.run.return_value        = "A brief summary."
        p_act.return_value.run.return_value        = "- Finish the report"
        p_tr.return_value.run.return_value         = "सभी का स्वागत है।"

        yield dict(
            om=p_om, ae=p_ae, wt=p_wt, lc=p_lc,
            refiner=p_ref, speakers=p_spk, summary=p_sum,
            actions=p_act, translator=p_tr,
            out_dir=out_dir, srt_path=srt_path,
        )


# ── Step message tests ────────────────────────────────────────────────────────

class TestRunPipelineSteps:
    """run_pipeline() yields human-readable step messages at each stage."""

    def test_yields_audio_extraction_step(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        steps, _, _ = _collect(run_pipeline(video, _settings()))
        assert any("ffmpeg" in s.lower() or "audio" in s.lower() for s in steps)

    def test_yields_whisper_step_with_model_name(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        steps, _, _ = _collect(run_pipeline(video, _settings(whisper_model="small")))
        assert any("small" in s for s in steps)

    def test_yields_llama_refinement_step(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        steps, _, _ = _collect(run_pipeline(video, _settings()))
        assert any("refin" in s.lower() for s in steps)

    def test_yields_save_step(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        steps, _, _ = _collect(run_pipeline(video, _settings()))
        assert any("saving" in s.lower() or "transcript" in s.lower() for s in steps)

    def test_no_llama_step_not_yielded_when_llama_disabled(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        steps, _, _ = _collect(run_pipeline(video, _settings(use_llama=False)))
        assert not any("refin" in s.lower() for s in steps)


# ── Results keys tests ────────────────────────────────────────────────────────

class TestRunPipelineResults:
    """run_pipeline() returns correct result keys depending on feature flags."""

    def test_always_includes_transcript_and_srt(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings()))
        assert "Transcript" in results
        assert "Subtitles (SRT)" in results

    def test_all_features_produces_five_result_keys(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings()))
        assert set(results.keys()) == {
            "Transcript", "Subtitles (SRT)", "Speakers", "Summary", "Action Items"
        }

    def test_no_llama_produces_only_transcript_and_srt(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings(use_llama=False)))
        assert set(results.keys()) == {"Transcript", "Subtitles (SRT)"}

    def test_no_speakers_excludes_speakers_key(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings(use_speakers=False)))
        assert "Speakers" not in results

    def test_no_summary_excludes_summary_key(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings(use_summary=False)))
        assert "Summary" not in results

    def test_no_actions_excludes_action_items_key(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings(use_actions=False)))
        assert "Action Items" not in results

    def test_translation_included_when_language_set(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings(language="Hindi")))
        assert "Translation (Hindi)" in results

    def test_translation_excluded_when_language_blank(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings(language="")))
        assert not any("Translation" in k for k in results)

    def test_results_values_are_path_and_content_tuples(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, results, _ = _collect(run_pipeline(video, _settings()))
        for label, value in results.items():
            assert isinstance(value, tuple), f"{label} should be a (path, content) tuple"
            path, content = value
            assert isinstance(path, Path)
            assert isinstance(content, str)

    def test_out_dir_returned_in_done_event(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _, _, out_dir = _collect(run_pipeline(video, _settings()))
        assert out_dir == mock_pipeline["out_dir"]


# ── Pipeline delegation tests ─────────────────────────────────────────────────

class TestRunPipelineDelegation:
    """run_pipeline() passes correct arguments to each pipeline class."""

    def test_whisper_transcriber_called_with_correct_model(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(whisper_model="medium")))
        mock_pipeline["wt"].assert_called_once_with(model_size="medium")

    def test_llama_client_created_with_correct_model(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(llama_model="mistral")))
        mock_pipeline["lc"].assert_called_once_with(model="mistral")

    def test_llama_client_not_created_when_use_llama_false(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(use_llama=False)))
        mock_pipeline["lc"].assert_not_called()

    def test_speaker_detector_not_called_when_disabled(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(use_speakers=False)))
        mock_pipeline["speakers"].return_value.run.assert_not_called()

    def test_summariser_not_called_when_disabled(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(use_summary=False)))
        mock_pipeline["summary"].return_value.run.assert_not_called()

    def test_action_extractor_not_called_when_disabled(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(use_actions=False)))
        mock_pipeline["actions"].return_value.run.assert_not_called()

    def test_translator_called_with_correct_language(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(language="Spanish")))
        mock_pipeline["translator"].return_value.run.assert_called_once_with(
            REFINED_TEXT, "Spanish"
        )

    def test_translator_not_called_when_no_language(self, tmp_path, mock_pipeline):
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings(language="")))
        mock_pipeline["translator"].return_value.run.assert_not_called()

    def test_audio_cleanup_called(self, tmp_path, mock_pipeline):
        """Temp audio file should be deleted after pipeline completes."""
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        audio_path = mock_pipeline["out_dir"] / "test_video_audio.wav"
        mock_pipeline["om"].return_value.path.return_value = audio_path
        audio_path.write_bytes(b"fake audio")
        _collect(run_pipeline(video, _settings()))
        assert not audio_path.exists()

    def test_refiner_receives_raw_text(self, tmp_path, mock_pipeline):
        """TranscriptRefiner should be called with raw Whisper text."""
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings()))
        mock_pipeline["refiner"].return_value.run.assert_called_once_with(RAW_TEXT)

    def test_downstream_enhancers_receive_refined_text(self, tmp_path, mock_pipeline):
        """SpeakerDetector and others should receive refined text, not raw."""
        video = tmp_path / "test_video.mp4"
        video.write_bytes(b"x")
        _collect(run_pipeline(video, _settings()))
        mock_pipeline["speakers"].return_value.run.assert_called_once_with(REFINED_TEXT)
        mock_pipeline["summary"].return_value.run.assert_called_once_with(REFINED_TEXT)
        mock_pipeline["actions"].return_value.run.assert_called_once_with(REFINED_TEXT)
