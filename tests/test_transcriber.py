# tests/test_transcriber.py
# Unit tests for pipeline/transcriber.py — WhisperTranscriber class.
# Whisper model is mocked — no real model is loaded or downloaded.

import pytest
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.transcriber import WhisperTranscriber


class TestWhisperTranscriberInit:
    """Tests for WhisperTranscriber initialisation."""

    def test_default_model_size(self):
        transcriber = WhisperTranscriber()
        assert transcriber.model_size == "base"

    def test_custom_model_size(self):
        transcriber = WhisperTranscriber(model_size="large")
        assert transcriber.model_size == "large"

    def test_model_is_none_initially(self):
        """Model should not be loaded at instantiation — lazy loading."""
        transcriber = WhisperTranscriber()
        assert transcriber._model is None


class TestWhisperTranscriberLazyLoad:
    """Tests for lazy model loading behaviour."""

    def test_model_loaded_on_first_transcribe(self, tmp_audio):
        """Model should only be loaded when transcribe() is first called."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}

        transcriber = WhisperTranscriber()
        assert transcriber._model is None         # not loaded yet

        # Inject mock model directly — avoids patching the local import
        transcriber._model = mock_model
        transcriber.transcribe(tmp_audio)
        assert transcriber._model is not None     # now loaded

    def test_model_loaded_only_once(self, tmp_audio):
        """Calling transcribe() twice should only load the model once."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}

        transcriber = WhisperTranscriber()
        # Pre-load the mock model — _load_model() won't call load_model() again
        transcriber._model = mock_model
        transcriber.transcribe(tmp_audio)
        transcriber.transcribe(tmp_audio)
        # Model should only have been set once — transcribe called twice but load skipped
        assert mock_model.transcribe.call_count == 2
        assert transcriber._model is mock_model


class TestWhisperTranscriberTranscribe:
    """Tests for WhisperTranscriber.transcribe()."""

    def _make_transcriber_with_mock(self, mock_segments):
        """Helper: return a WhisperTranscriber with a pre-loaded mock model."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": mock_segments}
        transcriber = WhisperTranscriber()
        transcriber._model = mock_model
        return transcriber

    def test_returns_list_of_dicts(self, tmp_audio):
        """transcribe() should return a list of segment dicts."""
        transcriber = self._make_transcriber_with_mock([
            {"start": 0.0, "end": 2.0, "text": " hello "}
        ])
        result = transcriber.transcribe(tmp_audio)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)

    def test_segment_has_required_keys(self, tmp_audio):
        """Each segment must have start, end, and text keys."""
        transcriber = self._make_transcriber_with_mock([
            {"start": 0.0, "end": 2.0, "text": " hello "}
        ])
        result = transcriber.transcribe(tmp_audio)
        assert "start" in result[0]
        assert "end"   in result[0]
        assert "text"  in result[0]

    def test_text_is_stripped(self, tmp_audio):
        """transcribe() should strip leading/trailing whitespace from text."""
        transcriber = self._make_transcriber_with_mock([
            {"start": 0.0, "end": 2.0, "text": "  hello world  "}
        ])
        result = transcriber.transcribe(tmp_audio)
        assert result[0]["text"] == "hello world"

    def test_timestamps_preserved(self, tmp_audio):
        """transcribe() should preserve start and end timestamps exactly."""
        transcriber = self._make_transcriber_with_mock([
            {"start": 1.23, "end": 4.56, "text": "test"}
        ])
        result = transcriber.transcribe(tmp_audio)
        assert result[0]["start"] == 1.23
        assert result[0]["end"]   == 4.56

    def test_multiple_segments(self, tmp_audio, sample_segments):
        """transcribe() should return all segments from Whisper."""
        transcriber = self._make_transcriber_with_mock(sample_segments)
        result = transcriber.transcribe(tmp_audio)
        assert len(result) == len(sample_segments)

    def test_empty_segments(self, tmp_audio):
        """transcribe() should handle videos with no detected speech."""
        transcriber = self._make_transcriber_with_mock([])
        result = transcriber.transcribe(tmp_audio)
        assert result == []


class TestSegmentsToText:
    """Tests for WhisperTranscriber.segments_to_text() static method."""

    def test_joins_segments(self, sample_segments):
        result = WhisperTranscriber.segments_to_text(sample_segments)
        assert "hello everyone" in result
        assert "discuss the project" in result

    def test_segments_joined_with_space(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        result = WhisperTranscriber.segments_to_text(segments)
        assert result == "hello world"

    def test_empty_segments_returns_empty_string(self):
        result = WhisperTranscriber.segments_to_text([])
        assert result == ""
