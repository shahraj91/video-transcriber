# tests/conftest.py
# Shared pytest fixtures available to all test files automatically.

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ── Sample data fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_segments():
    """Realistic Whisper output segments."""
    return [
        {"start": 0.0,  "end": 3.5,  "text": "um hello everyone welcome to the meeting"},
        {"start": 3.5,  "end": 7.2,  "text": "uh today we are going to discuss the project"},
        {"start": 7.2,  "end": 11.0, "text": "like we need to finish the report by friday"},
        {"start": 11.0, "end": 15.0, "text": "and john will handle the client presentation"},
    ]

@pytest.fixture
def sample_raw_text(sample_segments):
    """Raw text joined from segments — mimics WhisperTranscriber.segments_to_text()."""
    return " ".join(seg["text"] for seg in sample_segments)

@pytest.fixture
def sample_refined_text():
    """Cleaned transcript after Llama refinement."""
    return (
        "Hello everyone, welcome to the meeting. "
        "Today we are going to discuss the project. "
        "We need to finish the report by Friday. "
        "John will handle the client presentation."
    )

@pytest.fixture
def mock_llama_client():
    """A mock LlamaClient that returns predictable responses."""
    client = MagicMock()
    client.ask.return_value = "Mocked Llama response."
    return client

@pytest.fixture
def tmp_video(tmp_path):
    """A fake video file (empty) for path-based tests."""
    video = tmp_path / "test_video.mp4"
    video.write_bytes(b"fake video content")
    return video

@pytest.fixture
def tmp_audio(tmp_path):
    """A fake WAV audio file for path-based tests."""
    audio = tmp_path / "test_audio.wav"
    audio.write_bytes(b"fake audio content")
    return audio
