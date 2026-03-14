# tests/conftest.py
# Shared pytest fixtures available to all test files automatically.
# Includes both mock fixtures (for unit/integration tests)
# and real file fixtures (for pytest -m real tests).

import pytest
from pathlib import Path
from unittest.mock import MagicMock


# ── Mock fixtures ─────────────────────────────────────────────────────────────

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


# ── Real file fixtures ────────────────────────────────────────────────────────
# Used by tests/test_real.py — requires assets to be generated first.
# Run: python tests/assets/generate_assets.py

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def require_asset(filename: str) -> Path:
    """Return path to asset, skipping the test if file doesn't exist."""
    path = ASSETS_DIR / filename
    if not path.exists():
        pytest.skip(
            f"Asset '{filename}' not found. "
            f"Run: python tests/assets/generate_assets.py"
        )
    return path


@pytest.fixture
def english_clear_video() -> Path:
    """8-second speech-rhythm audio — primary real transcription test."""
    return require_asset("english_clear.mp4")


@pytest.fixture
def silence_video() -> Path:
    """5-second silent video — edge case, no speech."""
    return require_asset("silence.mp4")


@pytest.fixture
def background_noise_video() -> Path:
    """8-second speech + white noise — noisy audio test."""
    return require_asset("background_noise.mp4")


@pytest.fixture
def short_clip_video() -> Path:
    """2-second clip — minimal audio edge case."""
    return require_asset("short_clip.mp4")


@pytest.fixture
def multi_tone_video() -> Path:
    """Alternating low/high tones — simulates speaker changes."""
    return require_asset("multi_tone.mp4")


@pytest.fixture
def real_output_dir(tmp_path) -> Path:
    """Temporary output directory for real pipeline runs."""
    out = tmp_path / "real_outputs"
    out.mkdir()
    return out
