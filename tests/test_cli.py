# tests/test_cli.py
# CLI behaviour tests — verifies flags, help text, argument parsing,
# and that the script responds correctly to various command-line inputs.
# All tests use subprocess to invoke the script exactly as a user would.

import subprocess
import sys
import pytest
from pathlib import Path


# Helper: run transcribe.py with given args, return (stdout, stderr, returncode)
def run_cli(*args, cwd=None):
    result = subprocess.run(
        [sys.executable, "transcribe.py", *args],
        capture_output=True,
        text=True,
        cwd=cwd or Path(__file__).parent.parent
    )
    return result.stdout, result.stderr, result.returncode


# ── Help & Usage ──────────────────────────────────────────────────────────────

class TestHelpText:
    """--help should print useful, accurate information and exit cleanly."""

    def test_help_exits_with_zero(self):
        """--help should exit with code 0 (success)."""
        _, _, code = run_cli("--help")
        assert code == 0

    def test_help_shows_description(self):
        """--help should show the script description."""
        stdout, _, _ = run_cli("--help")
        assert "Transcribe" in stdout

    def test_help_shows_whisper_model_flag(self):
        """--help should document --whisper-model flag."""
        stdout, _, _ = run_cli("--help")
        assert "--whisper-model" in stdout

    def test_help_shows_llama_model_flag(self):
        """--help should document --llama-model flag."""
        stdout, _, _ = run_cli("--help")
        assert "--llama-model" in stdout

    def test_help_shows_language_flag(self):
        """--help should document --language flag."""
        stdout, _, _ = run_cli("--help")
        assert "--language" in stdout

    def test_help_shows_no_llama_flag(self):
        """--help should document --no-llama flag."""
        stdout, _, _ = run_cli("--help")
        assert "--no-llama" in stdout

    def test_help_shows_no_speakers_flag(self):
        stdout, _, _ = run_cli("--help")
        assert "--no-speakers" in stdout

    def test_help_shows_no_summary_flag(self):
        stdout, _, _ = run_cli("--help")
        assert "--no-summary" in stdout

    def test_help_shows_no_actions_flag(self):
        stdout, _, _ = run_cli("--help")
        assert "--no-actions" in stdout

    def test_help_shows_output_dir_flag(self):
        stdout, _, _ = run_cli("--help")
        assert "--output-dir" in stdout

    def test_help_shows_whisper_model_choices(self):
        """--help should list all valid Whisper model sizes."""
        stdout, _, _ = run_cli("--help")
        for model in ["tiny", "base", "small", "medium", "large"]:
            assert model in stdout


# ── Missing / Invalid Arguments ───────────────────────────────────────────────

class TestInvalidArguments:
    """Script should fail clearly with a non-zero exit code on bad input."""

    def test_no_args_exits_nonzero(self):
        """Running with no arguments should exit with a non-zero code."""
        _, _, code = run_cli()
        assert code != 0

    def test_no_args_shows_usage(self):
        """Running with no arguments should print usage hint."""
        _, stderr, _ = run_cli()
        assert "usage" in stderr.lower() or "error" in stderr.lower()

    def test_invalid_whisper_model_exits_nonzero(self):
        """Passing an invalid --whisper-model should fail."""
        _, _, code = run_cli("video.mp4", "--whisper-model", "superultra")
        assert code != 0

    def test_invalid_whisper_model_shows_error(self):
        """Passing an invalid --whisper-model should print an error."""
        _, stderr, _ = run_cli("video.mp4", "--whisper-model", "superultra")
        assert "invalid choice" in stderr.lower() or "error" in stderr.lower()

    def test_underscore_flag_fails(self):
        """Using --whisper_model (underscore) should fail — hyphens are required."""
        _, stderr, _ = run_cli("video.mp4", "--whisper_model", "medium")
        assert "error" in stderr.lower() or "unrecognized" in stderr.lower()

    def test_unknown_flag_exits_nonzero(self):
        """Passing an unknown flag should exit with non-zero."""
        _, _, code = run_cli("video.mp4", "--unknown-flag")
        assert code != 0


# ── Missing Video File ────────────────────────────────────────────────────────

class TestMissingVideoFile:
    """Script should handle a missing video file gracefully."""

    def test_missing_video_exits_nonzero(self):
        """Passing a non-existent video path should exit with code 1."""
        _, _, code = run_cli("/tmp/this_video_does_not_exist.mp4")
        assert code == 1

    def test_missing_video_prints_error_message(self):
        """Error message should mention the file was not found."""
        stdout, _, _ = run_cli("/tmp/this_video_does_not_exist.mp4")
        assert "not found" in stdout.lower() or "error" in stdout.lower()

    def test_missing_video_message_is_not_a_traceback(self):
        """Error output should not expose a raw Python traceback to the user."""
        stdout, stderr, _ = run_cli("/tmp/this_video_does_not_exist.mp4")
        combined = stdout + stderr
        assert "Traceback" not in combined
        assert "File \"" not in combined


# ── Flag Combinations ─────────────────────────────────────────────────────────

class TestFlagCombinations:
    """Valid flag combinations should be parsed without error."""

    def test_no_llama_flag_parsed(self, tmp_path):
        """--no-llama flag should be accepted (video missing — exit 1 not arg error)."""
        _, _, code = run_cli(str(tmp_path / "fake.mp4"), "--no-llama")
        # Exit 1 = video not found (expected), not argparse error (exit 2)
        assert code == 1

    def test_no_summary_and_no_actions_combined(self, tmp_path):
        """--no-summary and --no-actions together should be accepted."""
        _, _, code = run_cli(str(tmp_path / "fake.mp4"), "--no-summary", "--no-actions")
        assert code == 1

    def test_language_flag_accepted(self, tmp_path):
        """--language flag should be accepted with any string value."""
        _, _, code = run_cli(str(tmp_path / "fake.mp4"), "--language", "Hindi")
        assert code == 1

    def test_output_dir_flag_accepted(self, tmp_path):
        """--output-dir flag should be accepted."""
        _, _, code = run_cli(str(tmp_path / "fake.mp4"), "--output-dir", str(tmp_path))
        assert code == 1

    def test_short_output_dir_flag(self, tmp_path):
        """-o shorthand for --output-dir should be accepted."""
        _, _, code = run_cli(str(tmp_path / "fake.mp4"), "-o", str(tmp_path))
        assert code == 1
