# tests/test_enhancers.py
# Unit tests for pipeline/enhancers.py — all 5 enhancer classes.
# LlamaClient is mocked in all tests — no real Ollama calls made.

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.enhancers import (
    TranscriptRefiner,
    SpeakerDetector,
    TranscriptSummariser,
    ActionItemExtractor,
    TranscriptTranslator,
)


# ── TranscriptRefiner ─────────────────────────────────────────────────────────

class TestTranscriptRefiner:

    def test_returns_llama_response(self, mock_llama_client, sample_raw_text):
        """run() should return Llama's cleaned response."""
        mock_llama_client.ask.return_value = "Cleaned text."
        result = TranscriptRefiner(mock_llama_client).run(sample_raw_text)
        assert result == "Cleaned text."

    def test_falls_back_to_raw_text_if_llama_fails(self, mock_llama_client, sample_raw_text):
        """run() should return raw_text if Llama returns None."""
        mock_llama_client.ask.return_value = None
        result = TranscriptRefiner(mock_llama_client).run(sample_raw_text)
        assert result == sample_raw_text

    def test_prompt_contains_transcript(self, mock_llama_client, sample_raw_text):
        """The prompt sent to Llama should include the raw transcript."""
        mock_llama_client.ask.return_value = "refined"
        TranscriptRefiner(mock_llama_client).run(sample_raw_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert sample_raw_text in prompt

    def test_prompt_instructs_no_commentary(self, mock_llama_client, sample_raw_text):
        """Prompt must tell Llama to return ONLY cleaned text."""
        mock_llama_client.ask.return_value = "refined"
        TranscriptRefiner(mock_llama_client).run(sample_raw_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert "no commentary" in prompt.lower()

    def test_calls_llama_once(self, mock_llama_client, sample_raw_text):
        """run() should make exactly one Llama call."""
        TranscriptRefiner(mock_llama_client).run(sample_raw_text)
        mock_llama_client.ask.assert_called_once()


# ── SpeakerDetector ───────────────────────────────────────────────────────────

class TestSpeakerDetector:

    def test_returns_llama_response(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "Speaker 1: Hello.\nSpeaker 2: Hi."
        result = SpeakerDetector(mock_llama_client).run(sample_refined_text)
        assert "Speaker 1" in result

    def test_falls_back_to_refined_text_if_llama_fails(self, mock_llama_client, sample_refined_text):
        """run() should return refined_text unchanged if Llama returns None."""
        mock_llama_client.ask.return_value = None
        result = SpeakerDetector(mock_llama_client).run(sample_refined_text)
        assert result == sample_refined_text

    def test_prompt_mentions_speaker_labels(self, mock_llama_client, sample_refined_text):
        """Prompt should instruct Llama to use Speaker 1, Speaker 2 labels."""
        mock_llama_client.ask.return_value = "Speaker 1: text"
        SpeakerDetector(mock_llama_client).run(sample_refined_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert "Speaker 1" in prompt

    def test_prompt_contains_transcript(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "Speaker 1: text"
        SpeakerDetector(mock_llama_client).run(sample_refined_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert sample_refined_text in prompt


# ── TranscriptSummariser ──────────────────────────────────────────────────────

class TestTranscriptSummariser:

    def test_returns_llama_response(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "This is a summary."
        result = TranscriptSummariser(mock_llama_client).run(sample_refined_text)
        assert result == "This is a summary."

    def test_returns_fallback_message_if_llama_fails(self, mock_llama_client, sample_refined_text):
        """run() should return a fallback message string, not None or raw text."""
        mock_llama_client.ask.return_value = None
        result = TranscriptSummariser(mock_llama_client).run(sample_refined_text)
        assert "could not be generated" in result.lower()

    def test_prompt_requests_3_to_5_sentences(self, mock_llama_client, sample_refined_text):
        """Prompt should specify 3-5 sentences for consistent output length."""
        mock_llama_client.ask.return_value = "summary"
        TranscriptSummariser(mock_llama_client).run(sample_refined_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert "3" in prompt and "5" in prompt

    def test_prompt_contains_transcript(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "summary"
        TranscriptSummariser(mock_llama_client).run(sample_refined_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert sample_refined_text in prompt


# ── ActionItemExtractor ───────────────────────────────────────────────────────

class TestActionItemExtractor:

    def test_returns_llama_response(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "- Finish report by Friday\n- John to handle presentation"
        result = ActionItemExtractor(mock_llama_client).run(sample_refined_text)
        assert "Finish report" in result

    def test_returns_fallback_message_if_llama_fails(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = None
        result = ActionItemExtractor(mock_llama_client).run(sample_refined_text)
        assert "could not be extracted" in result.lower()

    def test_prompt_requests_bullet_format(self, mock_llama_client, sample_refined_text):
        """Prompt should instruct Llama to use bullet points starting with '- '."""
        mock_llama_client.ask.return_value = "- item"
        ActionItemExtractor(mock_llama_client).run(sample_refined_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert "- " in prompt

    def test_prompt_handles_no_action_items(self, mock_llama_client, sample_refined_text):
        """Prompt should tell Llama what to write if no action items are found."""
        mock_llama_client.ask.return_value = "No action items identified."
        ActionItemExtractor(mock_llama_client).run(sample_refined_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert "no action items" in prompt.lower()

    def test_prompt_contains_transcript(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "- item"
        ActionItemExtractor(mock_llama_client).run(sample_refined_text)
        prompt = mock_llama_client.ask.call_args[0][0]
        assert sample_refined_text in prompt


# ── TranscriptTranslator ──────────────────────────────────────────────────────

class TestTranscriptTranslator:

    def test_returns_llama_response(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "नमस्ते सबको।"
        result = TranscriptTranslator(mock_llama_client).run(sample_refined_text, "Hindi")
        assert result == "नमस्ते सबको।"

    def test_returns_fallback_if_llama_fails(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = None
        result = TranscriptTranslator(mock_llama_client).run(sample_refined_text, "Hindi")
        assert "could not be generated" in result.lower()

    def test_prompt_includes_language(self, mock_llama_client, sample_refined_text):
        """Prompt should specify the target language."""
        mock_llama_client.ask.return_value = "translated"
        TranscriptTranslator(mock_llama_client).run(sample_refined_text, "Spanish")
        prompt = mock_llama_client.ask.call_args[0][0]
        assert "Spanish" in prompt

    def test_different_languages(self, mock_llama_client, sample_refined_text):
        """run() should work with any language string."""
        for lang in ["Hindi", "French", "Japanese", "Arabic"]:
            mock_llama_client.ask.return_value = f"translated to {lang}"
            result = TranscriptTranslator(mock_llama_client).run(sample_refined_text, lang)
            assert result == f"translated to {lang}"

    def test_prompt_contains_transcript(self, mock_llama_client, sample_refined_text):
        mock_llama_client.ask.return_value = "translated"
        TranscriptTranslator(mock_llama_client).run(sample_refined_text, "Hindi")
        prompt = mock_llama_client.ask.call_args[0][0]
        assert sample_refined_text in prompt
