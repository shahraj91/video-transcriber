# tests/test_llama.py
# Unit tests for pipeline/llama.py — LlamaClient class.
# All HTTP calls are mocked — no real Ollama server needed.

import pytest
import json
from unittest.mock import patch, MagicMock
from io import BytesIO
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from pipeline.llama import LlamaClient


class TestLlamaClientInit:
    """Tests for LlamaClient initialisation."""

    def test_default_model(self):
        client = LlamaClient()
        assert client.model == "llama3:8b"

    def test_custom_model(self):
        client = LlamaClient(model="mistral")
        assert client.model == "mistral"

    def test_default_api_url(self):
        client = LlamaClient()
        assert "11434" in client.api_url

    def test_custom_timeout(self):
        client = LlamaClient(timeout=60)
        assert client.timeout == 60


class TestLlamaClientAsk:
    """Tests for LlamaClient.ask()."""

    def _mock_response(self, response_text):
        """Helper: build a mock urllib response returning given text."""
        payload = json.dumps({"response": response_text}).encode()
        mock_resp = MagicMock()
        mock_resp.read.return_value = payload
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("pipeline.llama.urllib.request.urlopen")
    def test_ask_returns_response_text(self, mock_urlopen):
        """ask() should return the response field from Ollama's JSON."""
        mock_urlopen.return_value = self._mock_response("Hello from Llama.")
        client = LlamaClient()
        result = client.ask("Say hello", "Test")
        assert result == "Hello from Llama."

    @patch("pipeline.llama.urllib.request.urlopen")
    def test_ask_strips_whitespace(self, mock_urlopen):
        """ask() should strip leading/trailing whitespace from response."""
        mock_urlopen.return_value = self._mock_response("  hello  \n")
        client = LlamaClient()
        result = client.ask("prompt", "Test")
        assert result == "hello"

    @patch("pipeline.llama.urllib.request.urlopen")
    def test_ask_returns_none_on_empty_response(self, mock_urlopen):
        """ask() should return None if Ollama returns an empty response."""
        mock_urlopen.return_value = self._mock_response("")
        client = LlamaClient()
        result = client.ask("prompt", "Test")
        assert result is None

    @patch("pipeline.llama.urllib.request.urlopen")
    def test_ask_returns_none_on_connection_error(self, mock_urlopen):
        """ask() should return None gracefully if Ollama is unreachable."""
        mock_urlopen.side_effect = Exception("Connection refused")
        client = LlamaClient()
        result = client.ask("prompt", "Test")
        assert result is None

    @patch("pipeline.llama.urllib.request.urlopen")
    def test_ask_sends_correct_model(self, mock_urlopen):
        """ask() should include the configured model name in the request."""
        mock_urlopen.return_value = self._mock_response("response")
        client = LlamaClient(model="mistral")
        client.ask("prompt", "Test")
        call_args = mock_urlopen.call_args[0][0]
        payload = json.loads(call_args.data.decode())
        assert payload["model"] == "mistral"

    @patch("pipeline.llama.urllib.request.urlopen")
    def test_ask_sends_stream_false(self, mock_urlopen):
        """ask() should always send stream: false to get full response at once."""
        mock_urlopen.return_value = self._mock_response("response")
        client = LlamaClient()
        client.ask("prompt", "Test")
        call_args = mock_urlopen.call_args[0][0]
        payload = json.loads(call_args.data.decode())
        assert payload["stream"] is False

    @patch("pipeline.llama.urllib.request.urlopen")
    def test_ask_sends_prompt(self, mock_urlopen):
        """ask() should include the prompt in the request payload."""
        mock_urlopen.return_value = self._mock_response("response")
        client = LlamaClient()
        client.ask("my test prompt", "Test")
        call_args = mock_urlopen.call_args[0][0]
        payload = json.loads(call_args.data.decode())
        assert payload["prompt"] == "my test prompt"
