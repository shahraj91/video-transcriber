# ── pipeline/llama.py ────────────────────────────────────────────────────────
# Shared Llama client — all Ollama API communication goes through this class.
# Enhancers import this instead of making raw HTTP calls themselves.

import json
import urllib.request
from config import LLAMA_API_URL, LLAMA_MODEL, LLAMA_TIMEOUT


class LlamaClient:
    """
    Thin wrapper around the Ollama REST API.
    Centralises all HTTP communication with the local Llama model.
    """

    def __init__(self, model: str = LLAMA_MODEL, api_url: str = LLAMA_API_URL,
                 timeout: int = LLAMA_TIMEOUT):
        self.model   = model
        self.api_url = api_url
        self.timeout = timeout

    def ask(self, prompt: str, label: str = "Llama") -> str | None:
        """
        Send a prompt to Llama and return the response text.
        Returns None if Ollama is unreachable or returns an empty response.

        Args:
            prompt: The full prompt string to send.
            label:  Human-readable label shown in console output.
        """
        try:
            payload = json.dumps({
                "model":  self.model,
                "prompt": prompt,
                "stream": False,        # return full response at once
            }).encode()

            req = urllib.request.Request(
                self.api_url,
                data=payload,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data     = json.loads(resp.read())
                response = data.get("response", "").strip()
                if response:
                    print(f"  {label} complete.")
                    return response

        except Exception as e:
            print(f"  Llama unreachable for {label} ({e}). Skipping.")

        return None
