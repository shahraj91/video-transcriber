# ── config.py ────────────────────────────────────────────────────────────────
# Central configuration for the Video Transcription Pipeline.
# Change model names and API settings here — no need to touch other files.

LLAMA_API_URL = "http://localhost:11434/api/generate"   # Ollama default
LLAMA_MODEL   = "llama3:8b"                              # change to your Ollama model
WHISPER_MODEL = "base"                                   # tiny | base | small | medium | large
LLAMA_TIMEOUT = 180                                      # seconds to wait for Llama response
