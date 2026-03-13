# Video Transcriber

> 🔒 **Fully local & offline** — no data leaves your machine. No cloud APIs, no internet required.

Extracts audio from a video, transcribes it with **Whisper** (runs locally on your hardware), refines the text with your **local Llama model** via Ollama, and saves both a `.txt` and `.srt` file — entirely on your own machine.

---

## Pipeline

```
video → ffmpeg (audio) → Whisper (STT) → Llama (refine) → .txt + .srt + extras
```

---

## Project Structure

```
VoiceToText/
├── transcribe.py              # Entry point — orchestrates the pipeline
├── config.py                  # All settings in one place (models, API URL)
├── requirements.txt           # Python dependencies
├── pytest.ini                 # Test configuration
├── README.md
├── pipeline/
│   ├── __init__.py            # Makes pipeline a Python package
│   ├── audio.py               # AudioExtractor — ffmpeg audio extraction
│   ├── transcriber.py         # WhisperTranscriber — speech-to-text
│   ├── llama.py               # LlamaClient — shared Ollama API client
│   ├── enhancers.py           # Llama features: refine, speakers, summary, actions, translate
│   └── output.py              # OutputManager — timestamped folder + file saving
└── tests/
    ├── __init__.py
    ├── conftest.py            # Shared fixtures
    ├── test_audio.py          # AudioExtractor — 7 tests
    ├── test_transcriber.py    # WhisperTranscriber — 11 tests
    ├── test_llama.py          # LlamaClient — 7 tests
    ├── test_enhancers.py      # All 5 enhancer classes — 20 tests
    ├── test_output.py         # OutputManager — 14 tests
    ├── test_integration.py    # Integration + end-to-end — 5 tests
    ├── test_cli.py            # CLI flags, args, error handling — 21 tests
    └── test_ux.py             # Progress feedback, errors, output quality, SRT — 22 tests
```

---

## Prerequisites

### 1. ffmpeg
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### 2. Python virtual environment + dependencies
```bash
cd ~/Documents/VoiceToText
python3 -m venv venv
source venv/bin/activate
pip install openai-whisper ffmpeg-python pytest
```

### 3. Ollama + Llama model
```bash
# Check available models (if running in Docker):
sudo docker exec -it <container_id> ollama list

# Verify API is reachable from host:
curl http://localhost:11434/api/tags
```

---

## Configuration

Open `config.py` and update to match your setup:

```python
LLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL   = "llama3:8b"    # your Ollama model name
WHISPER_MODEL = "base"         # tiny | base | small | medium | large
LLAMA_TIMEOUT = 180            # seconds to wait for Llama response
```

| Whisper model | Speed | Accuracy | VRAM needed |
|---|---|---|---|
| tiny | fastest | ⭐⭐ | ~1 GB |
| base | fast | ⭐⭐⭐ | ~1 GB |
| small | moderate | ⭐⭐⭐ | ~2 GB |
| medium | slow | ⭐⭐⭐⭐ | ~5 GB |
| large | slowest | ⭐⭐⭐⭐⭐ | ~10 GB |

---

## Usage

```bash
source venv/bin/activate

# Basic — all features enabled
python3 transcribe.py path/to/video.mp4

# Use a larger Whisper model
python3 transcribe.py video.mp4 --whisper-model medium

# Translate transcript to another language
python3 transcribe.py video.mp4 --language Hindi

# Skip specific features
python3 transcribe.py video.mp4 --no-speakers
python3 transcribe.py video.mp4 --no-summary --no-actions

# Skip all Llama features (raw Whisper output only)
python3 transcribe.py video.mp4 --no-llama

# Save to a custom base directory
python3 transcribe.py video.mp4 --output-dir ~/transcripts
```

---

## Output

Each run creates a **timestamped folder** next to the video (or in `--output-dir`):

```
VID_20230624_231126_20260313_142305/
├── VID_20230624_231126_transcript.txt
├── VID_20230624_231126_transcript.srt
├── VID_20230624_231126_speakers.txt
├── VID_20230624_231126_summary.txt
├── VID_20230624_231126_actions.txt
└── VID_20230624_231126_translation_Hindi.txt   (only if --language passed)
```

---

## Testing

All tests use mocks — no real Whisper model, ffmpeg, or Ollama calls are made. Tests run fast and work on any machine.

### Run all tests
```bash
source venv/bin/activate
pytest
```

### Run a specific file
```bash
pytest tests/test_cli.py        # CLI behaviour
pytest tests/test_ux.py         # User experience
pytest tests/test_integration.py
```

### Run with verbose output
```bash
pytest -v
```

### Test coverage summary

| File | What it tests | Tests |
|---|---|---|
| `test_audio.py` | AudioExtractor — ffmpeg flags, failure handling | 7 |
| `test_transcriber.py` | WhisperTranscriber — lazy loading, segments, text joining | 11 |
| `test_llama.py` | LlamaClient — HTTP calls, fallback, payload format | 7 |
| `test_enhancers.py` | All 5 enhancers — prompts, fallbacks, responses | 20 |
| `test_output.py` | OutputManager — folders, file saving, SRT time format | 14 |
| `test_integration.py` | Full pipeline — classes working together, E2E | 5 |
| `test_cli.py` | CLI flags, --help, invalid args, error messages | 21 |
| `test_ux.py` | Progress feedback, output quality, SRT validity, folder UX | 22 |
| **Total** | | **141** |

### Test types
- **Unit** — each class tested in isolation with mocked dependencies
- **Integration** — classes tested working together
- **End-to-end** — full pipeline run with all external calls mocked
- **CLI** — script invoked via subprocess as a real user would
- **UX** — output quality, progress messages, error readability, SRT format validity

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ffmpeg not found` | `sudo apt install ffmpeg` |
| `whisper not found` | Activate venv then `pip install openai-whisper` |
| `externally-managed-environment` | Use venv — see Prerequisites |
| Llama refinement skipped | Run `curl http://localhost:11434/api/tags` to verify Ollama is up |
| Wrong model name | `sudo docker exec -it <id> ollama list` |
| CUDA out of memory | `CUDA_VISIBLE_DEVICES="" python3 transcribe.py ...` to force CPU |
| Flag not recognized | Use hyphens not underscores: `--whisper-model` not `--whisper_model` |
