# Video Transcriber

> 🔒 **Fully local & offline** — no data leaves your machine. No cloud APIs, no internet required.

Extracts audio from a video, transcribes it with **Whisper** (runs locally on your hardware), refines the text with your **local Llama model** via Ollama, and saves both a `.txt` and `.srt` file — entirely on your own machine.

---

## Pipeline

```
video → ffmpeg (audio) → Whisper (STT) → Llama (refine) → .txt + .srt + extras
```

---

## Usage

Two ways to run:

| Mode | Command | Best for |
|---|---|---|
| **Web UI** | `streamlit run ui.py` | Interactive use, easy downloads |
| **CLI** | `python3 transcribe.py video.mp4` | Scripting, automation, CI |

---

## Project Structure

```
VoiceToText/
├── transcribe.py              # CLI entry point — orchestrates the pipeline
├── ui.py                      # Streamlit web UI
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
    ├── conftest.py            # Shared fixtures (mock + real file)
    ├── test_audio.py          # AudioExtractor — 10 tests
    ├── test_transcriber.py    # WhisperTranscriber — 14 tests
    ├── test_llama.py          # LlamaClient — 11 tests
    ├── test_enhancers.py      # All 5 enhancer classes — 20 tests
    ├── test_output.py         # OutputManager — 21 tests
    ├── test_integration.py    # Integration + end-to-end — 7 tests
    ├── test_cli.py            # CLI flags, args, error handling — 21 tests
    ├── test_ux.py             # Progress feedback, errors, output quality, SRT — 22 tests
    ├── test_real.py           # Real file tests — 29 tests (pytest -m real)
    └── assets/
        ├── generate_assets.py # Generates synthetic test video files
        ├── english_clear.mp4  # 8s speech-rhythm audio
        ├── silence.mp4        # 5s silence — edge case
        ├── background_noise.mp4 # Speech + white noise
        ├── short_clip.mp4     # 2s clip — minimal audio
        └── multi_tone.mp4     # Alternating tones — speaker change simulation
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
pip install -r requirements.txt
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

## Web UI

```bash
source venv/bin/activate
streamlit run ui.py
# Opens at http://localhost:8501
```

The UI lets you:
- Upload any video file (mp4, mov, avi, mkv, webm, m4v, flv)
- Select Whisper model size and Ollama model
- Toggle individual Llama features on/off
- Enter a translation language
- View all outputs in tabs and download them individually or as a ZIP

---

## CLI Usage

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

The test suite has two layers — fast mock tests and slower real file tests.

### Mock tests (167) — run on any machine, no dependencies needed
```bash
source venv/bin/activate
pytest
```

### Real file tests (29) — require Whisper and generated assets
```bash
# Step 1: Generate test assets (run once)
python tests/assets/generate_assets.py

# Step 2: Run real tests
pytest -m real -v

# Skip Ollama-dependent tests (run without Ollama running)
pytest -m "real and not llama" -v
```

### Test your own video
Drop any `.mp4` into `tests/assets/your_video.mp4` — it will be picked up automatically by `test_your_own_video` when running `pytest -m real`.

### Run specific files
```bash
pytest tests/test_cli.py        # CLI behaviour
pytest tests/test_ux.py         # User experience
pytest tests/test_real.py       # Real file tests (assets required)
```

### Full test coverage

| File | What it tests | Tests | Requires |
|---|---|---|---|
| `test_audio.py` | AudioExtractor — ffmpeg flags, failure handling | 10 | Mocked |
| `test_transcriber.py` | WhisperTranscriber — lazy loading, segments | 14 | Mocked |
| `test_llama.py` | LlamaClient — HTTP calls, fallback, payload | 11 | Mocked |
| `test_enhancers.py` | All 5 enhancers — prompts, fallbacks | 20 | Mocked |
| `test_output.py` | OutputManager — folders, files, SRT format | 21 | Mocked |
| `test_integration.py` | Classes working together, E2E | 7 | Mocked |
| `test_cli.py` | CLI flags, --help, invalid args | 21 | Mocked |
| `test_ux.py` | Progress, output quality, SRT validity | 22 | Mocked |
| `test_real.py` | Real ffmpeg, Whisper, pipeline, edge cases | 29 | Real assets |
| `test_ui.py` | run_pipeline() — steps, results, delegation | 26 | Mocked |
| **Total** | | **196** | |

### Test types
- **Unit** — each class tested in isolation with mocked dependencies
- **Integration** — classes tested working together
- **End-to-end** — full pipeline run with all external calls mocked
- **CLI** — script invoked via subprocess as a real user would
- **UX** — output quality, progress messages, error readability, SRT validity
- **Real** — actual ffmpeg and Whisper calls with generated video assets

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
| Real tests skipped | Run `python tests/assets/generate_assets.py` first |
| Real tests all skip | Check `ASSETS_DIR` resolves correctly — needs `Path(__file__).resolve()` |
| Streamlit email prompt on first run | Create `~/.streamlit/credentials.toml` with `[general]\nemail = ""` |
| Streamlit port in use | `streamlit run ui.py --server.port 8502` |
