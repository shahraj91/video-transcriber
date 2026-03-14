# CLAUDE.md — VoiceToText Project Reference

## Project Purpose

Fully local, offline video transcription pipeline. Extracts audio from video, transcribes with OpenAI Whisper, and enhances with a local Llama LLM via Ollama. No cloud APIs, no internet required, no data leaves the machine.

---

## How to Run

### Web UI (recommended for interactive use)
```bash
streamlit run ui.py
# Opens at http://localhost:8501
```

### CLI (for scripting/automation)
```bash
# Basic usage
python3 transcribe.py video.mp4

# With options
python3 transcribe.py video.mp4 --whisper-model small --language Hindi
python3 transcribe.py video.mp4 --no-llama               # skip all Llama features
python3 transcribe.py video.mp4 --no-speakers --no-summary --no-actions
python3 transcribe.py video.mp4 --output-dir ~/transcripts
```

**CLI flags:**
- `--whisper-model` — `tiny` | `base` (default) | `small` | `medium` | `large`
- `--llama-model` — Ollama model name (default: `llama3:8b`)
- `--language` — translate output to this language
- `--no-llama` — disable all Llama/Ollama features
- `--no-speakers` — skip speaker detection
- `--no-summary` — skip summarization
- `--no-actions` — skip action item extraction
- `--output-dir` — custom base directory for output folders

**Prerequisites:**
- ffmpeg installed on system
- `pip install -r requirements.txt` (openai-whisper, ffmpeg-python, streamlit, pytest)
- Ollama running locally at `http://localhost:11434` (optional — degrades gracefully if absent)
- First-run Streamlit fix: create `~/.streamlit/credentials.toml` with `[general]\nemail = ""`

---

## Architecture

Modular pipeline following the **Page Object Model (POM)** pattern. Each stage is a separate, independently testable class.

```
Video File
    ↓
[1] AudioExtractor       pipeline/audio.py       ffmpeg → mono 16kHz WAV
    ↓
[2] WhisperTranscriber   pipeline/transcriber.py  local Whisper → [{start, end, text}, ...]
    ↓
[3] LlamaClient          pipeline/llama.py        shared Ollama HTTP client
    ↓
[4] Enhancers            pipeline/enhancers.py
    ├─ TranscriptRefiner       → cleaned text (always runs unless --no-llama)
    ├─ SpeakerDetector         → speaker labels (--no-speakers to skip)
    ├─ TranscriptSummariser    → 3-5 sentence summary (--no-summary to skip)
    ├─ ActionItemExtractor     → bullet-point action items (--no-actions to skip)
    └─ TranscriptTranslator    → translation (only with --language)
    ↓
[5] OutputManager        pipeline/output.py       saves files to {stem}_{YYYYMMDD_HHMMSS}/
```

**Key design patterns:**
- **Dependency injection** — single `LlamaClient` instance passed to all enhancers
- **Graceful degradation** — if Ollama unreachable, enhancers return raw Whisper output
- **Lazy model loading** — Whisper model loaded only on first `transcribe()` call
- **Feature toggles** — every non-core feature disabled via CLI flag
- **Timestamped output folders** — each run gets a unique folder, never overwrites

---

## File Structure

```
VoiceToText/
├── transcribe.py          # CLI entry point — orchestrates the full pipeline
├── ui.py                  # Streamlit web UI (upload, settings, download)
├── config.py              # Centralized settings: Ollama URL, model names, timeouts
├── requirements.txt       # openai-whisper, ffmpeg-python, streamlit, pytest
├── pytest.ini             # Pytest config + custom markers (real, llama)
├── .gitignore
├── README.md
├── CLAUDE.md              # This file
├── pipeline/
│   ├── __init__.py
│   ├── audio.py           # AudioExtractor
│   ├── transcriber.py     # WhisperTranscriber
│   ├── llama.py           # LlamaClient
│   ├── enhancers.py       # TranscriptRefiner, SpeakerDetector, TranscriptSummariser,
│   │                      #   ActionItemExtractor, TranscriptTranslator
│   └── output.py          # OutputManager
└── tests/
    ├── conftest.py         # Shared fixtures (mocks + real file helpers)
    ├── test_audio.py       # 10 unit tests — AudioExtractor
    ├── test_transcriber.py # 14 unit tests — WhisperTranscriber
    ├── test_llama.py       # 11 unit tests — LlamaClient
    ├── test_enhancers.py   # 20 unit tests — all 5 enhancers
    ├── test_output.py      # 21 unit tests — OutputManager
    ├── test_integration.py # 7 integration/E2E tests
    ├── test_cli.py         # 21 CLI tests (subprocess)
    ├── test_ux.py          # 22 UX tests (subprocess)
    ├── test_real.py        # 29 real file tests (requires assets + Whisper)
    └── assets/
        ├── generate_assets.py   # Creates synthetic test videos (run this to regenerate)
        ├── english_clear.mp4
        ├── silence.mp4
        ├── background_noise.mp4
        ├── short_clip.mp4
        └── multi_tone.mp4
```

---

## Pipeline Modules

### `transcribe.py` — CLI Orchestrator
Parses args, validates input, instantiates all pipeline classes, runs them in sequence, cleans up temp audio file, prints output folder path.

### `ui.py` — Streamlit Web UI
Sidebar: Whisper model selector, Ollama model input, Llama feature toggles (disabled when Llama off), translation language input.
Main: video file uploader → Transcribe button → live `st.status` progress → tabbed results view with per-file download buttons and a "Download all as ZIP" button.
Uses `st.session_state` to persist results across reruns. Saves uploaded file to a named temp file so `OutputManager` gets a proper stem.

### `config.py` — Settings
```python
LLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL   = "llama3:8b"
WHISPER_MODEL = "base"
LLAMA_TIMEOUT = 180
```

### `pipeline/audio.py` — AudioExtractor
- Runs ffmpeg to extract mono 16kHz PCM WAV from video
- Exits program on ffmpeg failure (critical dependency)

### `pipeline/transcriber.py` — WhisperTranscriber
- Lazy-loads Whisper model on first call
- Returns `[{start: float, end: float, text: str}, ...]`
- Static method `segments_to_text(segments)` joins segments into a single string
- Exits if openai-whisper not installed

### `pipeline/llama.py` — LlamaClient
- Single shared HTTP client using `urllib.request` (stdlib only)
- `ask(prompt, label) -> str | None` — returns response or `None` on any error
- All exceptions caught silently; callers check for `None`

### `pipeline/enhancers.py` — 5 Enhancer Classes
Each takes a `LlamaClient` at init. Each has `run(text) -> str`. Falls back to input text if Llama returns `None`.

| Class | Purpose | Fallback |
|---|---|---|
| `TranscriptRefiner` | Fix punctuation, remove fillers | raw text |
| `SpeakerDetector` | Label "Speaker 1:", "Speaker 2:" | raw text |
| `TranscriptSummariser` | 3-5 sentence summary | raw text |
| `ActionItemExtractor` | Bullet-point tasks/decisions | raw text |
| `TranscriptTranslator` | Translate to target language | raw text |

### `pipeline/output.py` — OutputManager
- Creates `{video_stem}_{YYYYMMDD_HHMMSS}/` folder on init
- `save_text(content, suffix, label)` — writes plain text file
- `save_srt(segments)` — generates SRT with `HH:MM:SS,mmm` timestamps
- `path(suffix)` — returns full path for a given file suffix

---

## Output Files

Each run produces a folder `{video_name}_{YYYYMMDD_HHMMSS}/` containing:

| File | Always? | Contents |
|---|---|---|
| `_transcript.txt` | Yes | Refined transcript |
| `_transcript.srt` | Yes | Subtitle file with timestamps |
| `_speakers.txt` | Optional | Speaker-labeled transcript |
| `_summary.txt` | Optional | 3-5 sentence summary |
| `_actions.txt` | Optional | Action items / decisions |
| `_translation_{Lang}.txt` | Optional | Translated transcript |

---

## Running Tests

```bash
# Fast tests only (141 tests, no external dependencies)
pytest

# All tests including real file tests (requires assets + Whisper installed)
pytest -m real

# Skip real file tests explicitly
pytest -m "not real"

# Real file tests but skip Ollama-dependent ones
pytest -m "real and not llama"

# Specific test file
pytest tests/test_enhancers.py -v
```

**Test markers** (defined in `pytest.ini`):
- `real` — requires synthetic video assets and Whisper installed
- `llama` — requires Ollama running locally

**Generating test assets** (needed for `real` tests):
```bash
python3 tests/assets/generate_assets.py
```

---

## Dependencies

| Tool | Role | Required? |
|---|---|---|
| ffmpeg (binary) | Audio extraction | Yes — hard fail without it |
| openai-whisper | Speech-to-text | Yes — hard fail without it |
| Ollama + llama3:8b | LLM enhancement | No — graceful fallback |
| ffmpeg-python | Python ffmpeg wrapper | Yes (pip) |
| streamlit | Web UI server | Yes for UI, not needed for CLI |
| pytest | Testing | Dev only |