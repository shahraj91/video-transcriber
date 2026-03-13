# Video Transcription Pipeline

Extracts audio from a video, transcribes it with **Whisper** (local/offline), refines the text with your **local Llama model** via Ollama, and saves both a `.txt` and `.srt` file.

---

## Pipeline

```
video → ffmpeg (audio) → Whisper (STT) → Llama (refine) → .txt + .srt
```

---

## Prerequisites

### 1. ffmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows – download from https://ffmpeg.org/download.html
```

### 2. Python dependencies
```bash
pip install openai-whisper
```
> `openai-whisper` will also install `torch` automatically.

### 3. Ollama + your Llama model
Make sure Ollama is running and your model is pulled:
```bash
ollama serve           # starts the local API on http://localhost:11434
ollama pull llama3     # or whichever model you use
```

---

## Configuration

Open `transcribe.py` and update these two lines near the top to match your setup:

```python
LLAMA_MODEL   = "llama3:8b"     # your Ollama model name  (ollama list)
WHISPER_MODEL = "base"       # tiny | base | small | medium | large
```

| Whisper model | Speed  | Accuracy |
|---------------|--------|----------|
| tiny          | fastest | lowest  |
| base          | fast    | good    |
| small         | medium  | better  |
| medium        | slow    | great   |
| large         | slowest | best    |

---

## Usage

```bash
# Basic – output files saved next to the video
python transcribe.py path/to/video.mp4

# Custom output directory
python transcribe.py path/to/video.mp4 --output-dir ./transcripts

# Skip Llama refinement (use raw Whisper text)
python transcribe.py path/to/video.mp4 --no-llama

# Use a larger Whisper model
python transcribe.py path/to/video.mp4 --whisper-model small

# Override Llama model at runtime
python transcribe.py path/to/video.mp4 --llama-model mistral
```

---

## Output

For a video named `lecture.mp4`, two files are created:

- `lecture_transcript.txt` — full cleaned transcript (Llama-refined)
- `lecture_transcript.srt` — subtitle file with timestamps from Whisper

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `ffmpeg not found` | Install ffmpeg and ensure it's on your PATH |
| `whisper not found` | `pip install openai-whisper` |
| Llama refinement skipped | Make sure `ollama serve` is running; or use `--no-llama` |
| Wrong model name | Run `ollama list` to see available models |
