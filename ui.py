"""
ui.py — Streamlit web UI for the VoiceToText pipeline.

Run with:
    streamlit run ui.py
"""

import io
import tempfile
import zipfile
from pathlib import Path

import streamlit as st

from config import LLAMA_MODEL, WHISPER_MODEL
from pipeline.audio import AudioExtractor
from pipeline.transcriber import WhisperTranscriber
from pipeline.llama import LlamaClient
from pipeline.enhancers import (
    TranscriptRefiner,
    SpeakerDetector,
    TranscriptSummariser,
    ActionItemExtractor,
    TranscriptTranslator,
)
from pipeline.output import OutputManager


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="VoiceToText",
    page_icon="🎙️",
    layout="wide",
)

st.title("🎙️ VoiceToText")
st.caption("Local, private video transcription powered by Whisper + Llama.")


# ── Sidebar: settings ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Settings")

    whisper_model = st.selectbox(
        "Whisper model",
        options=["tiny", "base", "small", "medium", "large"],
        index=["tiny", "base", "small", "medium", "large"].index(WHISPER_MODEL),
        help="Larger models are slower but more accurate.",
    )

    llama_model = st.text_input(
        "Ollama model",
        value=LLAMA_MODEL,
        help="Must match a model installed in Ollama (e.g. llama3:8b, mistral).",
    )

    st.divider()
    st.subheader("Llama features")

    use_llama = st.toggle("Enable Llama enhancements", value=True)

    with st.container():
        use_speakers = st.checkbox("Speaker detection", value=True, disabled=not use_llama)
        use_summary  = st.checkbox("Summary",           value=True, disabled=not use_llama)
        use_actions  = st.checkbox("Action items",      value=True, disabled=not use_llama)

    st.divider()
    st.subheader("Translation")

    language = st.text_input(
        "Translate to language",
        placeholder="e.g. Hindi, Spanish, French",
        help="Leave blank to skip translation.",
    ).strip()


# ── Main: upload + run ────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload a video file",
    type=["mp4", "mov", "avi", "mkv", "webm", "m4v", "flv"],
    help="Video is processed locally — nothing is sent to the cloud.",
)

run_button = st.button(
    "▶ Transcribe",
    type="primary",
    disabled=uploaded_file is None,
    use_container_width=True,
)


# ── Pipeline execution ────────────────────────────────────────────────────────

def run_pipeline(video_path: Path, settings: dict) -> dict:
    """Run the full pipeline and return a dict of {label: (path, content)}."""
    results = {}

    output = OutputManager(video_path)
    audio_path = output.path("_audio.wav")

    # Step 1: Extract audio
    yield "step", "Extracting audio with ffmpeg..."
    AudioExtractor().extract(video_path, audio_path)

    # Step 2: Transcribe
    yield "step", f"Transcribing with Whisper ({settings['whisper_model']})... (this may take a while)"
    transcriber = WhisperTranscriber(model_size=settings["whisper_model"])
    segments    = transcriber.transcribe(audio_path)
    raw_text    = WhisperTranscriber.segments_to_text(segments)

    # Step 3: Llama enhancements
    if not settings["use_llama"]:
        refined_text = raw_text
    else:
        client = LlamaClient(model=settings["llama_model"])

        yield "step", "Refining transcript with Llama..."
        refined_text = TranscriptRefiner(client).run(raw_text)

        if settings["use_speakers"]:
            yield "step", "Detecting speakers..."
            speakers_text = SpeakerDetector(client).run(refined_text)
            path = output.save_text(speakers_text, "_speakers.txt", "Speakers")
            results["Speakers"] = (path, speakers_text)

        if settings["use_summary"]:
            yield "step", "Generating summary..."
            summary_text = TranscriptSummariser(client).run(refined_text)
            path = output.save_text(summary_text, "_summary.txt", "Summary")
            results["Summary"] = (path, summary_text)

        if settings["use_actions"]:
            yield "step", "Extracting action items..."
            actions_text = ActionItemExtractor(client).run(refined_text)
            path = output.save_text(actions_text, "_actions.txt", "Action items")
            results["Action Items"] = (path, actions_text)

        if settings["language"]:
            yield "step", f"Translating to {settings['language']}..."
            translation_text = TranscriptTranslator(client).run(refined_text, settings["language"])
            path = output.save_text(
                translation_text,
                f"_translation_{settings['language']}.txt",
                f"Translation ({settings['language']})",
            )
            results[f"Translation ({settings['language']})"] = (path, translation_text)

    # Step 4: Save core outputs
    yield "step", "Saving transcript and subtitles..."
    transcript_path = output.save_text(refined_text, "_transcript.txt", "Transcript")
    srt_path        = output.save_srt(segments)

    results["Transcript"] = (transcript_path, refined_text)
    results["Subtitles (SRT)"] = (srt_path, srt_path.read_text(encoding="utf-8"))

    # Step 5: Cleanup
    audio_path.unlink(missing_ok=True)

    yield "done", (results, output.out_dir)


if run_button and uploaded_file is not None:
    settings = {
        "whisper_model": whisper_model,
        "llama_model":   llama_model,
        "use_llama":     use_llama,
        "use_speakers":  use_speakers and use_llama,
        "use_summary":   use_summary  and use_llama,
        "use_actions":   use_actions  and use_llama,
        "language":      language if use_llama else "",
    }

    # Save upload to a temp file
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = Path(tmp.name)

    # Rename temp file so OutputManager gets a nice stem
    named_path = tmp_path.parent / uploaded_file.name
    tmp_path.rename(named_path)
    video_path = named_path

    # Run pipeline with live status updates
    results = {}
    out_dir = None

    with st.status("Running pipeline...", expanded=True) as status:
        try:
            for event, payload in run_pipeline(video_path, settings):
                if event == "step":
                    st.write(payload)
                elif event == "done":
                    results, out_dir = payload

            status.update(label="✅ Done!", state="complete", expanded=False)
        except Exception as e:
            status.update(label=f"❌ Error: {e}", state="error", expanded=True)
            st.exception(e)
            st.stop()
        finally:
            video_path.unlink(missing_ok=True)

    st.session_state["results"] = results
    st.session_state["out_dir"] = out_dir


# ── Results display ───────────────────────────────────────────────────────────

if "results" in st.session_state and st.session_state["results"]:
    results = st.session_state["results"]
    out_dir: Path = st.session_state["out_dir"]

    st.divider()
    st.subheader("📄 Results")
    st.caption(f"Saved to: `{out_dir}`")

    # ── Download all as ZIP ───────────────────────────────────────────────────
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for label, (path, _) in results.items():
            zf.write(path, arcname=path.name)
    zip_buffer.seek(0)

    st.download_button(
        label="⬇️ Download all files as ZIP",
        data=zip_buffer,
        file_name=f"{out_dir.name}.zip",
        mime="application/zip",
        use_container_width=True,
    )

    st.divider()

    # ── Tab view for each output ──────────────────────────────────────────────
    tab_labels = list(results.keys())
    tabs = st.tabs(tab_labels)

    for tab, label in zip(tabs, tab_labels):
        path, content = results[label]
        with tab:
            st.text_area(
                label=label,
                value=content,
                height=400,
                label_visibility="collapsed",
            )
            st.download_button(
                label=f"⬇️ Download {path.name}",
                data=content.encode("utf-8"),
                file_name=path.name,
                mime="text/plain",
                key=f"dl_{label}",
            )