# ── pipeline/enhancers.py ────────────────────────────────────────────────────
# All Llama-powered text enhancement features.
# Each enhancer is a separate class with a single run() method.
# They all receive a LlamaClient instance so they share one HTTP client.

from pipeline.llama import LlamaClient


class TranscriptRefiner:
    """
    Feature 1: Cleans up raw Whisper output.
    Fixes punctuation, capitalisation, removes filler words (um, uh, like).
    """

    def __init__(self, client: LlamaClient):
        self.client = client

    def run(self, raw_text: str) -> str:
        print("  -> Refining transcript...")
        prompt = (
            "You are a transcript editor. Clean up the following raw speech transcript: "
            "fix punctuation, capitalisation, and remove filler words (um, uh, like). "
            "Return ONLY the cleaned transcript text, no commentary.\n\n"
            f"TRANSCRIPT:\n{raw_text}"
        )
        return self.client.ask(prompt, "Refinement") or raw_text


class SpeakerDetector:
    """
    Feature 2: Identifies speaker changes and labels them.
    Adds Speaker 1:, Speaker 2: labels based on conversational context.
    Falls back to the refined text if Llama is unreachable.
    """

    def __init__(self, client: LlamaClient):
        self.client = client

    def run(self, refined_text: str) -> str:
        print("  -> Detecting speakers...")
        prompt = (
            "You are a transcript editor. Read the following transcript and identify where "
            "the speaker changes based on context, tone shifts, or conversational cues. "
            "Label each speaker's turn as 'Speaker 1:', 'Speaker 2:', etc. "
            "If it appears to be a single speaker throughout, label all lines as 'Speaker 1:'. "
            "Return ONLY the labelled transcript, no commentary or explanation.\n\n"
            f"TRANSCRIPT:\n{refined_text}"
        )
        return self.client.ask(prompt, "Speaker detection") or refined_text


class TranscriptSummariser:
    """
    Feature 3: Generates a concise 3-5 sentence summary of the transcript.
    """

    def __init__(self, client: LlamaClient):
        self.client = client

    def run(self, refined_text: str) -> str:
        print("  -> Generating summary...")
        prompt = (
            "You are a transcript summariser. Read the following transcript and write a "
            "clear, concise summary in 3 to 5 sentences covering the main topics discussed. "
            "Return ONLY the summary, no preamble or commentary.\n\n"
            f"TRANSCRIPT:\n{refined_text}"
        )
        return self.client.ask(prompt, "Summary") or "Summary could not be generated."


class ActionItemExtractor:
    """
    Feature 4: Extracts action items, tasks, and decisions from the transcript.
    Especially useful for meeting recordings.
    """

    def __init__(self, client: LlamaClient):
        self.client = client

    def run(self, refined_text: str) -> str:
        print("  -> Extracting action items...")
        prompt = (
            "You are a meeting notes assistant. Read the following transcript and extract "
            "all action items, tasks, decisions, and follow-ups that were mentioned. "
            "Format each item as a bullet point starting with '- '. "
            "If no clear action items are found, write: 'No action items identified.' "
            "Return ONLY the bullet list, no preamble or commentary.\n\n"
            f"TRANSCRIPT:\n{refined_text}"
        )
        return self.client.ask(prompt, "Action items") or "Action items could not be extracted."


class TranscriptTranslator:
    """
    Feature 5: Translates the refined transcript into a specified language.
    """

    def __init__(self, client: LlamaClient):
        self.client = client

    def run(self, refined_text: str, language: str) -> str:
        print(f"  -> Translating to {language}...")
        prompt = (
            f"You are a professional translator. Translate the following transcript into {language}. "
            "Preserve the meaning and natural flow of speech. "
            "Return ONLY the translated text, no commentary or explanation.\n\n"
            f"TRANSCRIPT:\n{refined_text}"
        )
        return self.client.ask(prompt, f"Translation ({language})") or "Translation could not be generated."
