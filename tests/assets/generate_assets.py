#!/usr/bin/env python3
"""
generate_assets.py
Generates synthetic test audio/video assets for real file tests.
Run once to create all assets before running pytest -m real.

Usage:
    cd ~/Documents/VoiceToText
    source venv/bin/activate
    python tests/assets/generate_assets.py

Requirements:
    pip install numpy          (already in venv via openai-whisper)
    ffmpeg                     (already installed)

What it generates:
    english_clear.mp4          — sine wave tones simulating clear speech rhythm
    silence.mp4                — complete silence
    background_noise.mp4       — speech tones mixed with white noise
    short_clip.mp4             — very short 2s clip (edge case)
    multi_tone.mp4             — alternating tones (simulates speaker changes)
"""

import wave
import struct
import math
import subprocess
import os
import sys

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

ASSETS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RATE = 16000


def write_wav(filename: str, samples: list, sample_rate: int = SAMPLE_RATE) -> str:
    """Write a list of float samples [-1.0, 1.0] to a WAV file."""
    path = os.path.join(ASSETS_DIR, filename)
    with wave.open(path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for s in samples:
            clamped = max(-1.0, min(1.0, s))
            wf.writeframes(struct.pack('<h', int(clamped * 32767)))
    return path


def wav_to_mp4(wav_path: str, mp4_path: str) -> str:
    """Convert WAV to MP4 using ffmpeg."""
    full_mp4 = os.path.join(ASSETS_DIR, mp4_path)
    cmd = [
        "ffmpeg", "-y",
        "-i", wav_path,
        "-c:a", "aac",
        "-vn",
        full_mp4
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print(f"  ffmpeg error for {mp4_path}: {result.stderr.decode()[:200]}")
    return full_mp4


def sine_wave(freq: float, duration: float, amplitude: float = 0.5,
              sample_rate: int = SAMPLE_RATE) -> list:
    """Generate a sine wave at given frequency and duration."""
    n = int(duration * sample_rate)
    return [amplitude * math.sin(2 * math.pi * freq * i / sample_rate) for i in range(n)]


def silence(duration: float, sample_rate: int = SAMPLE_RATE) -> list:
    """Generate silence."""
    return [0.0] * int(duration * sample_rate)


def white_noise(duration: float, amplitude: float = 0.1,
                sample_rate: int = SAMPLE_RATE) -> list:
    """Generate white noise using numpy if available, else pseudo-random."""
    n = int(duration * sample_rate)
    if HAS_NUMPY:
        return (np.random.uniform(-amplitude, amplitude, n)).tolist()
    else:
        import random
        return [random.uniform(-amplitude, amplitude) for _ in range(n)]


def mix(samples_a: list, samples_b: list) -> list:
    """Mix two sample lists together (pad shorter with zeros)."""
    length = max(len(samples_a), len(samples_b))
    result = []
    for i in range(length):
        a = samples_a[i] if i < len(samples_a) else 0.0
        b = samples_b[i] if i < len(samples_b) else 0.0
        result.append(max(-1.0, min(1.0, a + b)))
    return result


def speech_rhythm(duration: float, sample_rate: int = SAMPLE_RATE) -> list:
    """
    Simulate speech rhythm — alternating tones and short pauses.
    Produces audio that Whisper will attempt to transcribe.
    """
    samples = []
    t = 0.0
    freqs = [200, 250, 180, 220, 260, 190, 240]  # vocal range frequencies
    i = 0
    while t < duration:
        # word-like tone burst (0.2-0.4s)
        word_len = 0.25
        samples += sine_wave(freqs[i % len(freqs)], word_len, amplitude=0.4)
        t += word_len
        # brief pause between "words"
        pause_len = 0.1
        samples += silence(pause_len)
        t += pause_len
        i += 1
    return samples[:int(duration * sample_rate)]


# ── Asset generators ──────────────────────────────────────────────────────────

def generate_english_clear():
    """Clear speech-rhythm audio — primary test asset."""
    print("  Generating english_clear.mp4...")
    samples = speech_rhythm(duration=8.0)
    wav = write_wav("english_clear.wav", samples)
    mp4 = wav_to_mp4(wav, "english_clear.mp4")
    os.remove(wav)
    print(f"  -> {mp4}")


def generate_silence():
    """Complete silence — tests edge case of no speech."""
    print("  Generating silence.mp4...")
    samples = silence(duration=5.0)
    wav = write_wav("silence.wav", samples)
    mp4 = wav_to_mp4(wav, "silence.mp4")
    os.remove(wav)
    print(f"  -> {mp4}")


def generate_background_noise():
    """Speech rhythm mixed with white noise — tests noisy audio handling."""
    print("  Generating background_noise.mp4...")
    speech = speech_rhythm(duration=8.0)
    noise  = white_noise(duration=8.0, amplitude=0.15)
    samples = mix(speech, noise)
    wav = write_wav("background_noise.wav", samples)
    mp4 = wav_to_mp4(wav, "background_noise.mp4")
    os.remove(wav)
    print(f"  -> {mp4}")


def generate_short_clip():
    """Very short 2-second clip — tests minimal audio edge case."""
    print("  Generating short_clip.mp4...")
    samples = speech_rhythm(duration=2.0)
    wav = write_wav("short_clip.wav", samples)
    mp4 = wav_to_mp4(wav, "short_clip.mp4")
    os.remove(wav)
    print(f"  -> {mp4}")


def generate_multi_tone():
    """
    Two distinct tone patterns separated by silence — simulates speaker changes.
    Pattern A: low frequencies (Speaker 1)
    Pattern B: high frequencies (Speaker 2)
    """
    print("  Generating multi_tone.mp4...")
    # Speaker 1 pattern — lower frequencies
    speaker1 = []
    for _ in range(3):
        speaker1 += sine_wave(150, 0.3, amplitude=0.4)
        speaker1 += silence(0.1)

    gap = silence(0.5)

    # Speaker 2 pattern — higher frequencies
    speaker2 = []
    for _ in range(3):
        speaker2 += sine_wave(350, 0.3, amplitude=0.4)
        speaker2 += silence(0.1)

    samples = speaker1 + gap + speaker2 + gap + speaker1 + gap + speaker2
    wav = write_wav("multi_tone.wav", samples)
    mp4 = wav_to_mp4(wav, "multi_tone.mp4")
    os.remove(wav)
    print(f"  -> {mp4}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating test assets...")
    print(f"Output directory: {ASSETS_DIR}\n")

    generate_english_clear()
    generate_silence()
    generate_background_noise()
    generate_short_clip()
    generate_multi_tone()

    print("\nDone. Assets generated:")
    for f in sorted(os.listdir(ASSETS_DIR)):
        if f.endswith(".mp4"):
            size = os.path.getsize(os.path.join(ASSETS_DIR, f))
            print(f"  {f:35s}  {size:,} bytes")

    print("\nYou can now run real file tests:")
    print("  pytest -m real -v")
