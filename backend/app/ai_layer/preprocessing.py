"""
Audio preprocessing utilities.

Responsibilities (Phase 1 only):
- Load an audio file from disk (any format supported by soundfile/librosa/ffmpeg).
- Convert audio to mono.
- Resample audio to 16 kHz (the standard input rate for most speech models
  used in later phases: clone detection, speaker verification, STT).
- Trim leading/trailing silence using a simple energy-based VAD.
- Report basic metadata about the processed audio.

No ML models are used here. This module only prepares audio so that later
phases (clone detection, speaker verification, STT) receive a consistent,
clean input format.
"""

from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


@dataclass
class ProcessedAudio:
    samples: np.ndarray
    sample_rate: int
    channels: int
    duration: float


def load_and_preprocess(file_path: str) -> ProcessedAudio:
    """
    Load an audio file, convert it to mono + 16kHz, and trim silence.

    librosa.load() already handles:
    - decoding common formats (wav, mp3, m4a, flac, etc. via soundfile/audioread)
    - downmixing to mono (mono=True)
    - resampling to the target sample rate

    So converting to mono + 16kHz is done in a single call, no manual
    resampling/mixing code needed.
    """
    samples, sample_rate = librosa.load(
        file_path, sr=TARGET_SAMPLE_RATE, mono=True
    )

    trimmed_samples = _trim_silence(samples)

    duration = float(len(trimmed_samples) / sample_rate)

    return ProcessedAudio(
        samples=trimmed_samples,
        sample_rate=sample_rate,
        channels=1,
        duration=round(duration, 2),
    )


def _trim_silence(samples: np.ndarray, top_db: int = 25) -> np.ndarray:
    """
    Trim leading/trailing silence using librosa's built-in energy-based VAD
    (librosa.effects.trim).

    Why librosa.effects.trim and not webrtcvad:
    - librosa is already a required dependency for loading/resampling audio,
      so trim() adds zero extra dependencies.
    - It works directly on the in-memory float waveform we already have,
      no extra format conversion (webrtcvad requires 16-bit PCM frames in
      fixed 10/20/30ms chunks).
    - For Phase 1's goal (basic silence trimming), energy-based thresholding
      is sufficient. A more accurate speech-specific VAD (e.g. webrtcvad or
      a neural VAD) can be introduced later if the fraud-detection pipeline
      needs frame-level speech/non-speech segmentation, at which point the
      added dependency would be justified.

    If trimming fails or removes everything (e.g. a silent file), fall back
    to the original samples so the pipeline doesn't crash.
    """
    if len(samples) == 0:
        return samples

    trimmed, _ = librosa.effects.trim(samples, top_db=top_db)

    if len(trimmed) == 0:
        return samples

    return trimmed


def save_processed_audio(processed: ProcessedAudio, output_path: str) -> None:
    """Write the processed (mono, 16kHz, trimmed) audio to disk as a WAV file."""
    sf.write(output_path, processed.samples, processed.sample_rate)