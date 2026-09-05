'@'
"""
Audio preprocessing pipeline for the voice clone/deepfake classifier
(Phase 2B -- preprocessing only, no model here).

Reuses app.ai_layer.preprocessing.load_and_preprocess() (Phase 1) for the
shared mono + 16kHz + silence-trim steps, so both the live upload endpoint
(routes_audio.py) and this training-data pipeline stay consistent.

On top of that, this module adds what's specific to preparing data for a
classifier:
    - safe amplitude normalization (peak-normalize, guarding against
      silent/near-silent clips)
    - converting the variable-length trimmed waveform into a FIXED-SIZE
      representation (pad short clips with zeros, truncate long clips),
      since most classifiers need consistent input dimensions

This module deliberately stops at a fixed-length raw waveform array -- it
does NOT compute model-specific features (e.g. MFCCs, mel-spectrograms,
Wav2Vec2 embeddings). Which representation a model actually wants is a
Phase 2C decision tied to the chosen architecture; this phase only makes
sure every sample is loaded, cleaned, and shaped consistently before that
choice is made.

The same function is applied to bonafide and spoof samples identically --
no class-specific logic exists in this pipeline, which matters for a fair
classifier (any preprocessing difference between classes would let the
model cheat by learning the preprocessing artifact instead of real
spoof cues).
"""

import os
import sys
from dataclasses import dataclass

import numpy as np

# Allow running this file directly (python app/ai_layer/audio_features.py)
# as well as as a module -- both need `app.ai_layer.preprocessing` to be
# importable, which requires the backend/ folder (parent of app/) on
# sys.path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ai_layer.preprocessing import TARGET_SAMPLE_RATE, load_and_preprocess

# 4 seconds at 16kHz is enough to cover the vast majority of ASVspoof
# utterances (they average ~2-4 sec) while keeping every sample the same
# fixed length for model training.
TARGET_DURATION_SECONDS = 4.0
TARGET_LENGTH = int(TARGET_DURATION_SECONDS * TARGET_SAMPLE_RATE)  # 64000 samples

SILENCE_EPSILON = 1e-8


@dataclass
class FeatureResult:
    waveform: np.ndarray  # shape: (TARGET_LENGTH,), dtype float32, range [-1, 1]
    original_duration: float
    was_padded: bool
    was_truncated: bool


def normalize_audio(samples: np.ndarray) -> np.ndarray:
    """
    Peak-normalize audio to the [-1, 1] range.

    Guards against divide-by-zero / blow-up on silent or near-silent clips
    (which do occur in real datasets) by only normalizing when the peak
    amplitude is above a small epsilon; otherwise the (already near-zero)
    samples are returned unchanged.
    """
    if samples.size == 0:
        return samples

    peak = np.max(np.abs(samples))
    if peak < SILENCE_EPSILON:
        return samples

    return samples / peak


def pad_or_truncate(samples: np.ndarray, target_length: int = TARGET_LENGTH):
    """
    Force `samples` to exactly `target_length` samples.

    - If longer: truncate from the start (keep the first `target_length`
      samples). Simple and deterministic; avoids randomness at this stage
      so preprocessing stays reproducible.
    - If shorter: zero-pad at the end.

    Returns (fixed_samples, was_padded, was_truncated).
    """
    current_length = len(samples)

    if current_length == target_length:
        return samples, False, False

    if current_length > target_length:
        return samples[:target_length], False, True

    padded = np.zeros(target_length, dtype=samples.dtype if samples.size else np.float32)
    padded[:current_length] = samples
    return padded, True, False


def extract_fixed_length_waveform(file_path: str) -> FeatureResult:
    """
    Full pipeline for one audio file:
        load -> mono + 16kHz + trim silence (Phase 1 logic) -> normalize
        -> pad/truncate to a fixed length

    This is the single function that should be called identically for
    every bonafide and spoof sample.
    """
    processed = load_and_preprocess(file_path)

    normalized = normalize_audio(processed.samples).astype(np.float32)

    fixed_waveform, was_padded, was_truncated = pad_or_truncate(normalized, TARGET_LENGTH)

    return FeatureResult(
        waveform=fixed_waveform,
        original_duration=processed.duration,
        was_padded=was_padded,
        was_truncated=was_truncated,
    )


if __name__ == "__main__":
    # Small smoke test: run the pipeline on a handful of files from the
    # prepared subset and print the resulting shapes, without writing
    # anything to disk. Useful to sanity-check the pipeline works before
    # wiring it into an actual training script in a later phase.
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the fixed-length audio preprocessing pipeline."
    )
    parser.add_argument(
        "--subset-dir",
        default=os.path.join("data", "asvspoof", "subset"),
        help="Path to the subset directory (expects bonafide/ and spoof/ subfolders)",
    )
    parser.add_argument("--samples-per-class", type=int, default=3, help="Files to test per class")
    args = parser.parse_args()

    for class_name in ("bonafide", "spoof"):
        class_dir = os.path.join(args.subset_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"{class_name}: directory not found ({class_dir})")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".flac", ".wav"))]
        print(f"\n{class_name} -- testing {min(args.samples_per_class, len(files))} file(s):")

        for filename in files[: args.samples_per_class]:
            path = os.path.join(class_dir, filename)
            result = extract_fixed_length_waveform(path)
            print(
                f"  {filename}: waveform shape={result.waveform.shape}, "
                f"original_duration={result.original_duration}s, "
                f"padded={result.was_padded}, truncated={result.was_truncated}"
            )
'@ | Set-Content -Encoding utf8 app\ai_layer\audio_features.py'