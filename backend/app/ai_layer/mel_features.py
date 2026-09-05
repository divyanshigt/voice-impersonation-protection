"""
Log-mel spectrogram feature extraction for the voice clone/deepfake
detector CNN (Phase 2C).

This module sits directly on top of Phase 2B's
`app.ai_layer.audio_features.extract_fixed_length_waveform`, which already
produces a clean, fixed-length (64000 samples, 16kHz, mono, normalized)
waveform identically for bonafide and spoof files. This module's only job
is to turn that waveform into the 2D representation the CNN expects:
a log-mel spectrogram.

Feature parameters (fixed, per Phase 2C spec):
    n_mels      = 64
    n_fft       = 1024
    hop_length  = 256

With a 64000-sample input, librosa produces a spectrogram of shape
approximately (n_mels, 251) -- exact frame count is computed once at
import time (see N_FRAMES below) so downstream code (model architecture,
training script) can rely on a fixed, known shape instead of recomputing
or hardcoding it.

This module does NOT do any audio loading, resampling, trimming, or
padding itself -- all of that is Phase 1/2B's job (kept there so there is
exactly one place that logic lives). This module starts from an already
fixed-length waveform array (or a file path, for convenience) and stops at
a log-mel spectrogram ready to be turned into a torch tensor.
"""

import os
import sys

import librosa
import numpy as np

# Allow running this file directly (python app/ai_layer/mel_features.py)
# as well as as a module -- both need `app.ai_layer.audio_features` to be
# importable, which requires the backend/ folder (parent of app/) on
# sys.path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ai_layer.audio_features import TARGET_LENGTH, TARGET_SAMPLE_RATE, extract_fixed_length_waveform

# Mel-spectrogram parameters -- fixed per the Phase 2C model spec.
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256

# Small epsilon to avoid log(0) when converting power to decibels for
# near-silent frames.
LOG_EPSILON = 1e-10

# Number of time frames librosa produces for a TARGET_LENGTH-sample input
# at the parameters above. Computed once here (rather than hardcoded) so
# it always matches reality even if TARGET_LENGTH/N_FFT/HOP_LENGTH change.
# librosa's default center=True padding gives:
#   n_frames = 1 + floor(n_samples / hop_length)
N_FRAMES = 1 + TARGET_LENGTH // HOP_LENGTH

# Expected output tensor shape: (1, N_MELS, N_FRAMES) once a channel
# dimension is added by the training script / model input.
FEATURE_SHAPE = (N_MELS, N_FRAMES)


def waveform_to_logmel(waveform: np.ndarray) -> np.ndarray:
    """
    Convert a fixed-length waveform (shape: (TARGET_LENGTH,), float32,
    range [-1, 1]) into a log-mel spectrogram of shape
    (N_MELS, N_FRAMES), dtype float32.

    Uses librosa.feature.melspectrogram (power spectrogram) followed by
    librosa.power_to_db to get log-scale values, which is the standard
    representation for CNN-based audio classifiers.
    """
    mel_spec = librosa.feature.melspectrogram(
        y=waveform,
        sr=TARGET_SAMPLE_RATE,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )

    log_mel = librosa.power_to_db(mel_spec, ref=np.max, amin=LOG_EPSILON)

    return log_mel.astype(np.float32)


def extract_mel_features(file_path: str) -> np.ndarray:
    """
    Full pipeline for one audio file, from disk to a log-mel spectrogram
    ready for the CNN:

        load + preprocess + pad/truncate (Phase 2B) -> log-mel (this module)

    Returns an array of shape (N_MELS, N_FRAMES), dtype float32.
    """
    feature_result = extract_fixed_length_waveform(file_path)
    return waveform_to_logmel(feature_result.waveform)


if __name__ == "__main__":
    # Small smoke test: run the pipeline on a handful of files from the
    # prepared subset and print the resulting shapes, without writing
    # anything to disk or touching the model.
    import argparse

    parser = argparse.ArgumentParser(
        description="Smoke-test the log-mel feature extraction pipeline."
    )
    parser.add_argument(
        "--subset-dir",
        default=os.path.join("data", "asvspoof", "subset"),
        help="Path to the subset directory (expects bonafide/ and spoof/ subfolders)",
    )
    parser.add_argument("--samples-per-class", type=int, default=3, help="Files to test per class")
    args = parser.parse_args()

    print(f"Expected feature shape: {FEATURE_SHAPE} (N_MELS={N_MELS}, N_FRAMES={N_FRAMES})")

    for class_name in ("bonafide", "spoof"):
        class_dir = os.path.join(args.subset_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"{class_name}: directory not found ({class_dir})")
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".flac", ".wav"))]
        print(f"\n{class_name} -- testing {min(args.samples_per_class, len(files))} file(s):")

        for filename in files[: args.samples_per_class]:
            path = os.path.join(class_dir, filename)
            log_mel = extract_mel_features(path)
            print(
                f"  {filename}: log_mel shape={log_mel.shape}, "
                f"min={log_mel.min():.2f}, max={log_mel.max():.2f}, "
                f"mean={log_mel.mean():.2f}"
            )