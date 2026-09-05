'@'
"""
Verify the prepared ASVspoof development subset.

Reads:
    data/asvspoof/subset/bonafide/*.flac (or .wav)
    data/asvspoof/subset/spoof/*.flac (or .wav)

For each file, reads only the audio HEADER (via soundfile.info) -- this is
fast and doesn't decode the full waveform, so checking ~2000 files stays
quick even on a normal laptop.

Prints:
    - total bonafide samples
    - total spoof samples
    - total samples
    - number of corrupted/unreadable audio files
    - audio duration statistics (min/max/mean/median)
    - sample rate statistics (distribution + most common)
    - file format statistics (extension counts)

Run:
    python app/ai_layer/dataset_check.py
"""

import argparse
import os
import statistics
from collections import Counter

import soundfile as sf

DEFAULT_SUBSET_DIR = os.path.join("data", "asvspoof", "subset")
AUDIO_EXTENSIONS = (".flac", ".wav")


def scan_class(class_dir: str, class_name: str):
    """
    Return a dict with per-file info for one class folder:
        - durations: list of float seconds (readable files only)
        - sample_rates: list of int (readable files only)
        - formats: Counter of file extensions
        - corrupted_files: list of filenames that failed to open
        - total_files: total files found (readable + corrupted)
    """
    durations = []
    sample_rates = []
    formats = Counter()
    corrupted_files = []

    if not os.path.isdir(class_dir):
        print(f"WARNING: {class_name} directory not found: {class_dir}")
        return {
            "durations": durations,
            "sample_rates": sample_rates,
            "formats": formats,
            "corrupted_files": corrupted_files,
            "total_files": 0,
        }

    files = [f for f in os.listdir(class_dir) if f.lower().endswith(AUDIO_EXTENSIONS)]

    for filename in files:
        ext = os.path.splitext(filename)[1].lower()
        formats[ext] += 1

        path = os.path.join(class_dir, filename)
        try:
            info = sf.info(path)
            if info.frames <= 0 or info.samplerate <= 0:
                raise ValueError("zero-length audio or invalid sample rate")
            duration = info.frames / info.samplerate
            durations.append(duration)
            sample_rates.append(info.samplerate)
        except Exception as exc:
            corrupted_files.append(filename)
            print(f"  Corrupted/unreadable file in {class_name}: {filename} ({exc})")

    return {
        "durations": durations,
        "sample_rates": sample_rates,
        "formats": formats,
        "corrupted_files": corrupted_files,
        "total_files": len(files),
    }


def print_duration_stats(durations):
    if not durations:
        print("  No readable files to compute duration statistics.")
        return
    print(f"  Count: {len(durations)}")
    print(f"  Min: {min(durations):.2f} sec")
    print(f"  Max: {max(durations):.2f} sec")
    print(f"  Mean: {statistics.mean(durations):.2f} sec")
    print(f"  Median: {statistics.median(durations):.2f} sec")


def print_sample_rate_stats(sample_rates):
    if not sample_rates:
        print("  No readable files to compute sample rate statistics.")
        return
    counts = Counter(sample_rates)
    most_common_rate, most_common_count = counts.most_common(1)[0]
    print(f"  Distinct sample rates: {dict(sorted(counts.items()))}")
    print(f"  Most common: {most_common_rate} Hz ({most_common_count} files)")


def print_format_stats(formats: Counter):
    if not formats:
        print("  No files found.")
        return
    for ext, count in sorted(formats.items()):
        print(f"  {ext}: {count} files")


def run_check(subset_dir: str) -> None:
    bonafide_dir = os.path.join(subset_dir, "bonafide")
    spoof_dir = os.path.join(subset_dir, "spoof")

    print(f"Checking dataset at: {subset_dir}")
    print("=" * 50)

    print("Scanning bonafide...")
    bonafide_info = scan_class(bonafide_dir, "bonafide")
    print("Scanning spoof...")
    spoof_info = scan_class(spoof_dir, "spoof")

    total_bonafide = bonafide_info["total_files"]
    total_spoof = spoof_info["total_files"]
    total_corrupted = len(bonafide_info["corrupted_files"]) + len(spoof_info["corrupted_files"])

    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total bonafide samples: {total_bonafide}")
    print(f"Total spoof samples: {total_spoof}")
    print(f"Total samples: {total_bonafide + total_spoof}")
    print(f"Corrupted/unreadable files: {total_corrupted}")

    all_durations = bonafide_info["durations"] + spoof_info["durations"]
    all_sample_rates = bonafide_info["sample_rates"] + spoof_info["sample_rates"]
    all_formats = bonafide_info["formats"] + spoof_info["formats"]

    print()
    print("Duration statistics (all readable files):")
    print_duration_stats(all_durations)

    print()
    print("Sample rate statistics (all readable files):")
    print_sample_rate_stats(all_sample_rates)

    print()
    print("File format statistics:")
    print_format_stats(all_formats)

    print()
    print("Per-class duration statistics:")
    print("Bonafide:")
    print_duration_stats(bonafide_info["durations"])
    print("Spoof:")
    print_duration_stats(spoof_info["durations"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify the prepared ASVspoof subset.")
    parser.add_argument("--subset-dir", default=DEFAULT_SUBSET_DIR, help="Path to the subset directory")
    args = parser.parse_args()

    run_check(args.subset_dir)
'@ | Set-Content -Encoding utf8 app\ai_layer\dataset_check.py'