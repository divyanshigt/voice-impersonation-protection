'@'
"""
Prepare a small, balanced, local ASVspoof development subset.

This script does NOT download anything. It expects that the raw audio pool
and protocol/metadata file already exist locally, e.g.:

    data/asvspoof/raw/<utterance_id>.flac
    data/asvspoof/metadata/protocol.txt

That local data can come from either:
    (a) app/ai_layer/download_asvspoof_subset.py (streams a small pool, see
        that file's docstring), or
    (b) your own extraction of a small number of files from an official
        ASVspoof protocol + LA partition you already have locally.

Either way, the protocol file must follow the official ASVspoof protocol
column layout:

    SPEAKER_ID  AUDIO_FILE_NAME  -  SYSTEM_ID  KEY

    e.g. "LA_0079 LA_T_1138215 - - bonafide"
         "LA_0079 LA_T_1271820 - A01 spoof"

Where KEY is either "bonafide" or "spoof". This is the same format used by
the official ASVspoof2019.LA.cm.*.trn/trl.txt protocol files.

What this script does:
    1. Parse the protocol file -> list of (filename, label).
    2. Split into bonafide list and spoof list.
    3. Randomly select a balanced subset (default 1000 + 1000) using a
       fixed random seed, so the selection is reproducible.
    4. Copy the selected audio files from raw/ into subset/bonafide/ or
       subset/spoof/, preserving the original ASVspoof bonafide/spoof label
       via which folder they land in.
    5. Write a manifest CSV of exactly what was selected.
    6. Print the number of bonafide and spoof samples selected.

No model training or feature extraction happens here (Phase 2A only).
"""

import argparse
import csv
import os
import random
import shutil
from typing import Optional

DEFAULT_PROTOCOL_PATH = os.path.join("data", "asvspoof", "metadata", "protocol.txt")
DEFAULT_RAW_DIR = os.path.join("data", "asvspoof", "raw")
DEFAULT_SUBSET_DIR = os.path.join("data", "asvspoof", "subset")
DEFAULT_MANIFEST_PATH = os.path.join("data", "asvspoof", "metadata", "subset_manifest.csv")

AUDIO_EXTENSIONS = (".flac", ".wav")


def parse_protocol(protocol_path: str):
    """
    Read an ASVspoof-style protocol file and return two lists of filenames:
    (bonafide_filenames, spoof_filenames).

    Each protocol line has 5 whitespace-separated columns; we only need the
    2nd column (filename) and the 5th column (label).
    """
    bonafide_files = []
    spoof_files = []

    with open(protocol_path, "r") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                print(f"Skipping malformed protocol line {line_number}: '{line}'")
                continue

            filename = parts[1]
            label = parts[4].lower()

            if label == "bonafide":
                bonafide_files.append(filename)
            elif label == "spoof":
                spoof_files.append(filename)
            else:
                print(f"Skipping line {line_number} with unknown label '{label}'")

    return bonafide_files, spoof_files


def find_audio_file(raw_dir: str, filename: str) -> Optional[str]:
    """Locate the audio file on disk, trying supported extensions."""
    for ext in AUDIO_EXTENSIONS:
        candidate = os.path.join(raw_dir, filename + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def build_subset(
    protocol_path: str,
    raw_dir: str,
    subset_dir: str,
    manifest_path: str,
    num_bonafide: int,
    num_spoof: int,
    seed: int,
) -> None:
    bonafide_files, spoof_files = parse_protocol(protocol_path)

    print(f"Protocol file: {protocol_path}")
    print(f"Total bonafide entries in protocol: {len(bonafide_files)}")
    print(f"Total spoof entries in protocol: {len(spoof_files)}")

    random.seed(seed)

    selected_bonafide = random.sample(bonafide_files, min(num_bonafide, len(bonafide_files)))
    selected_spoof = random.sample(spoof_files, min(num_spoof, len(spoof_files)))

    if len(selected_bonafide) < num_bonafide:
        print(
            f"WARNING: only {len(selected_bonafide)} bonafide files available "
            f"in the raw pool (requested {num_bonafide}). "
            f"Fetch a larger raw pool if you need the full {num_bonafide}."
        )
    if len(selected_spoof) < num_spoof:
        print(
            f"WARNING: only {len(selected_spoof)} spoof files available "
            f"in the raw pool (requested {num_spoof}). "
            f"Fetch a larger raw pool if you need the full {num_spoof}."
        )

    bonafide_out_dir = os.path.join(subset_dir, "bonafide")
    spoof_out_dir = os.path.join(subset_dir, "spoof")
    os.makedirs(bonafide_out_dir, exist_ok=True)
    os.makedirs(spoof_out_dir, exist_ok=True)

    manifest_rows = []
    copied_bonafide = 0
    copied_spoof = 0
    missing_files = 0

    for filename in selected_bonafide:
        source_path = find_audio_file(raw_dir, filename)
        if source_path is None:
            missing_files += 1
            continue
        dest_path = os.path.join(bonafide_out_dir, os.path.basename(source_path))
        shutil.copy2(source_path, dest_path)
        manifest_rows.append([filename, "bonafide", dest_path])
        copied_bonafide += 1

    for filename in selected_spoof:
        source_path = find_audio_file(raw_dir, filename)
        if source_path is None:
            missing_files += 1
            continue
        dest_path = os.path.join(spoof_out_dir, os.path.basename(source_path))
        shutil.copy2(source_path, dest_path)
        manifest_rows.append([filename, "spoof", dest_path])
        copied_spoof += 1

    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "subset_path"])
        writer.writerows(manifest_rows)

    if missing_files:
        print(
            f"WARNING: {missing_files} selected files were listed in the "
            f"protocol but not found in {raw_dir} (skipped)."
        )

    print(f"Manifest written -> {manifest_path}")
    print("---")
    print(f"Bonafide samples selected: {copied_bonafide}")
    print(f"Spoof samples selected: {copied_spoof}")
    print(f"Total subset size: {copied_bonafide + copied_spoof}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a balanced, seeded ASVspoof development subset from local raw audio + protocol."
    )
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL_PATH, help="Path to protocol/metadata file")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="Directory containing raw audio files")
    parser.add_argument("--subset-dir", default=DEFAULT_SUBSET_DIR, help="Output directory for the subset")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH, help="Path to write the subset manifest CSV")
    parser.add_argument("--num-bonafide", type=int, default=1000, help="Number of bonafide samples to select")
    parser.add_argument("--num-spoof", type=int, default=1000, help="Number of spoof samples to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible selection")
    args = parser.parse_args()

    build_subset(
        protocol_path=args.protocol,
        raw_dir=args.raw_dir,
        subset_dir=args.subset_dir,
        manifest_path=args.manifest,
        num_bonafide=args.num_bonafide,
        num_spoof=args.num_spoof,
        seed=args.seed,
    )
'@ | Set-Content -Encoding utf8 app\ai_layer\prepare_dataset.py'