"""
One-time helper script: fetch a SMALL local pool of ASVspoof 2019 LA audio,
without downloading the full official archive.

Why this script exists:
    The official ASVspoof 2019 LA dataset is distributed by the University of
    Edinburgh (https://datashare.ed.ac.uk/handle/10283/3336) as a single
    LA.zip file that bundles the train + dev + eval partitions together.
    That single archive is ~7.1 GB and there is no official way to download
    only part of it.

    To avoid pulling 7GB+ for a local development subset, this script uses
    the Hugging Face `datasets` library in STREAMING mode against a
    redistribution of the same official ASVspoof 2019 LA data
    (unmodified audio + labels, released under ODC-By license). Streaming
    mode downloads audio ONE EXAMPLE AT A TIME as we iterate, so we only
    pull the ~2,400 files this script actually reads, not the full dataset.

    The dataset used here corresponds to the official LA eval partition:
    "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA" on the Hugging Face Hub.
    Its `label` field (0 = bonafide, 1 = spoof) comes directly from the
    official protocol file ASVspoof2019.LA.cm.eval.trl.txt.

Output:
    - Raw audio files      -> data/asvspoof/raw/*.flac
    - Protocol/metadata    -> data/asvspoof/metadata/protocol.txt
      (written in the same 5-column format as the official ASVspoof
      protocol files: SPEAKER_ID FILENAME - SYSTEM_ID KEY)

This script only builds a RAW POOL (default: 1200 bonafide + 1200 spoof,
slightly larger than the 1000/1000 we ultimately need). The actual
balanced, seeded 1000/1000 selection happens in prepare_dataset.py, which
reads the protocol file this script writes -- exactly as it would if you
had extracted the official LA.zip yourself.

Run this once:
    python app/ai_layer/download_asvspoof_subset.py
"""

import argparse
import os

import soundfile as sf

RAW_DIR = os.path.join("data", "asvspoof", "raw")
METADATA_DIR = os.path.join("data", "asvspoof", "metadata")
PROTOCOL_PATH = os.path.join(METADATA_DIR, "protocol.txt")

HF_DATASET_NAME = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
HF_SPLIT = "test"  # this dataset only ships the official LA eval partition


def fetch_pool(num_bonafide: int, num_spoof: int, seed: int) -> None:
    # Imported here so the rest of the project (routes_audio.py, etc.)
    # never needs the `datasets`/`huggingface_hub` packages installed.
    from datasets import load_dataset

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

    print(f"Streaming from Hugging Face dataset: {HF_DATASET_NAME} ({HF_SPLIT})")
    print("Only the files written below are downloaded -- not the full dataset.\n")

    stream = load_dataset(HF_DATASET_NAME, split=HF_SPLIT, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=2000)

    bonafide_count = 0
    spoof_count = 0
    protocol_lines = []

    for example in stream:
        if bonafide_count >= num_bonafide and spoof_count >= num_spoof:
            break

        label_id = example["label"]  # 0 = bonafide, 1 = spoof (ClassLabel)
        label_name = "bonafide" if label_id == 0 else "spoof"

        if label_name == "bonafide" and bonafide_count >= num_bonafide:
            continue
        if label_name == "spoof" and spoof_count >= num_spoof:
            continue

        utterance_id = os.path.splitext(example["path"])[0]
        out_path = os.path.join(RAW_DIR, f"{utterance_id}.flac")

        audio = example["audio"]
        sf.write(out_path, audio["array"], audio["sampling_rate"])

        system_id = "-" if label_name == "bonafide" else "unknown"
        protocol_lines.append(f"SPK {utterance_id} - {system_id} {label_name}")

        if label_name == "bonafide":
            bonafide_count += 1
        else:
            spoof_count += 1

    with open(PROTOCOL_PATH, "w") as f:
        f.write("\n".join(protocol_lines) + "\n")

    print(f"Saved {bonafide_count} bonafide + {spoof_count} spoof raw audio files -> {RAW_DIR}")
    print(f"Wrote protocol/metadata file -> {PROTOCOL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch a small ASVspoof 2019 LA audio pool.")
    parser.add_argument("--bonafide", type=int, default=1200, help="Raw bonafide files to fetch")
    parser.add_argument("--spoof", type=int, default=1200, help="Raw spoof files to fetch")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    args = parser.parse_args()

    fetch_pool(args.bonafide, args.spoof, args.seed)