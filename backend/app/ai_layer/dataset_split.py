'@'
"""
Create a train/validation/test split (70/15/15) over the prepared ASVspoof
subset.

This does NOT copy or duplicate any audio files. It only writes a metadata
CSV recording which split each file belongs to
(data/asvspoof/metadata/dataset_splits.csv), so a future training script
can look up "give me all train files" without re-scanning or duplicating
~2000 audio files.

Data leakage avoidance:
    - Splitting is done separately per class (bonafide/spoof) and then
      combined, so both classes stay balanced in every split.
    - Every file appears in exactly one split (train, val, or test) --
      never more than one.
    - Fixed random seed (default 42) for reproducibility.
    - SPEAKER-LEVEL LEAKAGE: the ideal approach groups all utterances from
      the same speaker into a single split, so no speaker's voice is seen
      in both train and test. This script attempts that IF the local
      protocol/metadata file contains real, distinct speaker IDs.

      IMPORTANT LIMITATION WITH THE CURRENT DATA: the Phase 2A download
      helper (download_asvspoof_subset.py) wrote a PLACEHOLDER speaker id
      ("SPK") for every single row, because the Hugging Face redistribution
      of ASVspoof2019 LA used for streaming does not expose real per-file
      speaker identity. That means genuine speaker IDs are NOT currently
      available in data/asvspoof/metadata/protocol.txt.

      This script detects that case (too few distinct speaker IDs to be
      meaningful) and automatically falls back to a stratified random
      split by filename, printing a clear warning so this limitation is
      visible rather than silently ignored. If you later obtain a protocol
      file with real speaker IDs (e.g. from the official LA.zip), re-run
      this script against it and it will automatically use proper
      speaker-grouped splitting instead.

Run:
    python app/ai_layer/dataset_split.py
"""

import argparse
import csv
import os
from collections import defaultdict
import random

DEFAULT_SUBSET_DIR = os.path.join("data", "asvspoof", "subset")
DEFAULT_MANIFEST_PATH = os.path.join("data", "asvspoof", "metadata", "subset_manifest.csv")
DEFAULT_PROTOCOL_PATH = os.path.join("data", "asvspoof", "metadata", "protocol.txt")
DEFAULT_SPLITS_PATH = os.path.join("data", "asvspoof", "metadata", "dataset_splits.csv")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# If a protocol file has fewer distinct speaker IDs than this (relative to
# file count), we treat speaker info as non-informative/placeholder and
# fall back to a plain stratified split instead of speaker-grouped split.
MIN_DISTINCT_SPEAKERS_FOR_GROUPING = 2


def load_manifest(manifest_path: str, subset_dir: str):
    """
    Return a list of (filename, label, path) tuples.

    Prefers the manifest CSV written by prepare_dataset.py (has the exact
    files that were selected). Falls back to scanning the subset folders
    directly if the manifest is missing.
    """
    entries = []

    if os.path.exists(manifest_path):
        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append((row["filename"], row["label"], row["subset_path"]))
        return entries

    print(f"Manifest not found at {manifest_path}, scanning {subset_dir} directly.")
    for label in ("bonafide", "spoof"):
        class_dir = os.path.join(subset_dir, label)
        if not os.path.isdir(class_dir):
            continue
        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".flac", ".wav")):
                utterance_id = os.path.splitext(filename)[0]
                entries.append((utterance_id, label, os.path.join(class_dir, filename)))

    return entries


def load_speaker_map(protocol_path: str):
    """
    Read filename -> speaker_id from the protocol file, if it exists.
    Returns an empty dict if the protocol file is missing.
    """
    speaker_map = {}
    if not os.path.exists(protocol_path):
        return speaker_map

    with open(protocol_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            speaker_id, filename = parts[0], parts[1]
            speaker_map[filename] = speaker_id

    return speaker_map


def speaker_info_is_usable(speaker_map: dict, entries: list) -> bool:
    """
    Decide whether speaker IDs in speaker_map are informative enough to
    group by, or just a non-informative placeholder (e.g. every file
    mapped to the same single speaker id).
    """
    if not speaker_map:
        return False

    relevant_speaker_ids = {
        speaker_map[filename] for filename, _, _ in entries if filename in speaker_map
    }

    return len(relevant_speaker_ids) >= MIN_DISTINCT_SPEAKERS_FOR_GROUPING


def stratified_random_split(filenames: list, seed: int):
    """
    Shuffle a list of filenames (fixed seed) and cut it into
    train/val/test according to TRAIN_RATIO/VAL_RATIO/TEST_RATIO.
    """
    shuffled = list(filenames)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_train = round(n * TRAIN_RATIO)
    n_val = round(n * VAL_RATIO)
    # whatever remains goes to test, so rounding never drops/duplicates a file
    n_test = n - n_train - n_val

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]

    return train, val, test


def speaker_grouped_split(filenames: list, speaker_map: dict, seed: int):
    """
    Group filenames by speaker_id, shuffle the GROUPS (fixed seed), then
    greedily assign whole groups to train/val/test to approximate the
    target ratios without ever splitting one speaker across splits.
    """
    groups = defaultdict(list)
    for filename in filenames:
        speaker_id = speaker_map.get(filename, f"__unknown__{filename}")
        groups[speaker_id].append(filename)

    group_list = list(groups.values())
    random.Random(seed).shuffle(group_list)

    n = len(filenames)
    target_train = n * TRAIN_RATIO
    target_val = n * VAL_RATIO

    train, val, test = [], [], []
    for group in group_list:
        if len(train) < target_train:
            train.extend(group)
        elif len(val) < target_val:
            val.extend(group)
        else:
            test.extend(group)

    return train, val, test


def build_split(entries: list, speaker_map: dict, seed: int):
    use_speaker_grouping = speaker_info_is_usable(speaker_map, entries)

    if use_speaker_grouping:
        print("Real, distinct speaker IDs found in protocol -- using speaker-grouped splitting.")
    else:
        print(
            "WARNING: no usable speaker identity information found in the local protocol "
            "file (data/asvspoof/metadata/protocol.txt only contains a placeholder speaker "
            "id for every row -- see this script's docstring for why). Falling back to a "
            "stratified random split by filename. Speaker-level leakage between splits "
            "cannot be ruled out with the current metadata."
        )

    by_class = defaultdict(list)
    for filename, label, _path in entries:
        by_class[label].append(filename)

    split_assignment = {}  # filename -> "train"/"val"/"test"

    for label, filenames in by_class.items():
        if use_speaker_grouping:
            train, val, test = speaker_grouped_split(filenames, speaker_map, seed)
        else:
            train, val, test = stratified_random_split(filenames, seed)

        for filename in train:
            split_assignment[filename] = "train"
        for filename in val:
            split_assignment[filename] = "val"
        for filename in test:
            split_assignment[filename] = "test"

    return split_assignment


def write_split_csv(entries: list, split_assignment: dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "split", "path"])
        for filename, label, path in entries:
            split = split_assignment.get(filename, "unassigned")
            writer.writerow([filename, label, split, path])


def print_summary(entries: list, split_assignment: dict):
    counts = defaultdict(lambda: defaultdict(int))
    for filename, label, _path in entries:
        split = split_assignment.get(filename, "unassigned")
        counts[split][label] += 1

    def show(split_name):
        print(f"{split_name.capitalize()}:")
        print(f"  Bonafide: {counts[split_name]['bonafide']}")
        print(f"  Spoof: {counts[split_name]['spoof']}")

    print()
    print("=" * 40)
    print("SPLIT SUMMARY")
    print("=" * 40)
    show("train")
    show("val")
    show("test")

    if counts["unassigned"]:
        print(f"WARNING: {sum(counts['unassigned'].values())} files were not assigned to any split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a train/val/test split for the ASVspoof subset.")
    parser.add_argument("--subset-dir", default=DEFAULT_SUBSET_DIR)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument("--output", default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    entries = load_manifest(args.manifest, args.subset_dir)
    if not entries:
        print("No dataset entries found. Run prepare_dataset.py first.")
        raise SystemExit(1)

    speaker_map = load_speaker_map(args.protocol)
    split_assignment = build_split(entries, speaker_map, args.seed)

    write_split_csv(entries, split_assignment, args.output)
    print(f"\nSplit metadata written -> {args.output}")

    print_summary(entries, split_assignment)
'@ | Set-Content -Encoding utf8 app\ai_layer\dataset_split.py'