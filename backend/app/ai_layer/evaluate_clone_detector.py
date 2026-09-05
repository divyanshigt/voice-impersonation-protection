"""
Phase 2D -- Evaluation of the already-trained voice clone detector.

Loads the existing trained model (models/clone_detector.pt) and its saved
config (models/model_config.json), re-extracts log-mel features for ONLY
the held-out TEST rows in data/asvspoof/metadata/dataset_splits.csv (same
split file used during training -- no new split is created), runs a
single forward pass in eval mode, computes the standard metric suite, and
saves the results + two plots.

This script deliberately reuses app.ai_layer.train_clone_detector's own
functions (load_split_dataframe, extract_features_for_split,
make_dataloader, run_epoch, compute_binary_metrics, print_metrics)
instead of reimplementing them, so the evaluation pipeline is guaranteed
to match the training pipeline exactly -- same feature extraction, same
label convention, same metric formulas.

This script does NOT:
    - train or fine-tune anything
    - create a new dataset split
    - modify dataset_splits.csv
    - overwrite clone_detector.pt or model_config.json

Outputs (written to models/, alongside the existing artifacts):
    models/evaluation_metrics.json
    models/roc_curve.png
    models/confusion_matrix.png

Run (from the backend/ directory, with the venv/session's dependencies
available):
    python app/ai_layer/evaluate_clone_detector.py

IMPORTANT if running on Colab: this script needs the actual test-set audio
files on disk (the same data/asvspoof/raw/ files used for training) to
re-extract features from -- it does not read features from clone_detector.pt.
If your Colab session ended after training, those files (which lived on
the ephemeral /content disk, not Drive) are gone. Re-run the notebook's
download + reorganize + prepare_dataset.py + dataset_split.py cells first
(same NUM_BONAFIDE/NUM_SPOOF and seed=42 as before reproduce the exact
same dataset_splits.csv deterministically) before running this script.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # write PNGs directly, no display needed
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve as sk_roc_curve

# Allow running this file directly (python app/ai_layer/evaluate_clone_detector.py)
# as well as as a module -- needs the backend/ folder (parent of app/) on
# sys.path when run directly, same convention as the other ai_layer files.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ai_layer.model_architecture import build_model
from app.ai_layer.train_clone_detector import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_MODELS_DIR,
    DEFAULT_SPLITS_CSV,
    compute_binary_metrics,
    extract_features_for_split,
    load_split_dataframe,
    make_dataloader,
    print_metrics,
    run_epoch,
)


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        print(f"ERROR: model_config.json not found at {config_path}.")
        raise SystemExit(1)
    with open(config_path, "r") as f:
        return json.load(f)


def load_trained_model(model_path: str, dropout: float, device: str):
    """Builds the architecture and loads the EXISTING trained weights.
    Does not train or modify the checkpoint in any way."""
    if not os.path.exists(model_path):
        print(f"ERROR: trained model not found at {model_path}. Run train_clone_detector.py first.")
        raise SystemExit(1)

    model = build_model(dropout=dropout).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, roc_auc: float, output_path: str) -> None:
    fpr, tpr, _ = sk_roc_curve(y_true.astype(int), y_prob)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve -- Clone Detector (Test Set)")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_confusion_matrix(cm_matrix: list, output_path: str) -> None:
    cm = np.array(cm_matrix)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["bonafide (0)", "spoof (1)"])

    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix -- Clone Detector (Test Set)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2D: evaluate the existing trained clone detector on the held-out test set only."
    )
    parser.add_argument("--splits-csv", default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ---- Load existing model + config (read-only) ----
    config = load_config(args.config_path)
    dropout = config["model"]["dropout"]

    model = load_trained_model(args.model_path, dropout, device)
    print(f"Loaded trained model -> {args.model_path} (weights not modified)")

    # ---- Load the EXISTING split, evaluate ONLY the test rows ----
    df = load_split_dataframe(args.splits_csv)
    X_test, y_test = extract_features_for_split(df, "test")

    if len(X_test) == 0:
        print(
            "\nERROR: no test-set audio files were found on disk (0 features extracted). "
            "If you're on Colab and the session restarted since training, the raw audio "
            "files on the ephemeral disk are gone -- re-run the dataset download + "
            "prepare_dataset.py + dataset_split.py cells (seed=42 reproduces the same "
            "split deterministically) before running this script."
        )
        raise SystemExit(1)

    print(f"Test set: X={X_test.shape}, y={y_test.shape}")

    test_loader = make_dataloader(X_test, y_test, args.batch_size, shuffle=False)
    criterion = nn.BCEWithLogitsLoss()

    # ---- Single forward pass, eval mode, no gradients, no training ----
    test_loss, y_true, y_prob = run_epoch(model, test_loader, criterion, optimizer=None, device=device)

    metrics = compute_binary_metrics(y_true, y_prob)
    metrics["loss"] = float(test_loss)
    print_metrics("test (Phase 2D evaluation)", metrics)

    os.makedirs(args.models_dir, exist_ok=True)

    # ---- Save metrics ----
    metrics_path = os.path.join(args.models_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation metrics saved -> {metrics_path}")

    # ---- Save plots ----
    roc_path = os.path.join(args.models_dir, "roc_curve.png")
    if metrics["roc_auc"] is not None:
        plot_roc_curve(y_true, y_prob, metrics["roc_auc"], roc_path)
        print(f"ROC curve saved -> {roc_path}")
    else:
        print("Skipped ROC curve: only one class present in test set.")

    cm_path = os.path.join(args.models_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics["confusion_matrix"]["matrix"], cm_path)
    print(f"Confusion matrix plot saved -> {cm_path}")

    print("\nEvaluation complete. clone_detector.pt and model_config.json were not modified.")


if __name__ == "__main__":
    main()