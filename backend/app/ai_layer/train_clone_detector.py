"""
Train the voice clone / speech deepfake detector CNN (Phase 2C).

Reads the train/val/test split written by Phase 2B
(data/asvspoof/metadata/dataset_splits.csv, columns:
 filename, label, split, path -- label in {"bonafide","spoof"},
 split in {"train","val","test"}), extracts log-mel features via
app.ai_layer.mel_features, trains app.ai_layer.model_architecture's
CloneDetectorCNN with early stopping on validation loss, evaluates once
on the held-out test set, and writes:

    backend/models/clone_detector.pt    (best model state_dict)
    backend/models/model_config.json    (architecture/feature/training
                                          config + final test metrics)

Label convention (must match dataset_splits.csv and the model):
    0 = bonafide / real
    1 = spoof / synthetic

Run (from the backend/ directory, with the venv activated):
    python app/ai_layer/train_clone_detector.py

Optional arguments (defaults shown; see "LR fix" note below):
    --epochs 50 --batch-size 32 --lr 3e-4 --weight-decay 1e-4
    --patience 7 --seed 42

LR fix (applied after the first training run showed unstable/spiking val
loss with the original lr=1e-3, patience=5 defaults -- see
training_history in model_config.json from that run):
    - Default learning rate lowered from 1e-3 to 3e-4. With only ~44
      batches per epoch and BatchNorm layers, 1e-3 was large enough to
      cause the optimizer to periodically overshoot, producing the val
      loss spikes (0.57 -> 1.15 -> 0.94 -> 1.37 -> 1.42) seen in that run.
    - Added torch.optim.lr_scheduler.ReduceLROnPlateau: if val loss
      doesn't improve for 2 epochs, LR is halved (factor=0.5). This lets
      training recover automatically from a spike instead of continuing
      at a learning rate that's too large for the current loss landscape.
    - Default early-stopping patience raised from 5 to 7, since a lower
      LR converges more slowly and the scheduler itself needs 2 stalled
      epochs to react before patience should start counting down harder.
    - Nothing else changed: same architecture, same features, same
      dataset split, same loss/optimizer type, same seed.
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running this file directly (python app/ai_layer/train_clone_detector.py)
# as well as as a module -- needs the backend/ folder (parent of app/) on
# sys.path when run directly, same convention as the other ai_layer files.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ai_layer.mel_features import (
    FEATURE_SHAPE,
    HOP_LENGTH,
    N_FFT,
    N_FRAMES,
    N_MELS,
    extract_mel_features,
)
from app.ai_layer.model_architecture import build_model
from app.ai_layer.audio_features import TARGET_DURATION_SECONDS, TARGET_LENGTH
from app.ai_layer.preprocessing import TARGET_SAMPLE_RATE

DEFAULT_SPLITS_CSV = os.path.join("data", "asvspoof", "metadata", "dataset_splits.csv")
DEFAULT_MODELS_DIR = os.path.join("models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODELS_DIR, "clone_detector.pt")
DEFAULT_CONFIG_PATH = os.path.join(DEFAULT_MODELS_DIR, "model_config.json")

# Label convention -- MUST stay consistent with dataset_splits.csv string
# labels and with the "0 = bonafide, 1 = spoof" convention used everywhere
# else in Phase 2C (mel_features docstring, model_architecture docstring).
LABEL_TO_INT = {"bonafide": 0, "spoof": 1}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split_dataframe(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        print(f"ERROR: dataset_splits.csv not found at {csv_path}. Run app/ai_layer/dataset_split.py first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path)

    expected_columns = {"filename", "label", "split", "path"}
    if not expected_columns.issubset(df.columns):
        print(f"ERROR: {csv_path} is missing expected columns. Found: {df.columns.tolist()}")
        raise SystemExit(1)

    unknown_labels = set(df["label"].unique()) - set(LABEL_TO_INT.keys())
    if unknown_labels:
        print(f"ERROR: unexpected label values in dataset_splits.csv: {unknown_labels}")
        raise SystemExit(1)

    return df


def extract_features_for_split(df: pd.DataFrame, split_name: str) -> tuple:
    """
    Extract log-mel features + integer labels for every row belonging to
    `split_name` ("train"/"val"/"test").

    Features are computed once here (not on-the-fly every epoch) since the
    dataset is small (~2000 files total) and this keeps epoch time fast --
    audio decoding + mel extraction is the expensive part, not the model
    forward pass.

    Returns (X, y) as numpy arrays:
        X: shape (n_samples, 1, N_MELS, N_FRAMES), dtype float32
        y: shape (n_samples,), dtype float32 (0.0 or 1.0)
    """
    split_df = df[df["split"] == split_name].reset_index(drop=True)
    if len(split_df) == 0:
        print(f"WARNING: no rows found for split='{split_name}'.")

    features = []
    labels = []

    start = time.time()
    for i, row in split_df.iterrows():
        file_path = row["path"]
        if not os.path.exists(file_path):
            print(f"WARNING: file not found, skipping: {file_path}")
            continue

        log_mel = extract_mel_features(file_path)  # (N_MELS, N_FRAMES)
        features.append(log_mel)
        labels.append(LABEL_TO_INT[row["label"]])

        if (i + 1) % 200 == 0:
            print(f"  [{split_name}] processed {i + 1}/{len(split_df)} files...")

    elapsed = time.time() - start
    print(f"[{split_name}] extracted {len(features)} feature arrays in {elapsed:.1f}s")

    X = np.stack(features, axis=0).astype(np.float32)  # (n, N_MELS, N_FRAMES)
    X = np.expand_dims(X, axis=1)  # (n, 1, N_MELS, N_FRAMES) -- add channel dim
    y = np.array(labels, dtype=np.float32)

    return X, y


def make_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """
    Run one epoch. If `optimizer` is provided, runs in training mode
    (backprop + step); otherwise runs in eval mode (no grad).

    Returns (avg_loss, all_labels, all_probs) so callers can compute
    whatever metrics they need (accuracy during training, full metric
    suite on the final test pass).
    """
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    n_samples = 0
    all_labels = []
    all_probs = []

    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)  # (batch,) -> (batch, 1)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = X_batch.size(0)
            total_loss += loss.item() * batch_size
            n_samples += batch_size

            probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
            all_probs.extend(probs.tolist())
            all_labels.extend(y_batch.detach().cpu().numpy().flatten().tolist())

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss, np.array(all_labels), np.array(all_probs)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute the metrics required by the Phase 2C spec:
    Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix, EER.

    Uses scikit-learn for standard metrics. EER (Equal Error Rate) is
    computed manually from the ROC curve: it's the point where the false
    positive rate equals the false negative rate (1 - true positive rate).
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    y_pred = (y_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true_int, y_pred)),
        "precision": float(precision_score(y_true_int, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_int, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_int, y_pred, zero_division=0)),
    }

    # ROC-AUC and EER both require both classes to be present.
    if len(np.unique(y_true_int)) < 2:
        print("WARNING: only one class present in this split -- skipping ROC-AUC/EER.")
        metrics["roc_auc"] = None
        metrics["eer"] = None
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true_int, y_prob))

        fpr, tpr, _ = roc_curve(y_true_int, y_prob)
        fnr = 1 - tpr
        # EER is where FPR and FNR curves cross; find the index minimizing
        # |fpr - fnr| and take the average of the two values there as the
        # standard EER estimate.
        eer_index = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[eer_index] + fnr[eer_index]) / 2.0)
        metrics["eer"] = eer

    cm = confusion_matrix(y_true_int, y_pred, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "labels": ["bonafide (0)", "spoof (1)"],
        "matrix": cm.tolist(),  # [[TN, FP], [FN, TP]]
    }

    return metrics


def print_metrics(split_name: str, metrics: dict) -> None:
    print(f"\n{split_name.upper()} METRICS")
    print("-" * 40)
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    if metrics["eer"] is not None:
        print(f"  EER:       {metrics['eer']:.4f}")
    cm = metrics["confusion_matrix"]["matrix"]
    print(f"  Confusion Matrix [[TN, FP], [FN, TP]]: {cm}")


def main():
    parser = argparse.ArgumentParser(description="Train the Phase 2C voice clone detector CNN.")
    parser.add_argument("--splits-csv", default=DEFAULT_SPLITS_CSV)
    parser.add_argument("--models-dir", default=DEFAULT_MODELS_DIR)
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs (early stopping will likely stop sooner)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4, help="Lowered from the original 1e-3 -- see LR fix note in the module docstring")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience (epochs without val-loss improvement)")
    parser.add_argument("--scheduler-patience", type=int, default=2, help="Epochs without val-loss improvement before ReduceLROnPlateau halves the LR")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Multiplicative LR decay factor applied by ReduceLROnPlateau")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    # Auto-detects a GPU (e.g. Colab's T4) if available, falls back to CPU
    # otherwise (e.g. a local laptop run) -- no other code path changes
    # based on device; everything else in this script is device-agnostic.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # ---- Load split metadata ----
    df = load_split_dataframe(args.splits_csv)
    print(f"Loaded {len(df)} rows from {args.splits_csv}")

    # ---- Feature extraction (once, cached in memory) ----
    print("\nExtracting log-mel features for each split...")
    X_train, y_train = extract_features_for_split(df, "train")
    X_val, y_val = extract_features_for_split(df, "val")
    X_test, y_test = extract_features_for_split(df, "test")

    print(f"\nTrain: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")

    train_loader = make_dataloader(X_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, args.batch_size, shuffle=False)
    test_loader = make_dataloader(X_test, y_test, args.batch_size, shuffle=False)

    # ---- Model / loss / optimizer ----
    model = build_model(dropout=args.dropout).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Halves the LR if val loss stalls for `scheduler_patience` epochs, so a
    # single bad/unstable epoch doesn't force early stopping at a mediocre
    # optimum -- training can recover at a smaller step size instead.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor, patience=args.scheduler_patience
    )

    # ---- Training loop with early stopping on val loss ----
    print("\nStarting training...")
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    best_state_dict = None
    history = []

    os.makedirs(args.models_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_labels, train_probs = run_epoch(model, train_loader, criterion, optimizer, device)
        train_acc = float(((train_probs >= 0.5).astype(int) == train_labels.astype(int)).mean())

        val_loss, val_labels, val_probs = run_epoch(model, val_loader, criterion, optimizer=None, device=device)
        val_acc = float(((val_probs >= 0.5).astype(int) == val_labels.astype(int)).mean())

        # Step the scheduler on val loss BEFORE printing, so the logged LR
        # reflects any decay that just triggered from this epoch's result.
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{epoch_time:.1f}s"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"\nEarly stopping: no val_loss improvement for {args.patience} epochs.")
                break

    if best_state_dict is None:
        print("WARNING: no improving epoch was recorded; saving final epoch weights instead.")
        best_state_dict = model.state_dict()

    model.load_state_dict(best_state_dict)

    # ---- Final evaluation on test set (only once, at the end) ----
    print("\nEvaluating best model on the held-out test set...")
    test_loss, test_labels, test_probs = run_epoch(model, test_loader, criterion, optimizer=None, device=device)
    test_metrics = compute_binary_metrics(test_labels, test_probs)
    test_metrics["loss"] = float(test_loss)
    print_metrics("test", test_metrics)

    # ---- Save model weights ----
    model_path = os.path.join(args.models_dir, "clone_detector.pt")
    torch.save(best_state_dict, model_path)
    print(f"\nModel weights saved -> {model_path}")

    # ---- Save config + metrics ----
    config = {
        "model": {
            "architecture": "CloneDetectorCNN",
            "dropout": args.dropout,
        },
        "features": {
            "type": "log_mel_spectrogram",
            "sample_rate": TARGET_SAMPLE_RATE,
            "target_duration_seconds": TARGET_DURATION_SECONDS,
            "target_length_samples": TARGET_LENGTH,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "feature_shape": list(FEATURE_SHAPE),
            "n_frames": N_FRAMES,
        },
        "labels": {"bonafide": 0, "spoof": 1},
        "training": {
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "max_epochs": args.epochs,
            "early_stopping_patience": args.patience,
            "lr_scheduler": {
                "type": "ReduceLROnPlateau",
                "factor": args.scheduler_factor,
                "patience": args.scheduler_patience,
            },
            "epochs_trained": len(history),
            "best_val_loss": best_val_loss,
        },
        "dataset": {
            "splits_csv": args.splits_csv,
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
        },
        "test_metrics": test_metrics,
        "training_history": history,
    }

    config_path = os.path.join(args.models_dir, "model_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config + metrics saved -> {config_path}")


if __name__ == "__main__":
    main()