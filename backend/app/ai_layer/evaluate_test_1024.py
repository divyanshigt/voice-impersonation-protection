import sys
from pathlib import Path

# backend directory
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from app.ai_layer.mel_features import extract_mel_features
from app.ai_layer.model_architecture import CloneDetectorCNN

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

TEST_DIR = BACKEND_DIR / "asvspoof_test_1024"
LABELS_CSV = TEST_DIR / "test_1024_labels.csv"
MODEL_PATH = BACKEND_DIR / "models" / "clone_detector.pt"


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# --------------------------------------------------
# LOAD LABELS
# --------------------------------------------------

df = pd.read_csv(LABELS_CSV)

print("\nLabels loaded:", len(df))
print(df["label"].value_counts())


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = CloneDetectorCNN()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("\nModel loaded successfully.")
print("Model:", MODEL_PATH)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

y_true = []
y_pred = []
y_prob = []

total = len(df)

print("\nExtracting features and predicting...\n")

with torch.no_grad():

    for i, row in df.iterrows():

        utt_id = row["utt_id"]
        label = row["label"]

        audio_path = TEST_DIR / f"{utt_id}.flac"

        if not audio_path.exists():
            print("Missing:", audio_path)
            continue

        # Extract Mel features
        feature = extract_mel_features(audio_path)

        # Shape: (64, 251)
        x = torch.tensor(
            feature,
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)

        x = x.to(device)

        # Model prediction
        output = model(x)

        # Convert logits to probability
        probability = torch.sigmoid(output).item()

        prediction = 1 if probability >= 0.5 else 0

        y_true.append(1 if label == "spoof" else 0)
        y_pred.append(prediction)
        y_prob.append(probability)

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{total}")


# --------------------------------------------------
# METRICS
# --------------------------------------------------

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

cm = confusion_matrix(y_true, y_pred)


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n")
print("=" * 50)
print("TEST 1024 - FINAL EVALUATION")
print("=" * 50)

print(f"Samples:    {len(y_true)}")
print(f"Accuracy:   {accuracy:.4f}")
print(f"Precision:  {precision:.4f}")
print(f"Recall:     {recall:.4f}")
print(f"F1 Score:   {f1:.4f}")
print(f"ROC-AUC:    {roc_auc:.4f}")

print("\nConfusion Matrix [[TN, FP], [FN, TP]]:")
print(cm)

print("=" * 50)