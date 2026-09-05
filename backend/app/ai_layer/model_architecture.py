"""
CNN architecture for the voice clone / speech deepfake binary classifier
(Phase 2C).

Input:
    Log-mel spectrogram from app.ai_layer.mel_features, shape
    (batch, 1, N_MELS, N_FRAMES) = (batch, 1, 64, 251) with the fixed
    feature parameters used elsewhere in Phase 2C.

Output:
    Single logit per sample, shape (batch, 1).
    Apply sigmoid() at inference time to get P(spoof) in [0, 1].
    Label convention (must match dataset_splits.csv / training script):
        0 = bonafide / real
        1 = spoof / synthetic

Architecture (fixed per Phase 2C spec):
    Conv2D(16, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
    Conv2D(32, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
    Conv2D(64, 3x3) -> BatchNorm -> ReLU -> MaxPool(2x2)
    AdaptiveAvgPool2D(1x1) -> Flatten
    Dropout(0.3) -> Linear(64 -> 32) -> ReLU
    Dropout(0.3) -> Linear(32 -> 1)

AdaptiveAvgPool2D is used (instead of a fixed-size Flatten after the conv
stack) so the network doesn't hardcode a specific spatial size after the
three pooling layers -- it works for the 64x251 input used here without
needing a manually-computed flatten dimension, and stays robust if the
mel feature shape ever changes slightly.
"""

import os
import sys

import torch
import torch.nn as nn

# Allow running this file directly (python app/ai_layer/model_architecture.py)
# as well as as a module -- needed for the __main__ smoke test below, which
# imports app.ai_layer.mel_features. Requires the backend/ folder (parent
# of app/) on sys.path when run directly, same convention as the other
# ai_layer files (audio_features.py, mel_features.py, train_clone_detector.py).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CloneDetectorCNN(nn.Module):
    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 1 -> 16 channels
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 2: 16 -> 32 channels
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            # Block 3: 32 -> 64 channels
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        # Collapses whatever spatial size remains after the conv/pool
        # stack down to a single value per channel -> output is always
        # (batch, 64, 1, 1) regardless of input H/W, so no manual
        # flatten-dimension arithmetic is needed.
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, 1, N_MELS, N_FRAMES)
        returns: (batch, 1) raw logits (no sigmoid applied -- use
                 BCEWithLogitsLoss for training, torch.sigmoid() for
                 inference probabilities).
        """
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def build_model(dropout: float = 0.3) -> CloneDetectorCNN:
    """Factory function, kept separate from the class for a clean import
    surface in the training script (mirrors the style of the other
    Phase 2B modules, which expose one clear entry-point function)."""
    return CloneDetectorCNN(dropout=dropout)


if __name__ == "__main__":
    # Smoke test: verify the architecture runs end-to-end on the exact
    # shape mel_features.py produces, without touching any real data.
    from app.ai_layer.mel_features import FEATURE_SHAPE  # (64, 251)

    model = build_model()
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch_size = 4
    dummy_input = torch.randn(batch_size, 1, *FEATURE_SHAPE)

    model.eval()
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape:  {tuple(dummy_input.shape)}")
    print(f"Output shape: {tuple(output.shape)}")
    print(f"Total parameters:     {num_params:,}")
    print(f"Trainable parameters: {num_trainable:,}")
    assert output.shape == (batch_size, 1), "Unexpected output shape!"
    print("Smoke test passed.")