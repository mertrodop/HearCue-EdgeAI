"""Confusion matrix plotting with Keras model."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from hearcue.utils.constants import AUDIO, MODEL
from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.utils.helpers import load_audio

def confusion_matrix(y_true, y_pred):
    labels = list(MODEL.class_labels)
    matrix = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        matrix[t, p] += 1
    return matrix

def plot_confusion_matrix(y_true, y_pred, normalize: bool = False, title: str | None = None) -> plt.Figure:
    matrix = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
            matrix = np.nan_to_num(matrix)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(matrix, annot=True, fmt=".2f" if normalize else "d", cmap="Blues", ax=ax,
                xticklabels=MODEL.class_labels, yticklabels=MODEL.class_labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title or f"Accuracy: {np.mean(np.array(y_true) == np.array(y_pred)):.1%}")
    fig.tight_layout()
    return fig

def load_validation_batch(data_dir: Path):
    """
    Loads validation files, forces them to fixed length, computes spectrograms,
    and applies normalization.
    """
    features = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(MODEL.class_labels)}

    # Define the exact input length your model expects (e.g., 2 seconds)
    target_samples = AUDIO.sample_rate * 2

    print(f"Loading audio files from {data_dir}...")

    for label in MODEL.class_labels:
        class_dir = data_dir / label
        if not class_dir.exists():
            continue

        for wav_path in class_dir.glob("*.wav"):
            waveform, _ = load_audio(wav_path, AUDIO.sample_rate)

            # Force exact length (pad or truncate)
            if len(waveform) < target_samples:
                pad_width = target_samples - len(waveform)
                waveform = np.pad(waveform, (0, pad_width), mode="constant")
            elif len(waveform) > target_samples:
                waveform = waveform[:target_samples]

            spec = log_mel_spectrogram(waveform)

            # Defensive: ensure 2D spec
            if spec.ndim != 2:
                continue

            features.append(spec)
            labels.append(label_map[label])

    if not features:
        raise RuntimeError("No validation samples found. Check paths!")

    # Stack now that shapes are consistent
    X = np.array(features, dtype=np.float32)

    # Normalize (standardize)
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-6)

    # Add Channel Dimension
    X = X[..., np.newaxis]

    return X, np.array(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/processed/val", help="Path to validation folder")
    parser.add_argument("--model_path", default="hearcue/model/hearcue_cnn.keras", help="Path to .keras model")
    parser.add_argument("--normalize", action="store_true", help="Show percentages instead of counts")
    parser.add_argument("--save", default="confusion_matrix.png")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    model_path = Path(args.model_path)

    # 1. Load Data (Pre-normalized)
    print(f"Loading data from {data_path}...")
    X_val, y_true = load_validation_batch(data_path)
    print(f"Data Loaded. Shape: {X_val.shape}")

    # 2. Load Model
    print(f"Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # 3. Predict
    print("Running predictions...")
    y_pred_probs = model.predict(X_val, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Plot
    fig = plot_confusion_matrix(y_true, y_pred, normalize=args.normalize)
    fig.savefig(args.save, dpi=200)
    print(f"Success! Saved matrix to: {args.save}")
    
    # Show plot
    plt.show()
