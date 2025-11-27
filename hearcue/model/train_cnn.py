"""Train the HearCue tiny CNN using log-mel features."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf

from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import load_audio


def load_dataset(data_dir: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    features = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(MODEL.class_labels)}
    for label in MODEL.class_labels:
        class_dir = data_dir / label
        if not class_dir.exists():
            continue
        for wav_path in class_dir.glob("*.wav"):
            waveform, _ = load_audio(wav_path, AUDIO.sample_rate)
            spec = log_mel_spectrogram(waveform)
            features.append(spec)
            labels.append(label_map[label])
    if not features:
        raise RuntimeError("No training samples found. Populate data/processed/<label>.")
    X = np.array(features, dtype=np.float32)
    y = tf.keras.utils.to_categorical(labels, num_classes=len(MODEL.class_labels))
    X = X[..., np.newaxis]
    return X, y


def build_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(8, (3, 3), activation="relu", input_shape=input_shape, padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(len(MODEL.class_labels), activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train(data_dir: str | Path, epochs: int, batch_size: int, output_dir: str | Path) -> Path:
    X, y = load_dataset(data_dir)
    model = build_model(X.shape[1:])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.1)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "hearcue_cnn.h5"
    model.save(model_path)
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HearCue CNN.")
    parser.add_argument("--data_dir", default="data/processed", help="Directory with class subfolders of WAV files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="hearcue/model", help="Where to save the trained model")
    args = parser.parse_args()
    model_path = train(args.data_dir, args.epochs, args.batch_size, args.output_dir)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
