"""Train the HearCue tiny CNN using log-mel features."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.utils import class_weight

from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import load_audio, spec_augment


def load_dataset(data_dir: str | Path, augment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(data_dir)
    raw_specs = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(MODEL.class_labels)}

    # Debug: track shapes
    mel_bins_seen = Counter()
    frames_seen = Counter()

    for label in MODEL.class_labels:
        class_dir = data_dir / label
        if not class_dir.exists():
            continue
        for wav_path in class_dir.glob("*.wav"):
            waveform, _ = load_audio(wav_path, AUDIO.sample_rate)
            # Fix waveform length to a consistent duration (~2 seconds)
            target_samples = AUDIO.sample_rate * 2
            if len(waveform) < target_samples:
                pad = np.zeros(target_samples - len(waveform), dtype=np.float32)
                waveform = np.concatenate([waveform, pad])
            elif len(waveform) > target_samples:
                waveform = waveform[-target_samples:]

            spec = log_mel_spectrogram(waveform)

            # Defensive: skip weird outputs
            if not isinstance(spec, np.ndarray) or spec.ndim != 2:
                continue

            if augment and np.random.rand() > 0.5:
                spec = spec_augment(spec)

            raw_specs.append(spec.astype(np.float32, copy=False))
            labels.append(label_map[label])

            frames_seen[spec.shape[0]] += 1
            mel_bins_seen[spec.shape[1]] += 1

    if not raw_specs:
        raise RuntimeError("No training samples found. Populate data/processed/<label>.")

    # Choose a consistent target shape
    target_frames = max(s.shape[0] for s in raw_specs)
    target_mels = max(s.shape[1] for s in raw_specs)

    # Debug prints (helps you see if mels are inconsistent)
    if len(mel_bins_seen) > 1:
        print("WARNING: Multiple mel-bin sizes detected:", dict(mel_bins_seen))
    # print("Frames distribution (top):", frames_seen.most_common(5))

    features = []
    for spec in raw_specs:
        frames, mels = spec.shape

        # Pad/truncate time (frames)
        if frames < target_frames:
            spec = np.pad(spec, ((0, target_frames - frames), (0, 0)), mode="constant")
        elif frames > target_frames:
            spec = spec[-target_frames:, :]

        # Pad/truncate frequency (mel bins)
        if mels < target_mels:
            spec = np.pad(spec, ((0, 0), (0, target_mels - mels)), mode="constant")
        elif mels > target_mels:
            spec = spec[:, :target_mels]

        features.append(spec)

    X = np.stack(features, axis=0).astype(np.float32)
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-6)

    y = tf.keras.utils.to_categorical(labels, num_classes=len(MODEL.class_labels))
    X = X[..., np.newaxis]
    return X, y


def build_model(input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            # Block 1 - Capture low-level features
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=input_shape),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.15),

            # Block 2 - Capture patterns
            tf.keras.layers.Conv2D(64, (3, 3), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.15),

            # Block 3 - Capture complex textures
            tf.keras.layers.Conv2D(128, (3, 3), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.15),

            # Block 4 - Deep features then pool
            tf.keras.layers.Conv2D(256, (3, 3), padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.GlobalAveragePooling2D(),

            # Dense Head - Increased capacity
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(len(MODEL.class_labels), activation="softmax")
        ]
    )
    
    # Reset optimizer to standard LR
    opt = tf.keras.optimizers.Adam(learning_rate=0.001) 
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def train(train_dir: str | Path, val_dir: str | Path | None, epochs: int, batch_size: int, output_dir: str | Path) -> Path:
    X_train, y_train = load_dataset(train_dir, augment=True)
    print(f"DEBUG: X_train shape is: {X_train.shape}")
    model = build_model(X_train.shape[1:])

    # Balanced class weighting to combat class imbalance
    y_ints = np.argmax(y_train, axis=1)
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_ints),
        y=y_ints,
    )
    class_weight_dict = dict(enumerate(class_weights))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "hearcue_cnn_best.keras"),
            monitor="val_accuracy",
            save_best_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    if val_dir is not None:
        X_val, y_val = load_dataset(val_dir, augment=False)
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight_dict,
        )
    else:
        model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            class_weight=class_weight_dict,
        )

    model_path = output_dir / "hearcue_cnn.keras"
    model.save(model_path)
    return model_path



def main() -> None:
    parser = argparse.ArgumentParser(description="Train HearCue CNN.")
    parser.add_argument("--train_dir", default="data/processed/train", help="Train dir with class subfolders")
    parser.add_argument("--val_dir", default="data/processed/val", help="Val dir with class subfolders")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--output_dir", default="hearcue/model", help="Where to save the trained model")
    args = parser.parse_args()
    model_path = train(args.train_dir, args.val_dir, args.epochs, args.batch_size, args.output_dir)
    print(f"Saved trained model to {model_path}")


if __name__ == "__main__":
    main()
