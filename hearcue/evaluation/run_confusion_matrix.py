from __future__ import annotations

import argparse
from pathlib import Path
from hearcue.utils.constants import MODEL
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import tensorflow as tf

from hearcue.audio.logmelspec import log_mel_spectrogram


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]  # mono
    return x, sr


def confusion_matrix(y_true: list[int], y_pred: list[int], n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .h5 model")
    parser.add_argument("--data_dir", required=True, help=".../val/<class>/*.wav")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--save", default="cm.png")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    # class labels = folder names (sorted for stable mapping)
    class_labels = list(MODEL.class_labels)
    label_to_idx = {l: i for i, l in enumerate(class_labels)}

    model = tf.keras.models.load_model(args.model)

    y_true: list[int] = []
    y_pred: list[int] = []

    for cls in class_labels:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        for wav in cls_dir.glob("*.wav"):
            audio, sr = load_wav(wav)

            # Your pipeline: waveform -> log-mel (model input)
            feat = log_mel_spectrogram(audio, sr)  # expected shape: (T, M)
            x = feat[np.newaxis, ..., np.newaxis]  # (1, T, M, 1)

            probs = model.predict(x, verbose=0)[0]
            pred = int(np.argmax(probs))

            y_true.append(label_to_idx[cls])
            y_pred.append(pred)

    if not y_true:
        raise RuntimeError("No validation samples found. Check --data_dir path.")

    cm = confusion_matrix(y_true, y_pred, len(class_labels))
    # Print per-class precision / recall / F1
    print("\nPer-class metrics:")
    for i, cls in enumerate(class_labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        print(f"{cls:10s}  P={precision:.2f}  R={recall:.2f}  F1={f1:.2f}  (n={cm[i,:].sum()})")

    if args.normalize:
        with np.errstate(divide="ignore", invalid="ignore"):
            cm = cm / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix" + (" (Normalized)" if args.normalize else ""))
    plt.tight_layout()
    plt.savefig(args.save, dpi=200)
    print(f"Saved: {args.save}")
    ##plt.show()


if __name__ == "__main__":
    main()
