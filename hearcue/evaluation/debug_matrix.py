"""
debug_matrix.py
Run this to generate the correct confusion matrix.
"""
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Imports from your project
from hearcue.utils.constants import AUDIO, MODEL
from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.utils.helpers import load_audio

def load_validation_batch_normalized(data_dir: Path):
    print(f"Loading files from: {data_dir}")
    features = []
    labels = []
    label_map = {label: idx for idx, label in enumerate(MODEL.class_labels)}
    
    # 1. Load all files
    file_count = 0
    for label in MODEL.class_labels:
        class_dir = data_dir / label
        if not class_dir.exists(): continue
        
        for wav_path in class_dir.glob("*.wav"):
            waveform, _ = load_audio(wav_path, AUDIO.sample_rate)
            spec = log_mel_spectrogram(waveform)
            features.append(spec)
            labels.append(label_map[label])
            file_count += 1
            
    if not features:
        raise RuntimeError("No .wav files found! Check your path.")

    # 2. Convert to Array
    X = np.array(features, dtype=np.float32)
    
    # 3. DEBUG: Print stats before normalization
    print(f"Stats BEFORE Norm: Mean={X.mean():.4f}, Std={X.std():.4f}, Min={X.min():.4f}, Max={X.max():.4f}")
    
    # 4. CRITICAL: Normalize exactly like Training
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-6)
    
    # 5. DEBUG: Print stats after normalization
    print(f"Stats AFTER Norm:  Mean={X.mean():.4f}, Std={X.std():.4f}")
    
    # 6. Add Channel Dimension
    X = X[..., np.newaxis]
    
    return X, np.array(labels)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--save", default="confusion_matrix_fixed.png")
    args = parser.parse_args()

    # 1. Load Data
    X_val, y_true = load_validation_batch_normalized(Path(args.data_dir))

    # 2. Load Model
    print(f"Loading model: {args.model_path}")
    model = tf.keras.models.load_model(args.model_path)

    # 3. Predict
    print("Predicting...")
    y_pred_probs = model.predict(X_val, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 4. Calculate Accuracy
    acc = np.mean(y_pred == y_true)
    print(f"\nFINAL ACCURACY: {acc:.2%}") # <--- This should match your logs (~60%)

    # 5. Plot
    cm = tf.math.confusion_matrix(y_true, y_pred).numpy()
    # Normalize for plot
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=MODEL.class_labels, yticklabels=MODEL.class_labels)
    plt.title(f"Accuracy: {acc:.1%}")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(args.save)
    print(f"Saved matrix to {args.save}")

if __name__ == "__main__":
    main()