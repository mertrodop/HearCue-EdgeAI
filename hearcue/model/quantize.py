"""Quantize trained model to a fully-integer TFLite file."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import load_audio


def representative_dataset(dataset_dir: str | Path, sample_limit: int = 20) -> Iterable[np.ndarray]:
    dataset_dir = Path(dataset_dir)
    count = 0
    for wav in dataset_dir.rglob("*.wav"):
        waveform, _ = load_audio(wav, AUDIO.sample_rate)
        spec = log_mel_spectrogram(waveform)
        spec = spec[np.newaxis, ..., np.newaxis].astype(np.float32)
        yield spec
        count += 1
        if count >= sample_limit:
            break


def quantize(model_path: str | Path, output_path: str | Path, rep_data_dir: str | Path) -> Path:
    model_path = Path(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(tf.keras.models.load_model(model_path))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(rep_data_dir)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite_model = converter.convert()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize HearCue CNN to TFLite.")
    parser.add_argument("--model_path", default="hearcue/model/hearcue_cnn.h5")
    parser.add_argument("--rep_data", default="data/processed")
    parser.add_argument("--output", default=MODEL.model_path)
    args = parser.parse_args()
    output = quantize(args.model_path, args.output, args.rep_data)
    print(f"Quantized TFLite model saved to {output}")


if __name__ == "__main__":
    main()
