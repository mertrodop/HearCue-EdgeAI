"""Convenience wrapper for training and quantization."""
from __future__ import annotations

import argparse

from hearcue.model.quantize import quantize
from hearcue.model.train_cnn import train
from hearcue.utils.constants import MODEL


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and quantize the HearCue CNN.")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_dir", default="hearcue/model")
    parser.add_argument("--rep_data", default="data/processed")
    parser.add_argument("--output", default=MODEL.model_path)
    args = parser.parse_args()
    model_path = train(args.data_dir, args.epochs, args.batch_size, args.model_dir)
    tflite_path = quantize(model_path, args.output, args.rep_data)
    print(f"Quantized model stored at {tflite_path}")


if __name__ == "__main__":
    main()
