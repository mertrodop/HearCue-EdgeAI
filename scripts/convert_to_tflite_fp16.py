import argparse
from pathlib import Path

import tensorflow as tf


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Keras model to FP16 TFLite.")
    parser.add_argument("--model_path", default="hearcue/model/hearcue_cnn_best.keras")
    parser.add_argument("--output_path", default="hearcue/model/hearcue_cnn_fp16.tflite")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
