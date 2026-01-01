import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf


def export_saved_model(model_path: Path, export_dir: Path) -> None:
    model = tf.keras.models.load_model(model_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    model.export(str(export_dir))
    print(f"SavedModel exported to {export_dir}")


def convert_to_tflite(export_dir: Path, tflite_path: Path) -> None:
    converter = tf.lite.TFLiteConverter.from_saved_model(str(export_dir))
    converter.optimizations = []  # keep float32
    tflite_model = converter.convert()
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_path.write_bytes(tflite_model)
    print(f"Wrote {tflite_path}")


def sanity_check(tflite_path: Path) -> None:
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    print("input:", inp["shape"], inp["dtype"])
    print("output:", out["shape"], out["dtype"])

    x = np.random.randn(*inp["shape"]).astype(np.float32)
    interpreter.set_tensor(inp["index"], x)
    interpreter.invoke()
    y = interpreter.get_tensor(out["index"])[0]
    print(
        "sum(probs)=",
        float(np.sum(y)),
        "top=",
        int(np.argmax(y)),
        "maxprob=",
        float(np.max(y)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Keras model to TFLite.")
    parser.add_argument("--model_path", default="hearcue/model/hearcue_cnn_best.keras")
    parser.add_argument("--export_dir", default="export_savedmodel")
    parser.add_argument("--tflite_path", default="hearcue/model/hearcue_fp32.tflite")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    export_dir = Path(args.export_dir)
    tflite_path = Path(args.tflite_path)

    export_saved_model(model_path, export_dir)
    convert_to_tflite(export_dir, tflite_path)
    sanity_check(tflite_path)


if __name__ == "__main__":
    main()
