from __future__ import annotations

import argparse
import numpy as np
import sounddevice as sd
import tensorflow as tf

from hearcue.model.infer import TFLiteAudioClassifier
from hearcue.system.device_controller import DeviceController
from hearcue.system.symbolic_ui import SymbolicUI
from hearcue.utils.constants import AUDIO


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeviceController with mic input (debug-aligned).")
    parser.add_argument(
        "--model_path",
        default="hearcue/model/hearcue_cnn_fp16.tflite",
        help="Path to TFLite (or Keras) model",
    )
    args = parser.parse_args()

    # Prefer TFLite; fallback to Keras
    try:
        _ = tf.lite.Interpreter(model_path=args.model_path)
        classifier = TFLiteAudioClassifier(model_path=args.model_path)
    except Exception:
        classifier = TFLiteAudioClassifier(model_path=args.model_path)

    dc = DeviceController(classifier=classifier, ui=SymbolicUI())

    print("Listening... Ctrl+C to stop")

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        chunk = np.asarray(indata[:, 0], dtype=np.float32)
        dc.process_chunk(chunk)

    with sd.InputStream(
        channels=1,
        samplerate=AUDIO.sample_rate,
        blocksize=AUDIO.chunk_size,
        dtype="float32",
        callback=callback,
    ):
        while True:
            sd.sleep(100)


if __name__ == "__main__":
    main()
