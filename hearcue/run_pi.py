from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd
import tensorflow as tf

from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import normalize_signal
from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.audio.ring_buffer import RingBuffer
from hearcue.system.symbolic_ui import SymbolicUI

# Dataset-level normalization (from training stats)
TRAIN_MEAN = -10.733114242553711
TRAIN_STD = 5.043337821960449

# Runtime config (aligned with realtime_pc_debug.py)
FRAMES_EXPECTED = 197
WINDOW_SAMPLES = AUDIO.frame_length + AUDIO.hop_length * (FRAMES_EXPECTED - 1)
RMS_SILENCE = 0.03
MIN_MARGIN = 0.25
CLASS_THRESHOLDS = {
    "dog": 0.60,
    "car": 0.65,
    "speech": 0.70,
    "ringtone": 0.75,
    "alarm": 0.80,
    "other": 0.0,
}
SMOOTH_WINDOW = 5
REQUIRED_HITS = {
    "dog": 3,
    "car": 3,
    "speech": 4,
    "ringtone": 4,
    "alarm": 4,
}
PRINT_HZ = 4.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live inference with debug-style UI.")
    parser.add_argument(
        "--model_path",
        default="hearcue/model/hearcue_cnn_fp16.tflite",
        help="Path to TFLite (or Keras) model",
    )
    args = parser.parse_args()

    labels = list(MODEL.class_labels)
    history: list[str | None] = []
    rb = RingBuffer(size=max(getattr(AUDIO, "ring_buffer_size", 8192), WINDOW_SAMPLES * 2))
    ui = SymbolicUI()

    # Prefer TFLite; fallback to Keras
    try:
        interpreter = tf.lite.Interpreter(model_path=args.model_path)
        interpreter.allocate_tensors()
        _inp = interpreter.get_input_details()[0]
        _out = interpreter.get_output_details()[0]

        def probs_fn(x: np.ndarray) -> np.ndarray:
            interpreter.set_tensor(_inp["index"], x.astype(np.float32))
            interpreter.invoke()
            return interpreter.get_tensor(_out["index"])[0]

        print(f"Loaded TFLite model: {args.model_path}")
    except (OSError, ValueError):
        model = tf.keras.models.load_model(args.model_path)

        def probs_fn(x: np.ndarray) -> np.ndarray:
            return model.predict(x, verbose=0)[0]

        print(f"Loaded Keras model: {args.model_path}")

    last_print = 0.0
    print("Listening... Ctrl+C to stop")

    def callback(indata, frames, time_info, status):
        nonlocal last_print, history
        if status:
            print(status)
        chunk = np.asarray(indata[:, 0], dtype=np.float32)
        rb.write(chunk)

        wave = rb.read(WINDOW_SAMPLES)
        if wave is None:
            return

        rms = float(np.sqrt(np.mean(wave * wave)))
        if rms < RMS_SILENCE:
            now = time.time()
            if now - last_print >= 1.0 / PRINT_HZ:
                last_print = now
                ui.show(
                    top_label="other",
                    top_conf=1.0,
                    margin=1.0,
                    rms=rms,
                    spec_min=None,
                    spec_max=None,
                    spec_std=None,
                    triggered=None,
                )
            return

        wave = normalize_signal(wave.astype(np.float32))
        spec = log_mel_spectrogram(
            wave,
            sample_rate=AUDIO.sample_rate,
            frame_length=AUDIO.frame_length,
            hop_length=AUDIO.hop_length,
            n_mels=AUDIO.n_mels,
            fmin=AUDIO.fmin,
            fmax=AUDIO.fmax,
        )
        spec = (spec - TRAIN_MEAN) / (TRAIN_STD + 1e-6)

        x = spec[np.newaxis, ..., np.newaxis].astype(np.float32)
        if x.shape[1] != FRAMES_EXPECTED or x.shape[2] != AUDIO.n_mels:
            print("BAD SHAPE:", x.shape)
            return

        probs = probs_fn(x)

        sorted_idx = np.argsort(probs)
        top_i = int(sorted_idx[-1])
        top2_i = int(sorted_idx[-2]) if len(sorted_idx) > 1 else top_i
        top_label = labels[top_i]
        top_conf = float(probs[top_i])
        margin = float(probs[top_i] - probs[top2_i])

        # threshold + margin gate
        if margin < MIN_MARGIN or top_conf < CLASS_THRESHOLDS.get(top_label, 0.0):
            label = None
        else:
            label = top_label

        # smoothing
        history.append(label)
        if len(history) > SMOOTH_WINDOW:
            history.pop(0)
        triggered = None
        if label is not None:
            hits = sum(1 for l in history if l == label)
            need = REQUIRED_HITS.get(label, SMOOTH_WINDOW)
            if hits >= need:
                triggered = label

        now = time.time()
        if now - last_print >= 1.0 / PRINT_HZ:
            last_print = now
            mn, mx, sdv = float(spec.min()), float(spec.max()), float(spec.std())
            ui.show(
                top_label=top_label,
                top_conf=top_conf,
                margin=margin,
                rms=rms,
                spec_min=mn,
                spec_max=mx,
                spec_std=sdv,
                triggered=triggered,
            )

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
