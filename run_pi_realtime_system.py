import time
import queue
import numpy as np
import sounddevice as sd
import tensorflow as tf

from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import normalize_signal
from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.audio.ring_buffer import RingBuffer
from hearcue.system.symbolic_ui import SymbolicUI

try:
    from hearcue.system.haptic import HapticController
except Exception:
    HapticController = None

# Dataset-level normalization (from training stats)
TRAIN_MEAN = -10.733114242553711
TRAIN_STD = 5.043337821960449

# ---------------- config ----------------
MODEL_PATH = "hearcue/model/hearcue_cnn_best.keras"
TFLITE_MODEL_PATH = "hearcue/model/hearcue_fp32.tflite"
PRINT_HZ = 8.0

MIC_DEVICE = 0
MIC_SR = 48000

FRAMES_EXPECTED = 197
WINDOW_SAMPLES = AUDIO.frame_length + AUDIO.hop_length * (FRAMES_EXPECTED - 1)

RMS_SILENCE = 0.005

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
# ---------------------------------------

labels = list(MODEL.class_labels)
history = []

rb = RingBuffer(size=max(getattr(AUDIO, "ring_buffer_size", 8192), WINDOW_SAMPLES * 2))

# --- model runtime ---
probs_fn = None
try:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors()
    _tflite_inp = interpreter.get_input_details()[0]
    _tflite_out = interpreter.get_output_details()[0]

    def _tflite_predict(x: np.ndarray) -> np.ndarray:
        interpreter.set_tensor(_tflite_inp["index"], x.astype(np.float32))
        interpreter.invoke()
        return interpreter.get_tensor(_tflite_out["index"])[0]

    probs_fn = _tflite_predict
    print(f"Loaded TFLite model: {TFLITE_MODEL_PATH}")
except (OSError, ValueError):
    model = tf.keras.models.load_model(MODEL_PATH)
    probs_fn = lambda x: model.predict(x, verbose=0)[0]
    print(f"Loaded Keras model: {MODEL_PATH}")

# --- event queue for main thread UI/haptics ---
evt_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)

last_print = 0.0
last_rms_debug = 0.0
COOLDOWN = {"car": 1.2, "alarm": 3.0, "ringtone": 2.0, "speech": 0.8, "dog": 2.0}
last_trigger: dict[str, float] = {}


def callback(indata, frames, time_info, status):
    global last_print, last_trigger, last_rms_debug

    if status:
        print(status)

    ch0_48k = indata[:, 0].astype(np.float32) / (2**31)
    # If channel 0 is quiet on the device, try channel 1 instead.
    # ch0_48k = indata[:, 1].astype(np.float32) / (2**31)
    ch0_16k = ch0_48k[::3]

    now = time.time()
    if now - last_rms_debug >= 0.5:
        last_rms_debug = now
        rms = float(np.sqrt(np.mean(ch0_16k * ch0_16k)))
        print("RMS16k:", rms)

    rb.write(ch0_16k)

    wave = rb.read(WINDOW_SAMPLES)
    if wave is None:
        return

    rms = float(np.sqrt(np.mean(wave * wave)))
    if rms < RMS_SILENCE:
        now = time.time()
        if now - last_print >= (1.0 / PRINT_HZ):
            last_print = now
            ui_label = "idle"
            try:
                evt_q.put_nowait(
                    {
                        "top_label": ui_label,
                        "top_conf": 1.0,
                        "margin": 1.0,
                        "rms": rms,
                        "spec_min": None,
                        "spec_max": None,
                        "spec_std": None,
                        "triggered": None,
                    }
                )
            except queue.Full:
                pass
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
    margin = float(probs[top_i] - probs[top2_i])
    top_label = labels[top_i]
    top_conf = float(probs[top_i])

    # --- Apply margin and class thresholds (DEBUG POLICY) ---
    if margin < MIN_MARGIN or top_conf < CLASS_THRESHOLDS.get(top_label, 0.0):
        label = None
    else:
        label = top_label
    if label == "other":
        label = None

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
    if now - last_print >= (1.0 / PRINT_HZ):
        last_print = now
        mn, mx, sdv = float(spec.min()), float(spec.max()), float(spec.std())
        ui_label = "idle" if top_label == "other" else top_label
        try:
            evt_q.put_nowait(
                {
                    "top_label": ui_label,
                    "top_conf": top_conf,
                    "margin": margin,
                    "rms": rms,
                    "spec_min": mn,
                    "spec_max": mx,
                    "spec_std": sdv,
                    "triggered": triggered,
                }
            )
        except queue.Full:
            pass

    if triggered is not None and triggered != "other":
        cd = COOLDOWN.get(triggered, 1.0)
        if (now - last_trigger.get(triggered, 0.0)) >= cd:
            last_trigger[triggered] = now
            try:
                evt_q.put_nowait({"haptic_trigger": triggered})
            except queue.Full:
                pass


def main():
    print("🎙️  Listening... Ctrl+C to stop")
    print("Labels:", labels)

    ui = SymbolicUI()
    haptic = None
    if HapticController is not None:
        try:
            haptic = HapticController()
            print("Haptics: enabled")
        except Exception as e:
            print("Haptics: disabled:", e)

    last_ui = {
        "top_label": "other",
        "top_conf": 0.0,
        "margin": 0.0,
        "rms": 0.0,
        "spec_min": None,
        "spec_max": None,
        "spec_std": None,
        "triggered": None,
    }

    with sd.InputStream(
        device=MIC_DEVICE,
        channels=2,
        samplerate=MIC_SR,
        dtype="int32",
        blocksize=AUDIO.chunk_size * 3,  # keeps the same time span as 16k chunks
        callback=callback,
    ):
        while True:
            try:
                evt = evt_q.get(timeout=0.05)
            except queue.Empty:
                evt = None

            # Idle redraw to keep UI responsive
            if evt is None:
                ui.show(
                    top_label="idle" if last_ui["top_label"] == "other" else last_ui["top_label"],
                    top_conf=last_ui["top_conf"],
                    margin=last_ui["margin"],
                    rms=last_ui["rms"],
                    spec_min=last_ui["spec_min"],
                    spec_max=last_ui["spec_max"],
                    spec_std=last_ui["spec_std"],
                    triggered=last_ui["triggered"],
                )
                time.sleep(0.005)
                continue

            if evt:
                if "haptic_trigger" in evt:
                    label = evt["haptic_trigger"]
                    if haptic:
                        haptic.pulse(label)
                    ui.note_haptic(label)
                else:
                    last_ui = evt
                    ui.show(
                        top_label="idle" if evt["top_label"] == "other" else evt["top_label"],
                        top_conf=evt["top_conf"],
                        margin=evt["margin"],
                        rms=evt["rms"],
                        spec_min=evt["spec_min"],
                        spec_max=evt["spec_max"],
                        spec_std=evt["spec_std"],
                        triggered=evt["triggered"],
                    )

            time.sleep(0.005)


if __name__ == "__main__":
    main()
