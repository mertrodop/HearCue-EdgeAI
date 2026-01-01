import time
import numpy as np
import sounddevice as sd
import tensorflow as tf

from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import normalize_signal
from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.audio.ring_buffer import RingBuffer

# Dataset-level normalization (from training stats)
TRAIN_MEAN = -10.733114242553711
TRAIN_STD = 5.043337821960449

# ---------------- config ----------------
MODEL_PATH = "hearcue/model/hearcue_cnn_best.keras"# <-- your trained model file
TFLITE_MODEL_PATH = "hearcue/model/hearcue_fp32.tflite"  # optional TFLite runtime
PRINT_HZ = 4.0                         # how often to print
FRAMES_EXPECTED = 197                  # matches training input
WINDOW_SAMPLES = AUDIO.frame_length + AUDIO.hop_length * (FRAMES_EXPECTED - 1)  # 31872
NORMALIZE_PER_WINDOW = False           # disable per-window norm to keep raw scale
RMS_SILENCE = 0.03                     # below this, force "other" and skip triggering

# Confidence + smoothing
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

# ring buffer must be >= window; 2x for safety
rb = RingBuffer(size=max(getattr(AUDIO, "ring_buffer_size", 8192), WINDOW_SAMPLES * 2))

# Prefer TFLite if available
_tflite_interpreter = None
_tflite_inp = None
_tflite_out = None
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
    # Fallback to Keras
    model = tf.keras.models.load_model(MODEL_PATH)
    probs_fn = lambda x: model.predict(x, verbose=0)[0]
    print(f"Loaded Keras model: {MODEL_PATH}")

last_print = 0.0

def callback(indata, frames, time_info, status):
    global last_print
    if status:
        # underrun/overrun info
        print(status)

    chunk = indata[:, 0].astype(np.float32)
    rb.write(chunk)

    # Read fixed window; if not enough data, skip
    wave = rb.read(WINDOW_SAMPLES)
    if wave is None:
        return

    # Silence gate: skip processing very quiet windows
    rms = float(np.sqrt(np.mean(wave * wave)))
    if rms < RMS_SILENCE:
        # Force "other" when below silence threshold
        probs = np.zeros(len(labels), dtype=np.float32)
        other_idx = labels.index("other")
        probs[other_idx] = 1.0
        now = time.time()
        if now - last_print >= (1.0 / PRINT_HZ):
            last_print = now
            print(f"other=1.00 | margin=1.00 | rms={rms:.3f} | crest=N/A | spec min/max/std: n/a")
        return

    # Normalize waveform amplitude
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

    # Apply margin and class thresholds
    sorted_idx = np.argsort(probs)
    top_i = int(sorted_idx[-1])
    top2_i = int(sorted_idx[-2]) if len(sorted_idx) > 1 else top_i
    margin = float(probs[top_i] - probs[top2_i])
    top_label = labels[top_i]
    top_conf = float(probs[top_i])

    if margin < MIN_MARGIN or top_conf < CLASS_THRESHOLDS.get(top_label, 0.0):
        label = None
    else:
        label = top_label

    # Temporal smoothing: K of N
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
        if triggered:
            print(
                f"TRIGGER {triggered}: conf={top_conf:.2f} | margin={margin:.2f} | rms={rms:.3f} "
                f"| spec min/max/std: {mn:.2f} {mx:.2f} {sdv:.2f}"
            )
        else:
            print(
                f"{top_label}={top_conf:.2f} (suppressed) | margin={margin:.2f} | rms={rms:.3f} "
                f"| spec min/max/std: {mn:.2f} {mx:.2f} {sdv:.2f}"
            )

print("🎙️  Listening... Ctrl+C to stop")
print("Labels:", labels)

with sd.InputStream(
    channels=1,
    samplerate=AUDIO.sample_rate,
    blocksize=AUDIO.chunk_size,
    dtype="float32",
    callback=callback,
):
    while True:
        time.sleep(0.05)
