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

TFLITE_MODEL_PATH = "hearcue/model/hearcue_cnn_fp16.tflite"

PRINT_HZ = 8.0

TARGET_SR = 16000

FRAMES_EXPECTED = 197
WINDOW_SAMPLES = AUDIO.frame_length + AUDIO.hop_length * (FRAMES_EXPECTED - 1)

RMS_SILENCE = 0.01

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
COOLDOWN = {"car": 1.2, "alarm": 3.0, "ringtone": 2.0, "speech": 0.8, "dog": 2.0}
# ---------------------------------------

labels = list(MODEL.class_labels)
history = []
last_trigger: dict[str, float] = {}

rb = RingBuffer(size=max(getattr(AUDIO, "ring_buffer_size", 8192), WINDOW_SAMPLES * 2))

# --- event queue for main thread UI/haptics ---
evt_q: "queue.Queue[dict]" = queue.Queue(maxsize=200)

# --- audio queue (callback -> main thread) ---
audio_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=50)


def pick_jbl_device(prefer=("JBL", "Quantum", "Stream"), ban=("sysdefault", "default", "pipewire", "pulse")):
    devs = sd.query_devices()
    best = None
    for i, d in enumerate(devs):
        if d.get("max_input_channels", 0) <= 0:
            continue
        name = d.get("name", "")
        lname = name.lower()

        # skip virtual / routing devices
        if any(b in lname for b in ban):
            continue

        score = sum(k.lower() in lname for k in prefer)

        # strongly prefer hardware endpoints
        if "(hw:" in lname:
            score += 10

        if best is None or score > best[0]:
            best = (score, i, name, float(d.get("default_samplerate", 0.0)))

    if best is None:
        # fallback: allow any input device (still print list for debugging)
        raise RuntimeError("Could not find a real hardware input device. Check sd.query_devices().")

    print(f"Using input device: {best[1]} {best[2]} (native SR {best[3]})")
    return best[1], int(best[3])


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear resampler. Good enough for realtime classification."""
    if src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    n = int(round(len(x) * dst_sr / src_sr))
    if n <= 1:
        return np.zeros((0,), dtype=np.float32)
    xp = np.arange(len(x), dtype=np.float32)
    xq = np.linspace(0, len(x) - 1, num=n, dtype=np.float32)
    return np.interp(xq, xp, x).astype(np.float32)


# Pick JBL + native SR (usually 44100)
MIC_DEVICE, MIC_NATIVE_SR = pick_jbl_device()

# --- model runtime ---
probs_fn = None
interpreter = None
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
except (OSError, ValueError) as e:
    print(f"TFLite load failed ({e}); falling back to Keras: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    probs_fn = lambda x: model.predict(x, verbose=0)[0]


def audio_callback(indata, frames, time_info, status):
    # Keep callback minimal to avoid overflows
    if status:
        # printing too often can itself cause overflows; keep it minimal
        # print(status)
        pass

    # JBL: typically int16 mono; but we set dtype explicitly below
    ch = indata[:, 0].astype(np.float32) / 32768.0
    ch16 = resample_linear(ch, MIC_NATIVE_SR, TARGET_SR)

    try:
        audio_q.put_nowait(ch16)
    except queue.Full:
        # drop newest chunk to keep realtime
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

    last_print = 0.0
    last_rms_debug = 0.0

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

    # Use mic native SR to avoid PortAudio resample overhead
    # Blocksize ~50ms at native SR
    blocksize = int(MIC_NATIVE_SR * 0.05)

    with sd.InputStream(
        device=MIC_DEVICE,
        channels=1,
        samplerate=MIC_NATIVE_SR,
        dtype="int16",
        blocksize=blocksize,
        callback=audio_callback,
    ):
        while True:
            # 1) Consume audio chunks (if any) and run inference (main thread)
            processed_any = False
            while True:
                try:
                    ch16 = audio_q.get_nowait()
                except queue.Empty:
                    break

                processed_any = True
                rb.write(ch16)

            if processed_any:
                wave = rb.read(WINDOW_SAMPLES)
                if wave is not None:
                    rms = float(np.sqrt(np.mean(wave * wave)))

                    now = time.time()
                    if now - last_rms_debug >= 0.5:
                        last_rms_debug = now
                        print("RMS16k:", rms)

                    if rms < RMS_SILENCE:
                        # show idle periodically
                        if now - last_print >= (1.0 / PRINT_HZ):
                            last_print = now
                            evt = {
                                "top_label": "idle",
                                "top_conf": 1.0,
                                "margin": 1.0,
                                "rms": rms,
                                "spec_min": None,
                                "spec_max": None,
                                "spec_std": None,
                                "triggered": None,
                            }
                            try:
                                evt_q.put_nowait(evt)
                            except queue.Full:
                                pass
                    else:
                        wave_n = normalize_signal(wave.astype(np.float32))
                        spec = log_mel_spectrogram(
                            wave_n,
                            sample_rate=TARGET_SR,  
                            frame_length=AUDIO.frame_length,
                            hop_length=AUDIO.hop_length,
                            n_mels=AUDIO.n_mels,
                            fmin=AUDIO.fmin,
                            fmax=AUDIO.fmax,
                        )
                        spec = (spec - TRAIN_MEAN) / (TRAIN_STD + 1e-6)

                        x = spec[np.newaxis, ..., np.newaxis].astype(np.float32)
                        if x.shape[1] == FRAMES_EXPECTED and x.shape[2] == AUDIO.n_mels:
                            probs = probs_fn(x)

                            sorted_idx = np.argsort(probs)
                            top_i = int(sorted_idx[-1])
                            top2_i = int(sorted_idx[-2]) if len(sorted_idx) > 1 else top_i
                            margin = float(probs[top_i] - probs[top2_i])
                            top_label = labels[top_i]
                            top_conf = float(probs[top_i])

                            # --- Apply margin and class thresholds ---
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
                                evt = {
                                    "top_label": ui_label,
                                    "top_conf": top_conf,
                                    "margin": margin,
                                    "rms": rms,
                                    "spec_min": mn,
                                    "spec_max": mx,
                                    "spec_std": sdv,
                                    "triggered": triggered,
                                }
                                try:
                                    evt_q.put_nowait(evt)
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

            # 2) UI / haptics updates
            try:
                evt = evt_q.get(timeout=0.05)
            except queue.Empty:
                evt = None

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
    try:
        main()
    finally:
        try:
            from hearcue.system.haptic import HapticController
            haptic = HapticController()
            haptic.off()
        except Exception:
            pass

