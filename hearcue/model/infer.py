"""Run quantized inference using the HearCue tiny CNN."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # Lazy guard to provide helpful error when TF missing.
    import tensorflow as tf
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "tensorflow is required for hearcue.model.infer. Install tensorflow-cpu>=2.12."
    ) from exc

from hearcue.utils.constants import MODEL
from hearcue.utils.helpers import softmax


@dataclass
class InferenceResult:
    label: str
    confidence: float
    logits: np.ndarray


class TFLiteAudioClassifier:
    def __init__(self, model_path: str = MODEL.model_path) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]

    def _quantize(self, tensor: np.ndarray) -> np.ndarray:
        if self.input_detail["dtype"] == np.float32:
            return tensor.astype(np.float32)
        scale, zero_point = self.input_detail["quantization"]
        return (tensor / scale + zero_point).astype(self.input_detail["dtype"])

    def _dequantize(self, tensor: np.ndarray) -> np.ndarray:
        if self.output_detail["dtype"] == np.float32:
            return tensor.astype(np.float32)
        scale, zero_point = self.output_detail["quantization"]
        return (tensor.astype(np.float32) - zero_point) * scale

    def predict_proba(self, log_mel: np.ndarray) -> np.ndarray:
        input_shape = self.input_detail["shape"]
        target_frames, target_mels = input_shape[1], input_shape[2]
        data = log_mel
        if data.shape[0] < target_frames:
            pad = np.zeros((target_frames - data.shape[0], data.shape[1]), dtype=np.float32)
            data = np.concatenate((data, pad), axis=0)
        elif data.shape[0] > target_frames:
            data = data[-target_frames:]
        if data.shape[1] != target_mels:
            data = data[:, :target_mels]
        input_data = data[np.newaxis, ..., np.newaxis].astype(np.float32)
        quantized = self._quantize(input_data)
        self.interpreter.set_tensor(self.input_detail["index"], quantized)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_detail["index"])
        dequant = self._dequantize(output[0])
        return softmax(dequant)

    def classify(self, log_mel: np.ndarray) -> InferenceResult:
        probs = self.predict_proba(log_mel)
        idx = int(np.argmax(probs))
        return InferenceResult(label=MODEL.class_labels[idx], confidence=float(probs[idx]), logits=probs)
