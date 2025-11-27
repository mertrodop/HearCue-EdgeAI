from pathlib import Path

import numpy as np
import pytest

from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.model.infer import TFLiteAudioClassifier
from hearcue.utils.constants import AUDIO, MODEL


def test_tflite_inference_runs():
    pytest.importorskip("tensorflow")
    model_file = Path(MODEL.model_path)
    if not model_file.exists():
        pytest.skip("Quantized model missing")
    classifier = TFLiteAudioClassifier(str(model_file))
    waveform = np.random.randn(AUDIO.frame_length * 20).astype(np.float32)
    features = log_mel_spectrogram(waveform)
    result = classifier.classify(features)
    assert result.label in MODEL.class_labels
    assert 0.0 <= result.confidence <= 1.0
