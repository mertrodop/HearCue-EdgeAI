import numpy as np

from hearcue.audio.logmelspec import log_mel_spectrogram
from hearcue.audio.ring_buffer import RingBuffer
from hearcue.utils.constants import AUDIO


def test_ring_buffer_basic():
    rb = RingBuffer(16)
    rb.write(np.arange(16, dtype=np.float32))
    window = rb.read(8)
    assert window is not None
    np.testing.assert_array_equal(window, np.arange(8, 16, dtype=np.float32))


def test_log_mel_shape():
    signal = np.random.randn(AUDIO.frame_length * 2).astype(np.float32)
    spec = log_mel_spectrogram(signal)
    assert spec.shape[1] == AUDIO.n_mels
