"""Log-mel spectrogram extraction utilities."""
from __future__ import annotations

import numpy as np

from hearcue.utils.constants import AUDIO


def pre_emphasis(signal: np.ndarray, coeff: float = AUDIO.pre_emphasis) -> np.ndarray:
    emphasized = np.append(signal[0], signal[1:] - coeff * signal[:-1])
    return emphasized.astype(np.float32)


def framing(signal: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    shape = (num_frames, frame_length)
    strides = (signal.strides[0] * hop_length, signal.strides[0])
    frames = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)
    window = np.hamming(frame_length).astype(np.float32)
    return frames * window


def power_spectrum(frames: np.ndarray, n_fft: int) -> np.ndarray:
    fft = np.fft.rfft(frames, n=n_fft)
    return (np.abs(fft) ** 2).astype(np.float32)


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int, fmin: int, fmax: int) -> np.ndarray:
    import librosa

    return librosa.filters.mel(
        sr=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    ).astype(np.float32)


def log_mel_spectrogram(
    signal: np.ndarray,
    sample_rate: int = AUDIO.sample_rate,
    frame_length: int = AUDIO.frame_length,
    hop_length: int = AUDIO.hop_length,
    n_mels: int = AUDIO.n_mels,
    fmin: int = AUDIO.fmin,
    fmax: int = AUDIO.fmax,
) -> np.ndarray:
    if len(signal) < frame_length:
        pad = np.zeros(frame_length - len(signal), dtype=np.float32)
        signal = np.concatenate((signal, pad))
    emphasized = pre_emphasis(signal)
    frames = framing(emphasized, frame_length, hop_length)
    spec = power_spectrum(frames, frame_length)
    mel_fb = mel_filterbank(sample_rate, frame_length, n_mels, fmin, fmax)
    mel_spec = np.dot(spec, mel_fb.T)
    mel_spec = np.maximum(mel_spec, 1e-10)
    log_mel = np.log(mel_spec)
    return log_mel.astype(np.float32)
