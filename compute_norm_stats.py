import numpy as np
from pathlib import Path

from hearcue.utils.constants import AUDIO, MODEL
from hearcue.utils.helpers import load_audio
from hearcue.audio.logmelspec import log_mel_spectrogram


def main() -> None:
    data_dir = Path("/home/rodor/Desktop/dataset_out/processed/train")
    target_samples = AUDIO.sample_rate * 2  # match training: 2s clips
    feats = []

    for label in MODEL.class_labels:
        p = data_dir / label
        if not p.exists():
            print("missing:", p)
            continue
        for wav in list(p.glob("*.wav"))[:200]:
            x, _ = load_audio(wav, AUDIO.sample_rate)
            if len(x) < target_samples:
                x = np.pad(x, (0, target_samples - len(x)), mode="constant")
            elif len(x) > target_samples:
                x = x[:target_samples]
            spec = log_mel_spectrogram(x)
            feats.append(spec)

    X = np.array(feats, dtype=np.float32)
    print("count", len(feats))
    if len(feats) == 0:
        return
    print("mean", float(X.mean()))
    print("std", float(X.std()))


if __name__ == "__main__":
    main()
