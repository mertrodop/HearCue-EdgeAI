# HearCue

HearCue is a Python-only simulation of an embedded acoustic awareness device for Deaf and Hard-of-Hearing users. It reproduces the MCU signal chain entirely in Python: streaming audio intake, DSP preprocessing, quantized CNN inference, decision logic, haptic/LED feedback, and scenario-driven studies.

## Features
- Log-mel feature extraction mirroring MCU DSP
- Tiny CNN training, quantization, and TFLite-style inference
- MCU-inspired decision policy with temporal smoothing and refractory timers
- Simulated haptic motor and RGB LED responses
- Indoor/outdoor/library scenario playback, latency, and false alarm testing
- Evaluation utilities for F1, confusion matrices, SUS, and NASA TLX

## Architecture
```
Audio Source -> RingBuffer -> Log-Mel DSP -> Quantized CNN -> Decision Policy
                                                       |-> HapticDriver
                                                       |-> LEDDriver
Simulation Scenarios -> EventPlayer -> DeviceController Loop
```

## Getting Started
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Populate `data/processed/<label>` with WAV files per class.
3. Train and quantize the CNN:
   ```bash
   python scripts/run_training.py --data_dir data/processed --epochs 5
   ```
4. Run a simulation scenario:
   ```bash
   python scripts/run_simulation.py indoor
   ```
5. Evaluate predictions stored in a CSV:
   ```bash
   python scripts/run_evaluation.py predictions.csv --output confmat.png
   ```

## Module Overview
- `hearcue/audio`: microphone streaming, ring buffer, and log-mel DSP.
- `hearcue/model`: CNN training (`train_cnn.py`), quantization (`quantize.py`), and inference (`infer.py`).
- `hearcue/system`: MCU-like decision policy, haptic and LED drivers, and `DeviceController` orchestrating the loop.
- `hearcue/simulation`: scenario builders, event synthesis, latency tester, and false alarm tester.
- `hearcue/evaluation`: metrics for F1, confusion matrices, SUS, and NASA TLX.
- `scripts`: entry points for training, simulation, and evaluation workflows.

## Testing
Run the unit tests locally:
```bash
pytest
```
Tests cover the DSP pipeline, decision policy timing logic, and quantized inference integration.

## Data Management
Raw assets belong in `data/raw`, while curated training clips live in `data/processed/<label>`. The helper scripts automatically create any missing directories.

## Notebooks
Two notebooks illustrate model training (`training_notebook.ipynb`) and evaluation (`evaluation_notebook.ipynb`). Launch them via Jupyter Lab for interactive exploration.