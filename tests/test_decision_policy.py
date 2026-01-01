import time

import numpy as np
import pytest

from hearcue.system.decision_policy import DecisionPolicy
from hearcue.utils.constants import MODEL


def _probs_for(label: str, conf: float) -> np.ndarray:
    probs = np.full(len(MODEL.class_labels), 0.01, dtype=np.float32)
    idx = list(MODEL.class_labels).index(label)
    probs[idx] = conf
    return probs


def test_decision_policy_threshold_and_smoothing():
    policy = DecisionPolicy(
        threshold=0.5,
        window=3,
        refractory_period=0.1,
        class_thresholds={"alarm": 0.5},
        required_hits={"alarm": 2},
    )

    # Below threshold -> no alert
    label, conf = policy.should_alert(_probs_for("alarm", 0.4))
    assert label is None
    assert conf == pytest.approx(0.4)

    # First above threshold hit -> not enough history yet
    label, conf = policy.should_alert(_probs_for("alarm", 0.6))
    assert label is None
    assert conf == pytest.approx(0.6)

    # Second hit within window -> triggers
    label, conf = policy.should_alert(_probs_for("alarm", 0.7))
    assert label == "alarm"
    assert conf == pytest.approx(0.7)

    # Refractory: immediate repeat should not trigger
    label, conf = policy.should_alert(_probs_for("alarm", 0.8))
    assert label is None
    assert conf == pytest.approx(0.8)

    time.sleep(0.11)
    label, conf = policy.should_alert(_probs_for("alarm", 0.8))
    assert label == "alarm"
    assert conf == pytest.approx(0.8)
