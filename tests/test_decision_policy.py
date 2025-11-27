import time

from hearcue.system.decision_policy import DecisionPolicy


def test_decision_policy_threshold():
    policy = DecisionPolicy(threshold=0.5, smoothing_window=2, refractory_period=0.1)
    assert policy.should_alert(0.4, "doorbell") is None
    assert policy.should_alert(0.8, "doorbell") == "doorbell"
    assert policy.should_alert(0.9, "doorbell") is None
    time.sleep(0.11)
    assert policy.should_alert(0.9, "doorbell") == "doorbell"
