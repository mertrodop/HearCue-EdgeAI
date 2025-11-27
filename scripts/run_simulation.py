"""Run HearCue scenario simulations."""
from __future__ import annotations

import argparse

from hearcue.simulation.false_alarm_tester import FalseAlarmTester
from hearcue.simulation.indoor import build_scenario as indoor_scenario
from hearcue.simulation.latency_tester import LatencyTester
from hearcue.simulation.library import build_scenario as library_scenario
from hearcue.simulation.outdoor import build_scenario as outdoor_scenario
from hearcue.system.device_controller import DeviceController

SCENARIOS = {
    "indoor": indoor_scenario,
    "outdoor": outdoor_scenario,
    "library": library_scenario,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HearCue simulations.")
    parser.add_argument("scenario", choices=SCENARIOS.keys(), help="Scenario name")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]()
    controller = DeviceController()
    latency = LatencyTester(controller, scenario).run()
    false_alarm = FalseAlarmTester(controller, scenario).run()
    print(f"Latency summary: {latency.summary()}")
    print(f"Missed events: {latency.misses}")
    print(f"False positives per minute: {false_alarm.per_minute:.2f}")


if __name__ == "__main__":
    main()
