from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from helix_lm.experiment_tracking import ExperimentTracker


class FakeMLflow:
    def __init__(self):
        self.metrics = []
        self.params = {}
        self.tags = {}
        self.status = None

    def set_tracking_uri(self, value):
        self.tracking_uri = value

    def set_experiment(self, value):
        self.experiment = value

    def start_run(self, run_name):
        self.run_name = run_name
        return SimpleNamespace(info=SimpleNamespace(run_id="run-123"))

    def log_params(self, values):
        self.params.update(values)

    def set_tag(self, key, value):
        self.tags[key] = value

    def log_metrics(self, values, step):
        self.metrics.append((step, values))

    def end_run(self, status):
        self.status = status


class ExperimentTrackerTest(unittest.TestCase):
    def test_logs_the_same_metrics_locally_and_remotely(self):
        with tempfile.TemporaryDirectory() as temporary:
            spool = Path(temporary) / "metrics.jsonl"
            remote = FakeMLflow()
            tracker = ExperimentTracker(
                tracking_uri="https://mlflow.example",
                experiment="Helix",
                run_name="branch62-court",
                spool_path=spool,
                params={"d_model": 768},
                tags={"source_head": "abc"},
                mlflow_module=remote,
            )

            self.assertEqual(tracker.start(), "run-123")
            tracker.log_metrics(
                {"train/loss": 2.5, "ignored": "not-numeric"},
                step=7,
                phase="train",
            )
            terminal = tracker.finish("FINISHED")

            events = [json.loads(line) for line in spool.read_text().splitlines()]
            metric_event = next(event for event in events if event["event"] == "metrics")
            self.assertEqual(metric_event["metrics"], {"train/loss": 2.5})
            self.assertEqual(remote.metrics, [(7, {"train/loss": 2.5})])
            self.assertEqual(remote.params["d_model"], "768")
            self.assertEqual(remote.tags["source_head"], "abc")
            self.assertEqual(remote.status, "FINISHED")
            self.assertEqual(terminal, "FINISHED")


if __name__ == "__main__":
    unittest.main()
