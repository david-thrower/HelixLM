"""MLflow projection with an append-only local metrics record.

The local JSONL file is the durable experiment log. MLflow is required to
admit a run at startup when ``require_remote`` is true, but a later tracking
outage does not destroy training progress; the failed projection is recorded
for replay and the terminal is marked accordingly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional


class ExperimentTracker:
    """Project training metrics to MLflow and retain a local JSONL spool."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment: str,
        run_name: str,
        spool_path: Path,
        params: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, Any]] = None,
        require_remote: bool = True,
        mlflow_module: Any = None,
    ) -> None:
        self.tracking_uri = tracking_uri
        self.experiment = experiment
        self.run_name = run_name
        self.spool_path = Path(spool_path)
        self.spool_path.parent.mkdir(parents=True, exist_ok=True)
        self.params = dict(params or {})
        self.tags = dict(tags or {})
        self.require_remote = bool(require_remote)
        self.mlflow = mlflow_module
        self.run_id: Optional[str] = None
        self.errors: list[str] = []
        self._append({"event": "tracker_initialized", "ts": time.time()})

    def _append(self, event: dict[str, Any]) -> None:
        with self.spool_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True, default=str) + "\n")

    def _remote(self, operation: str, action: Callable[[], Any]) -> Any:
        try:
            return action()
        except Exception as exc:  # pragma: no cover - remote failure is environmental
            message = f"{operation}: {type(exc).__name__}: {exc}"
            self.errors.append(message)
            self._append(
                {
                    "event": "mlflow_error",
                    "operation": operation,
                    "error": message,
                    "ts": time.time(),
                }
            )
            return None

    def start(self) -> str:
        """Start and identify the remote run before model training begins."""
        if self.mlflow is None:
            try:
                import mlflow
            except Exception as exc:
                message = f"import: {type(exc).__name__}: {exc}"
                self.errors.append(message)
                self._append(
                    {"event": "mlflow_unavailable", "error": message, "ts": time.time()}
                )
                if self.require_remote:
                    raise RuntimeError("MLflow is required but unavailable") from exc
                return ""
            self.mlflow = mlflow

        self.mlflow.set_tracking_uri(self.tracking_uri)
        self.mlflow.set_experiment(self.experiment)
        run = self._remote(
            "start_run",
            lambda: self.mlflow.start_run(run_name=self.run_name),
        )
        if run is None:
            if self.require_remote:
                raise RuntimeError("MLflow run could not be started")
            return ""
        self.run_id = str(run.info.run_id)
        self._remote(
            "log_params",
            lambda: self.mlflow.log_params(
                {key: str(value) for key, value in self.params.items()}
            ),
        )
        self._remote(
            "set_tags",
            lambda: [
                self.mlflow.set_tag(key, str(value))
                for key, value in self.tags.items()
            ],
        )
        self._append(
            {"event": "run_started", "run_id": self.run_id, "ts": time.time()}
        )
        return self.run_id

    def log_metrics(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        phase: str,
    ) -> None:
        clean = {
            str(key): float(value)
            for key, value in metrics.items()
            if value is not None and isinstance(value, (int, float))
        }
        self._append(
            {
                "event": "metrics",
                "phase": phase,
                "step": int(step),
                "metrics": clean,
                "ts": time.time(),
            }
        )
        if self.run_id and clean:
            self._remote(
                "log_metrics",
                lambda: self.mlflow.log_metrics(clean, step=int(step)),
            )

    def finish(self, status: str) -> str:
        """Close the remote projection and record whether it stayed complete."""
        if self.run_id:
            mlflow_status = "FINISHED" if status == "FINISHED" else "FAILED"
            self._remote("end_run", lambda: self.mlflow.end_run(status=mlflow_status))
        projected_status = status if not self.errors else f"{status}_WITH_MLFLOW_ERRORS"
        self._append(
            {
                "event": "run_finished",
                "status": projected_status,
                "errors": self.errors,
                "ts": time.time(),
            }
        )
        return projected_status
