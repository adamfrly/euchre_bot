"""MLflow-backed experiment logging for RL runs.

Why this exists
---------------
You cannot debug what you cannot see. RL training fails *silently* far more
often than it crashes: the loss looks fine, but the agent never improves. The
only way to catch that is to **log scalar metrics every iteration and watch the
curves** -- and, just as importantly, to keep every run's hyperparameters,
metrics, and checkpoints together so you can compare experiments later.

This project uses **MLflow** for that. MLflow is free, open-source, and runs
entirely on your machine -- no account, no server to stand up. MLflow 3.x writes
to a local ``mlflow.db`` (SQLite) plus an ``./mlartifacts/`` directory in your
working directory; browse it by running ``mlflow ui`` from the project root. See
``docs/monitoring.md`` for a full MLflow walkthrough and for *what* to log and
how to read the curves.

This :class:`MetricsLogger` is a thin wrapper around MLflow that:

* opens an MLflow **run** inside an **experiment** (a named bucket of runs);
* logs your **hyperparameters** once (``log_params``);
* logs **metrics by step** every iteration (``log``);
* attaches **checkpoints / files** to the run (``log_artifact``);
* also echoes metrics to **stdout** and mirrors them to a dependency-free
  ``runs/<run_name>/metrics.jsonl`` file, so you always have a local record
  even before MLflow is installed.

If MLflow is not installed it degrades to stdout + JSONL and tells you how to
install it -- nothing breaks.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Log params, metrics, and artifacts to MLflow (plus stdout + a JSONL mirror).

    Usage::

        with MetricsLogger(experiment="euchre-bot", run_name="ppo_v1",
                           params={"agent": "ppo", "lr": 3e-4}) as logger:
            for it in range(iterations):
                metrics = agent.learn(batch)
                logger.log(it, metrics, prefix="train/")
                if it % 25 == 0:
                    report = evaluate(agent, HeuristicAgent())
                    logger.log(it, {"win_rate": report.win_rate}, prefix="eval/")
                    agent.save("checkpoints/ppo_25.pt")
                    logger.log_artifact("checkpoints/ppo_25.pt")

    Then browse the results with ``mlflow ui`` (serves ``./mlruns`` at
    http://localhost:5000).
    """

    def __init__(
        self,
        experiment: str = "euchre-bot",
        run_name: str | None = None,
        params: dict[str, Any] | None = None,
        tracking_uri: str | None = None,
        local_mirror_dir: str = "runs",
        stdout: bool = True,
    ) -> None:
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.stdout = stdout
        self._mlflow = None

        # Dependency-free local mirror, so you always have a record on disk.
        self._dir = Path(local_mirror_dir) / run_name
        self._dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self._dir / "metrics.jsonl", "a")

        try:
            import mlflow

            self._mlflow = mlflow
        except ImportError:
            print(
                "[MetricsLogger] mlflow not installed; logging to stdout + "
                f"{self._dir / 'metrics.jsonl'} only. Install with: pip install mlflow"
            )

        if self._mlflow is not None:
            if tracking_uri is not None:
                self._mlflow.set_tracking_uri(tracking_uri)
            self._mlflow.set_experiment(experiment)
            self._mlflow.start_run(run_name=run_name)
            if params:
                self.log_params(params)

    def log_params(self, params: dict[str, Any]) -> None:
        """Record the run's hyperparameters once (e.g. lr, gamma, clip_eps, seed)."""
        if self._mlflow is not None:
            self._mlflow.log_params(params)
        if self.stdout:
            body = "  ".join(f"{k}={v}" for k, v in params.items())
            print(f"[params] {body}")

    def log(self, step: int, metrics: dict[str, Any], prefix: str = "") -> None:
        """Record a dict of scalar metrics at ``step`` (e.g. iteration number)."""
        flat = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
        if not flat:
            return
        self._jsonl.write(json.dumps({"step": step, **flat}) + "\n")
        self._jsonl.flush()
        if self._mlflow is not None:
            # MLflow allows '/' in metric keys, so our "eval/win_rate" naming
            # groups nicely in the UI.
            self._mlflow.log_metrics(flat, step=step)
        if self.stdout:
            body = "  ".join(f"{k}={v:.4f}" for k, v in flat.items())
            print(f"[step {step:>5}] {body}")

    def log_artifact(self, path: str) -> None:
        """Attach a file (e.g. a checkpoint) to the current MLflow run."""
        if self._mlflow is not None:
            self._mlflow.log_artifact(path)

    def close(self) -> None:
        self._jsonl.close()
        if self._mlflow is not None and self._mlflow.active_run() is not None:
            self._mlflow.end_run()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
