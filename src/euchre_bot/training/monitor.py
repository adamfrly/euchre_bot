"""Lightweight experiment logging for RL runs.

Why this exists
---------------
You cannot debug what you cannot see. RL training fails *silently* far more
often than it crashes: the loss looks fine, but the agent never improves. The
only way to catch that is to **log scalar metrics every iteration and watch the
curves**. This module gives you one object that fans those scalars out to three
places at once:

* **stdout** -- so you notice immediately when something is off;
* **a JSONL file** (``metrics.jsonl``) -- a dependency-free record you can load
  later with pandas/matplotlib for custom plots or post-hoc analysis;
* **TensorBoard** -- if it is installed, for live, zoomable curves.

TensorBoard is the de-facto standard and is genuinely good for this: launch
``tensorboard --logdir runs`` and you get live-updating plots of every scalar,
grouped by name. It is optional here -- if it is not installed you still get
stdout + JSONL, and nothing breaks. (Weights & Biases is a popular hosted
alternative with nicer dashboards and experiment comparison; the same
``log(step, metrics)`` shape ports to ``wandb.log`` almost verbatim.)

See ``docs/monitoring.md`` for *what* to log and how to read the curves.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class MetricsLogger:
    """Fan scalar metrics out to stdout, a JSONL file, and (optionally) TensorBoard.

    Usage::

        with MetricsLogger(run_name="ppo_v1") as logger:
            for it in range(iterations):
                metrics = agent.learn(batch)
                logger.log(it, metrics)                       # training metrics
                if it % 20 == 0:
                    report = evaluate(agent, HeuristicAgent())
                    logger.log(it, {"win_rate": report.win_rate,
                                    "point_diff": report.avg_point_diff},
                               prefix="eval/")
    """

    def __init__(
        self,
        logdir: str = "runs",
        run_name: str | None = None,
        use_tensorboard: bool = True,
        stdout: bool = True,
    ) -> None:
        run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.dir = Path(logdir) / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.stdout = stdout
        self._jsonl = open(self.dir / "metrics.jsonl", "a")
        self.writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.dir))
            except Exception:  # torch/tensorboard not installed -- degrade gracefully
                print(
                    "[MetricsLogger] TensorBoard unavailable; logging to stdout + "
                    f"{self.dir / 'metrics.jsonl'} only. "
                    "Install with: pip install tensorboard"
                )

    def log(self, step: int, metrics: dict[str, Any], prefix: str = "") -> None:
        """Record a dict of scalar metrics at ``step`` (e.g. iteration number)."""
        flat = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
        if not flat:
            return
        self._jsonl.write(json.dumps({"step": step, **flat}) + "\n")
        self._jsonl.flush()
        if self.writer is not None:
            for key, value in flat.items():
                self.writer.add_scalar(key, value, step)
        if self.stdout:
            body = "  ".join(f"{k}={v:.4f}" for k, v in flat.items())
            print(f"[step {step:>5}] {body}")

    def close(self) -> None:
        self._jsonl.close()
        if self.writer is not None:
            self.writer.close()

    def __enter__(self) -> MetricsLogger:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
