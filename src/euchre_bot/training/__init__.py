"""Self-play training loop, evaluation harness, and experiment logging."""

from .evaluate import EvalReport, evaluate
from .monitor import MetricsLogger
from .self_play import collect_batch, play_episode, train

__all__ = [
    "train",
    "collect_batch",
    "play_episode",
    "evaluate",
    "EvalReport",
    "MetricsLogger",
]
