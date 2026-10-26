"""Self-play training loop, evaluation harness, and experiment logging."""

from .evaluate import EvalReport, GameReport, evaluate, evaluate_games
from .league import OpponentPool, ScoreBlind, collect_league_batch
from .monitor import MetricsLogger
from .self_play import collect_batch, play_episode, train

__all__ = [
    "train",
    "collect_batch",
    "play_episode",
    "evaluate",
    "evaluate_games",
    "EvalReport",
    "GameReport",
    "MetricsLogger",
    "OpponentPool",
    "ScoreBlind",
    "collect_league_batch",
]
