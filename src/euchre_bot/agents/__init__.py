"""Agents: the common interface plus ready-to-use baselines.

Your own learning agents live in :mod:`euchre_bot.algos`; they subclass
:class:`Agent` from here so they slot into the same training and evaluation
machinery as these baselines.
"""

from .base import Agent, Trajectory, Transition
from .heuristic_agent import HeuristicAgent
from .random_agent import RandomAgent

__all__ = ["Agent", "Trajectory", "Transition", "RandomAgent", "HeuristicAgent"]
