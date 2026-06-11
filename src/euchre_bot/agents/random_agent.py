"""A uniform-random legal-move agent.

The simplest possible baseline and the most useful sanity check you have: any
agent you train should crush this one. It is also the reference implementation
of "how to use the action mask correctly".
"""

from __future__ import annotations

import numpy as np

from .base import Agent


class RandomAgent(Agent):
    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict | None = None,
        *,
        deterministic: bool = False,
    ) -> int:
        legal = np.flatnonzero(action_mask)
        return int(self._rng.choice(legal))
