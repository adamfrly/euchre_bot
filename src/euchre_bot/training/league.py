"""League training: self-play against a pool of past selves and fixed bots.

Why not pure self-play, or a single fixed opponent?
---------------------------------------------------
Training against one fixed opponent teaches the learner to *exploit that
opponent* -- against the default HeuristicAgent, which calls trump too loosely,
our agents learned to pass almost everything and euchre it, and that edge
evaporated against a tighter bidder. Pure self-play generalises better but can
chase its own tail (forgetting how to beat strategies it no longer plays).

A league mixes both. Each hand, the learner's opponents are drawn from:

* the **current policy** (plain self-play -- all four seats are training data);
* **frozen snapshots** of the learner from earlier in training;
* **fixed bots** -- e.g. HeuristicAgent at several bidding thresholds.

Only the learner's own seats are training data when it faces a snapshot or a
fixed bot.
"""

from __future__ import annotations

import copy
from collections import deque

import numpy as np

from ..agents.base import Agent, Trajectory
from ..env.euchre_env import EuchreEnv
from .self_play import play_episode


class OpponentPool:
    """Samples an opponent per hand: the learner itself, a snapshot, or a fixed bot.

    Args:
        fixed: Bots that are always available (heuristic variants, etc.).
        p_self: Probability of pure self-play for a hand.
        p_snapshot: Probability of facing a frozen snapshot (once any exist).
            The remaining probability goes to the fixed bots.
        max_snapshots: Oldest snapshots are dropped beyond this many.
    """

    def __init__(
        self,
        fixed: list[Agent],
        *,
        p_self: float = 0.4,
        p_snapshot: float = 0.3,
        max_snapshots: int = 20,
        seed: int | None = None,
    ) -> None:
        if p_self + p_snapshot > 1.0:
            raise ValueError("p_self + p_snapshot must be <= 1")
        self.fixed = list(fixed)
        self.p_self = p_self
        self.p_snapshot = p_snapshot
        self.snapshots: deque[Agent] = deque(maxlen=max_snapshots)
        self.rng = np.random.default_rng(seed)

    def add_snapshot(self, learner: Agent) -> None:
        """Freeze a copy of the learner's current policy into the pool."""
        self.snapshots.append(copy.deepcopy(learner))

    def sample(self) -> Agent | None:
        """Return an opponent, or ``None`` meaning "the learner itself"."""
        u = self.rng.random()
        if u < self.p_self:
            return None
        if self.snapshots and u < self.p_self + self.p_snapshot:
            return self.snapshots[int(self.rng.integers(len(self.snapshots)))]
        if self.fixed:
            return self.fixed[int(self.rng.integers(len(self.fixed)))]
        return None


def collect_league_batch(
    env: EuchreEnv, learner: Agent, num_episodes: int, pool: OpponentPool
) -> list[Trajectory]:
    """Like :func:`collect_batch`, but draws a fresh opponent from ``pool`` every hand."""
    batch: list[Trajectory] = []
    for _ in range(num_episodes):
        opponent = pool.sample()
        if opponent is None:
            trajectories, _ = play_episode(env, learner, deterministic=False)
            batch.extend(t for t in trajectories.values() if t)
        else:
            seating = [learner, opponent, learner, opponent]
            trajectories, _ = play_episode(env, seating, deterministic=False)
            batch.extend(trajectories[s] for s in (0, 2) if trajectories[s])
    return batch


class ScoreBlind(Agent):
    """Wrap an agent trained before game scores were observed; hide the score from it.

    Checkpoints trained with the score features always zero never learned those
    input weights, so a nonzero score would inject noise. The score is the last
    two observation features (see ``env/encoding.py``).
    """

    def __init__(self, inner: Agent) -> None:
        self.inner = inner

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        observation = observation.copy()
        observation[-2:] = 0.0
        return self.inner.act(observation, action_mask, info, deterministic=deterministic)
