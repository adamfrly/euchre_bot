"""The agent contract every policy in this project implements.

Keeping a single tiny interface means the training loop, the evaluation
harness, the human-play CLI, and *your* future RL agents are all
interchangeable. A random bot, a hand-written heuristic, and a deep network are
the same shape from the outside::

    action = agent.act(observation, action_mask, info)

Design notes that matter when you implement an RL agent:

* ``act`` receives the **observation vector** and the **action mask**. A
  learning agent should depend only on these two -- they are exactly what the
  network sees, and nothing in them leaks hidden information. The ``info`` dict
  (which carries the structured :class:`PlayerView`) exists so that *scripted*
  agents can reason in terms of cards; ignore it in a neural agent.
* The mask is not optional. The network outputs a score for all
  :data:`~euchre_bot.engine.actions.NUM_ACTIONS` actions; you must set the
  logits of illegal actions to ``-inf`` (or multiply probabilities by the mask
  and renormalise) *before* sampling. Sampling an illegal action will raise in
  the environment -- that is intentional, it catches a very common bug.
* Learning is decoupled from acting. The training loop collects whole
  trajectories and then calls :meth:`Agent.learn`. Because every algorithm in
  this project is *on-policy* (REINFORCE/A2C/PPO) or replays stored transitions
  (DQN), you can recompute log-probs and values inside ``learn`` from the stored
  ``(observation, action)`` pairs -- you do not need to stash tensors during
  ``act``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    """One decision made by one seat, plus what followed.

    ``reward`` is non-zero only on the final transition of a seat's trajectory
    (Euchre rewards are sparse and arrive at the end of the hand); ``done``
    marks that final transition. Your algorithm turns these into discounted
    returns / advantages.
    """

    observation: np.ndarray
    action_mask: np.ndarray
    action: int
    reward: float
    done: bool


#: A trajectory is the ordered list of transitions a single seat experienced in
#: one episode (hand). The training loop hands :meth:`Agent.learn` a *batch* of
#: these: ``list[Trajectory]``.
Trajectory = list[Transition]


class Agent(ABC):
    """Base class for everything that can choose Euchre actions."""

    @abstractmethod
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict | None = None,
        *,
        deterministic: bool = False,
    ) -> int:
        """Return a legal action id.

        Args:
            observation: The flat observation vector for the seat on turn.
            action_mask: ``1.0`` for legal actions, ``0.0`` otherwise.
            info: Optional extras from the environment (carries ``player_view``);
                learning agents should not need it.
            deterministic: If ``True``, act greedily (used during evaluation);
                if ``False``, sample (used during training for exploration).
        """

    def learn(self, batch: list[Trajectory]) -> dict[str, float]:
        """Update the policy from a batch of trajectories. Returns metrics.

        Non-learning agents (random, heuristic) keep the default no-op.
        """
        return {}

    def save(self, path: str) -> None:  # pragma: no cover - optional
        """Persist parameters. Optional; learning agents should override."""
        raise NotImplementedError

    def load(self, path: str) -> None:  # pragma: no cover - optional
        """Restore parameters. Optional; learning agents should override."""
        raise NotImplementedError
