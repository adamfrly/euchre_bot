"""Measure an agent against a fixed opponent over many hands.

This is your real progress signal. ``agent`` plays team 0 (seats 0 & 2),
``opponent`` plays team 1 (seats 1 & 3), and we report how the agent's team
does. The dealer is rotated every hand so neither team gets a positional edge,
and the agent acts *deterministically* (greedy) so the number reflects its
learned policy rather than exploration noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agents.base import Agent
from ..env.euchre_env import EuchreEnv
from .self_play import play_episode

AGENT_TEAM = 0


@dataclass
class EvalReport:
    hands: int
    win_rate: float            # fraction of hands the agent's team scored points
    avg_point_diff: float      # mean (agent points - opponent points) per hand
    points_for: int
    points_against: int

    def __str__(self) -> str:
        return (
            f"{self.hands} hands | win_rate={self.win_rate:.3f} "
            f"| avg_point_diff={self.avg_point_diff:+.3f} "
            f"| points {self.points_for}-{self.points_against}"
        )


def evaluate(
    agent: Agent,
    opponent: Agent,
    num_hands: int = 500,
    *,
    seed: int | None = 0,
    deterministic: bool = True,
) -> EvalReport:
    """Play ``num_hands`` and summarise the agent team's performance."""
    env = EuchreEnv(seed=seed)
    seating = [agent, opponent, agent, opponent]

    wins = 0
    points_for = 0
    points_against = 0
    for _ in range(num_hands):
        # reset() picks a random dealer each hand, so positional advantage
        # averages out over a large `num_hands`.
        _, info = play_episode(env, seating, deterministic=deterministic)
        scoring_team = info["scoring_team"]
        points = info["points"]
        if scoring_team == AGENT_TEAM:
            wins += 1
            points_for += points
        else:
            points_against += points

    return EvalReport(
        hands=num_hands,
        win_rate=wins / num_hands if num_hands else 0.0,
        avg_point_diff=(points_for - points_against) / num_hands if num_hands else 0.0,
        points_for=points_for,
        points_against=points_against,
    )
