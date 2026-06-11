"""Self-play rollout collection and a generic training loop.

This module is algorithm-agnostic: it knows how to *run hands* and gather
trajectories, then it hands them to ``agent.learn(...)``. Whatever you put in
``learn`` -- REINFORCE, A2C, PPO, DQN -- plugs in here unchanged.

Self-play in a nutshell
-----------------------
The same agent object sits in all four seats. Because the policy is symmetric
(observations are seat-relative), one network can play everyone. Each hand
produces up to four trajectories -- one per seat -- and all of them are training
data for that one agent. Partners cooperate and opponents compete *through the
reward*, not through any special code here.

A word of warning, recorded so you remember it later: a policy's win-rate
against a copy of itself is always about 50%. It tells you nothing about whether
the agent is improving. Always evaluate against a *fixed* reference
(``HeuristicAgent`` / ``RandomAgent``) via :mod:`euchre_bot.training.evaluate`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..agents.base import Agent, Trajectory, Transition
from ..env.euchre_env import NUM_PLAYERS, EuchreEnv


def _as_seating(agents: Agent | Sequence[Agent]) -> list[Agent]:
    """Normalise an agent (or 4 agents) into a per-seat list of length 4."""
    if isinstance(agents, Agent):
        return [agents] * NUM_PLAYERS
    seating = list(agents)
    if len(seating) != NUM_PLAYERS:
        raise ValueError(f"expected 1 agent or {NUM_PLAYERS}, got {len(seating)}")
    return seating


def play_episode(
    env: EuchreEnv,
    agents: Agent | Sequence[Agent],
    *,
    deterministic: bool = False,
    seed: int | None = None,
) -> tuple[dict[int, Trajectory], dict]:
    """Play one full hand. Return ``(trajectories_by_seat, terminal_info)``.

    A seat's trajectory is the list of decisions it made this hand, with the
    sparse terminal reward written onto its final transition. A lone maker's
    partner makes no decisions and so has an empty trajectory.
    """
    seating = _as_seating(agents)
    result = env.reset(seed=seed)
    trajectories: dict[int, Trajectory] = {seat: [] for seat in range(NUM_PLAYERS)}

    while not result.terminated:
        seat = result.current_player
        assert seat is not None and result.observation is not None
        action = seating[seat].act(
            result.observation, result.action_mask, result.info, deterministic=deterministic
        )
        trajectories[seat].append(
            Transition(result.observation, result.action_mask, action, 0.0, False)
        )
        result = env.step(action)

    # Attribute each seat's end-of-hand reward to its last decision.
    for seat, reward in result.rewards.items():
        traj = trajectories[seat]
        if traj:
            last = traj[-1]
            traj[-1] = Transition(
                last.observation, last.action_mask, last.action, float(reward), done=True
            )

    return trajectories, result.info


def collect_batch(
    env: EuchreEnv,
    learner: Agent,
    num_episodes: int,
    *,
    opponent: Agent | None = None,
) -> list[Trajectory]:
    """Gather a batch of trajectories for ``learner`` to train on.

    If ``opponent`` is ``None`` this is pure self-play and every seat's
    trajectory is collected. Otherwise ``learner`` plays team 0 (seats 0 & 2)
    against ``opponent`` on team 1, and only the learner's trajectories are
    returned.
    """
    batch: list[Trajectory] = []
    if opponent is None:
        for _ in range(num_episodes):
            trajectories, _ = play_episode(env, learner, deterministic=False)
            batch.extend(t for t in trajectories.values() if t)
    else:
        seating = [learner, opponent, learner, opponent]
        learner_seats = (0, 2)
        for _ in range(num_episodes):
            trajectories, _ = play_episode(env, seating, deterministic=False)
            batch.extend(trajectories[s] for s in learner_seats if trajectories[s])
    return batch


def train(
    learner: Agent,
    *,
    env: EuchreEnv | None = None,
    iterations: int = 100,
    episodes_per_iter: int = 64,
    opponent: Agent | None = None,
    on_iteration: Callable[[int, dict], None] | None = None,
) -> Agent:
    """Run the collect-then-learn loop.

    Args:
        learner: The agent being trained (its :meth:`Agent.learn` does the work).
        env: Environment to roll out in; a default :class:`EuchreEnv` if ``None``.
        iterations: Number of collect/learn cycles.
        episodes_per_iter: Hands collected before each ``learn`` call.
        opponent: Fixed opponent to train against; ``None`` for self-play.
        on_iteration: Optional callback ``(iteration, metrics)`` for logging or
            evaluation (e.g. call :func:`evaluate.evaluate` here every N iters).

    Returns:
        The same ``learner``, now trained.
    """
    env = env or EuchreEnv()
    for iteration in range(1, iterations + 1):
        batch = collect_batch(env, learner, episodes_per_iter, opponent=opponent)
        metrics = learner.learn(batch)
        if on_iteration is not None:
            on_iteration(iteration, metrics)
    return learner
