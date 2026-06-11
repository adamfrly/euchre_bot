"""Correctness checks for the worked REINFORCE reference.

These are fast and torch-guarded (skipped if torch isn't installed). They assert
the agent *behaves correctly* -- only ever plays legal actions, produces finite
training metrics, and round-trips through save/load -- not that it reaches some
win-rate, which would be slow and flaky. The end-to-end "does it learn" check
lives in the README/docs commands.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from euchre_bot.algos import ReinforceAgent  # noqa: E402
from euchre_bot.env.euchre_env import EuchreEnv  # noqa: E402
from euchre_bot.training.self_play import collect_batch, play_episode  # noqa: E402


def test_act_only_returns_legal_actions():
    env = EuchreEnv(seed=1)
    agent = ReinforceAgent(seed=0)
    result = env.reset(seed=1)
    while not result.terminated:
        action = agent.act(result.observation, result.action_mask)
        assert result.action_mask[action] == 1.0  # never an illegal action
        result = env.step(action)


def test_deterministic_act_is_repeatable():
    env = EuchreEnv(seed=2)
    agent = ReinforceAgent(seed=0)
    result = env.reset(seed=2)
    a1 = agent.act(result.observation, result.action_mask, deterministic=True)
    a2 = agent.act(result.observation, result.action_mask, deterministic=True)
    assert a1 == a2


def test_learn_returns_finite_metrics_and_updates_params():
    env = EuchreEnv(seed=3)
    agent = ReinforceAgent(seed=0, lr=1e-2)
    before = [p.detach().clone() for p in agent.policy.parameters()]

    batch = collect_batch(env, agent, num_episodes=8)
    metrics = agent.learn(batch)

    assert np.isfinite(metrics["loss"])
    assert metrics["entropy"] >= 0.0
    assert metrics["transitions"] > 0
    # At least one parameter tensor changed -- a gradient step actually happened.
    after = list(agent.policy.parameters())
    assert any(not p.equal(b) for p, b in zip(after, before, strict=True))


def test_save_and_load_roundtrip(tmp_path):
    agent = ReinforceAgent(seed=0)
    path = str(tmp_path / "reinforce.pt")
    agent.save(path)

    reloaded = ReinforceAgent(seed=123)  # different init
    reloaded.load(path)
    for p, q in zip(agent.policy.parameters(), reloaded.policy.parameters(), strict=True):
        assert p.equal(q)


def test_self_play_episode_with_reinforce():
    env = EuchreEnv(seed=4)
    agent = ReinforceAgent(seed=0)
    trajectories, info = play_episode(env, agent, seed=4)
    assert sum(info["tricks_won"]) == 5
    assert any(traj for traj in trajectories.values())
