"""Correctness checks for the A2C, PPO, and DQN agents.

Like ``test_reinforce.py`` these are fast and torch-guarded. They assert the
agents *behave correctly* -- legal actions only, finite metrics, a real
parameter update, and a save/load round-trip -- not that they reach a win-rate.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from euchre_bot.agents import HeuristicAgent  # noqa: E402
from euchre_bot.algos import A2CAgent, DQNAgent, PPOAgent  # noqa: E402
from euchre_bot.algos.a2c import discounted_returns  # noqa: E402
from euchre_bot.env.euchre_env import EuchreEnv  # noqa: E402
from euchre_bot.training.self_play import collect_batch  # noqa: E402

# DQN learns from its first batch only if the warm-up threshold is low.
FACTORIES = {
    "a2c": lambda **kw: A2CAgent(**kw),
    "ppo": lambda **kw: PPOAgent(**kw),
    "dqn": lambda **kw: DQNAgent(learning_starts=0, batch_size=32, **kw),
}


def _params(agent):
    net = agent.q_net if isinstance(agent, DQNAgent) else agent.net
    return list(net.parameters())


@pytest.fixture(params=sorted(FACTORIES))
def make(request):
    return FACTORIES[request.param]


def test_act_only_returns_legal_actions(make):
    env = EuchreEnv(seed=1)
    agent = make(seed=0)
    for deterministic in (False, True):
        result = env.reset(seed=1)
        while not result.terminated:
            action = agent.act(result.observation, result.action_mask,
                               deterministic=deterministic)
            assert result.action_mask[action] == 1.0
            result = env.step(action)


def test_deterministic_act_is_repeatable(make):
    env = EuchreEnv(seed=2)
    agent = make(seed=0)
    result = env.reset(seed=2)
    a1 = agent.act(result.observation, result.action_mask, deterministic=True)
    a2 = agent.act(result.observation, result.action_mask, deterministic=True)
    assert a1 == a2


@pytest.mark.parametrize("opponent", [None, HeuristicAgent()])
def test_learn_returns_finite_metrics_and_updates_params(make, opponent):
    env = EuchreEnv(seed=3)
    agent = make(seed=0, lr=1e-2)
    before = [p.detach().clone() for p in _params(agent)]

    batch = collect_batch(env, agent, num_episodes=16, opponent=opponent)
    metrics = agent.learn(batch)

    assert "loss" in metrics or "policy_loss" in metrics
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(not p.equal(b) for p, b in zip(_params(agent), before, strict=True))


def test_save_and_load_roundtrip(make, tmp_path):
    agent = make(seed=0)
    path = str(tmp_path / "agent.pt")
    agent.save(path)

    reloaded = make(seed=123)
    reloaded.load(path)
    for p, q in zip(_params(agent), _params(reloaded), strict=True):
        assert p.equal(q)


def test_discounted_returns():
    assert discounted_returns([0.0, 0.0, 2.0], gamma=0.5) == [0.5, 1.0, 2.0]


def test_ppo_gae_matches_monte_carlo_when_lambda_is_one():
    agent = PPOAgent(gamma=0.9, gae_lambda=1.0, seed=0)
    rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    values = np.array([0.3, -0.2, 0.5], dtype=np.float32)
    # With lambda=1, GAE telescopes to (MC return - V).
    expected = np.array(discounted_returns(list(rewards), 0.9)) - values
    np.testing.assert_allclose(agent._gae(rewards, values), expected, rtol=1e-5)


def test_dqn_pairs_next_states_within_a_trajectory():
    env = EuchreEnv(seed=5)
    agent = DQNAgent(seed=0)
    batch = collect_batch(env, agent, num_episodes=4)
    agent.learn(batch)

    n = sum(len(t) for t in batch)
    buf = agent.buffer
    assert len(buf) == n
    # Exactly one terminal transition per non-empty trajectory.
    assert int(buf.dones[:n].sum()) == len([t for t in batch if t])
    first = batch[0]
    if len(first) > 1:
        np.testing.assert_array_equal(buf.next_obs[0], first[1].observation)
        np.testing.assert_array_equal(buf.next_masks[0], first[1].action_mask > 0)
        assert buf.dones[0] == 0.0

