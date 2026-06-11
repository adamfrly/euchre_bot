"""Integration tests for the environment, encoding, and baseline agents."""

import numpy as np
import pytest

from euchre_bot.agents import HeuristicAgent, RandomAgent
from euchre_bot.engine import actions as A
from euchre_bot.engine.state import Phase
from euchre_bot.env.euchre_env import NUM_PLAYERS, EuchreEnv
from euchre_bot.training.self_play import play_episode


def _run_random_hand(env, seed):
    """Drive a hand with uniform-random legal moves; return the terminal result."""
    rng = np.random.default_rng(seed)
    result = env.reset(seed=seed)
    steps = 0
    while not result.terminated:
        legal = np.flatnonzero(result.action_mask)
        # Every state must offer at least one legal action.
        assert legal.size > 0
        action = int(rng.choice(legal))
        result = env.step(action)
        steps += 1
        assert steps < 200  # guard against an accidental infinite loop
    return result


@pytest.mark.parametrize("seed", range(25))
def test_random_hand_terminates_with_valid_rewards(seed):
    env = EuchreEnv(seed=seed)
    result = _run_random_hand(env, seed)
    assert result.terminated
    # Exactly five tricks were played.
    assert sum(result.info["tricks_won"]) == 5
    # Points are one of the legal Euchre values.
    assert result.info["points"] in (1, 2, 4)
    # Zero-sum reward: the four seats sum to zero.
    assert set(result.rewards.keys()) == set(range(NUM_PLAYERS))
    assert sum(result.rewards.values()) == pytest.approx(0.0)


def test_observation_and_mask_shapes():
    env = EuchreEnv(seed=1)
    result = env.reset(seed=1)
    assert result.observation.shape == (EuchreEnv.observation_dim,)
    assert result.action_mask.shape == (EuchreEnv.num_actions,)
    assert result.action_mask.sum() > 0


def test_illegal_action_raises():
    env = EuchreEnv(seed=1)
    env.reset(seed=1)
    legal = set(env.legal_actions())
    illegal = next(a for a in range(EuchreEnv.num_actions) if a not in legal)
    with pytest.raises(ValueError):
        env.step(illegal)


def test_stick_the_dealer_forbids_dealer_pass_in_round_two():
    env = EuchreEnv(stick_the_dealer=True, seed=3)
    s = env.state = None
    env.reset(seed=3)
    s = env.state
    # Force everyone to pass round 1 so we reach round 2.
    for _ in range(NUM_PLAYERS):
        assert s.phase == Phase.BID_ROUND_1
        env.step(A.PASS)
    assert s.phase == Phase.BID_ROUND_2
    # Pass the three non-dealers; the dealer must then be unable to pass.
    while s.current_player != s.dealer:
        env.step(A.PASS)
    assert s.current_player == s.dealer
    assert A.PASS not in env.legal_actions()


def test_without_stick_the_dealer_all_passes_redeal():
    env = EuchreEnv(stick_the_dealer=False, seed=5)
    env.reset(seed=5)
    s = env.state
    for _ in range(NUM_PLAYERS):
        env.step(A.PASS)  # round 1 all pass
    assert s.phase == Phase.BID_ROUND_2
    for _ in range(NUM_PLAYERS):
        env.step(A.PASS)  # round 2 all pass -> redeal
    # After a redeal we are back at the start of a fresh hand.
    assert env.state.phase == Phase.BID_ROUND_1
    assert env.state.trump is None


def test_going_alone_skips_partner():
    env = EuchreEnv(seed=7)
    env.reset(seed=7)
    s = env.state
    # First bidder orders up and goes alone.
    assert A.ORDER_UP_ALONE in env.legal_actions()
    maker = s.current_player
    env.step(A.ORDER_UP_ALONE)
    assert s.alone
    assert s.sitting_out == (maker + 2) % 4
    # Dealer discards, then play begins; the sitting-out partner never acts.
    env.step(env.legal_actions()[0])  # dealer discard
    assert s.phase == Phase.PLAY
    seen_seats = set()
    result = None
    while s.phase != Phase.DONE:
        seen_seats.add(s.current_player)
        result = env.step(env.legal_actions()[0])
        if result.terminated:
            break
    assert s.sitting_out not in seen_seats


def test_allow_going_alone_false_masks_alone_actions():
    env = EuchreEnv(allow_going_alone=False, seed=2)
    env.reset(seed=2)
    assert A.ORDER_UP_ALONE not in env.legal_actions()


def test_baselines_play_full_hands():
    env = EuchreEnv(seed=11)
    for agents in (RandomAgent(seed=1), HeuristicAgent()):
        trajectories, info = play_episode(env, agents, seed=11)
        assert sum(info["tricks_won"]) == 5
        # At least the four active seats produced decisions (3 if someone alone).
        nonempty = [t for t in trajectories.values() if t]
        assert len(nonempty) >= 3


def test_heuristic_beats_random_over_many_hands():
    from euchre_bot.training.evaluate import evaluate

    report = evaluate(HeuristicAgent(), RandomAgent(seed=0), num_hands=300, seed=0)
    # A bot that counts bowers and wins cheaply should beat random handily.
    assert report.win_rate > 0.55
