"""League training: the opponent pool, league batches, and ScoreBlind."""

import numpy as np

from euchre_bot.agents import HeuristicAgent, RandomAgent
from euchre_bot.env.euchre_env import EuchreEnv
from euchre_bot.training import OpponentPool, ScoreBlind, collect_league_batch
from euchre_bot.training.self_play import play_episode


class _Recorder(RandomAgent):
    def __init__(self):
        super().__init__(seed=0)
        self.last_obs = None

    def act(self, observation, action_mask, info=None, *, deterministic=False):
        self.last_obs = observation
        return super().act(observation, action_mask, info, deterministic=deterministic)


def test_pool_sampling_respects_probabilities():
    fixed = HeuristicAgent()
    pool = OpponentPool([fixed], p_self=0.5, p_snapshot=0.3, seed=0)
    # No snapshots yet: snapshot share falls through to the fixed bots.
    draws = [pool.sample() for _ in range(2000)]
    assert 0.45 < sum(d is None for d in draws) / 2000 < 0.55

    pool.add_snapshot(RandomAgent(seed=1))
    draws = [pool.sample() for _ in range(4000)]
    snap_share = sum(d is pool.snapshots[0] for d in draws) / 4000
    assert 0.25 < snap_share < 0.35


def test_snapshot_is_a_frozen_copy():
    learner = RandomAgent(seed=1)
    pool = OpponentPool([], seed=0)
    pool.add_snapshot(learner)
    assert pool.snapshots[0] is not learner


def test_league_batch_only_keeps_learner_seats_against_fixed_bots():
    env = EuchreEnv(seed=0)
    learner = RandomAgent(seed=2)
    vs_fixed = OpponentPool([HeuristicAgent()], p_self=0.0, p_snapshot=0.0, seed=0)
    vs_self = OpponentPool([], p_self=1.0, p_snapshot=0.0, seed=0)
    assert len(collect_league_batch(env, learner, 10, vs_fixed)) <= 20
    assert len(collect_league_batch(env, learner, 10, vs_self)) > 20


def test_score_blind_hides_score():
    recorder = _Recorder()
    play_episode(EuchreEnv(seed=0), ScoreBlind(recorder), scores=[7, 3])
    assert np.all(recorder.last_obs[-2:] == 0)
