"""Game scores in the environment and full-game evaluation."""

import numpy as np

from euchre_bot.agents import HeuristicAgent, RandomAgent
from euchre_bot.env.euchre_env import EuchreEnv
from euchre_bot.training import evaluate, evaluate_games
from euchre_bot.training.self_play import play_episode


class _Recorder(RandomAgent):
    def __init__(self):
        super().__init__(seed=0)
        self.last_obs = None

    def act(self, observation, action_mask, info=None, *, deterministic=False):
        self.last_obs = observation
        return super().act(observation, action_mask, info, deterministic=deterministic)


def test_reset_scores_reach_the_observation():
    env = EuchreEnv(seed=0)
    recorder = _Recorder()
    play_episode(env, recorder, scores=[7, 3])
    assert np.allclose(recorder.last_obs[-2:], [0.7, 0.3]) or np.allclose(
        recorder.last_obs[-2:], [0.3, 0.7]
    )
    env = EuchreEnv(seed=0, random_scores=True)
    seen = set()
    for _ in range(20):
        result = env.reset()
        seen.add(tuple(result.observation[-2:]))
    assert len(seen) > 1


def test_games_end_at_ten_and_tighter_heuristic_is_stronger():
    report = evaluate_games(HeuristicAgent(), HeuristicAgent(call_threshold=5), 100, seed=1)
    assert report.games == 100 and report.avg_hands >= 3
    assert report.win_rate < 0.5
    hands = evaluate(HeuristicAgent(), HeuristicAgent(call_threshold=5), 1000, seed=1)
    assert hands.avg_point_diff < 0
