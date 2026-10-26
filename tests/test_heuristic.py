"""HeuristicAgent behaviour across bidding thresholds."""

from euchre_bot.agents import HeuristicAgent
from euchre_bot.env.euchre_env import EuchreEnv
from euchre_bot.training.self_play import play_episode


def test_heuristic_variants_only_play_legal_actions():
    # Tight bidders pass suits in round 1 that they'd like to call in round 2,
    # after the up-card is hidden; they must never name the turned-down suit.
    env = EuchreEnv(seed=3)
    for threshold in (3, 4, 5, 6, 8):
        bot = HeuristicAgent(call_threshold=threshold)
        for _ in range(300):
            play_episode(env, bot)  # the env raises on any illegal action
