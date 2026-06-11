"""Play a hand of Euchre from the terminal against three bots.

Run with::

    python -m euchre_bot.play                 # you are seat 0, vs HeuristicAgent
    python -m euchre_bot.play --opponent random

This is a debugging / sanity tool, not part of training. It is also the easiest
way to convince yourself the rules engine behaves the way your house rules do.
"""

from __future__ import annotations

import argparse

from .agents import HeuristicAgent, RandomAgent
from .agents.base import Agent
from .engine import actions as A
from .engine.state import PlayerView
from .env.euchre_env import EuchreEnv

HUMAN_SEAT = 0


class HumanAgent(Agent):
    """Prompts you to choose from the legal actions at the terminal."""

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        view: PlayerView = info["player_view"]
        print(f"\n--- your turn (seat {view.seat}) | phase {view.phase.name} ---")
        print(f"trump: {view.trump.name if view.trump else '-'}  "
              f"up-card: {view.upcard if view.upcard else '-'}  "
              f"dealer: {view.dealer}")
        if view.current_trick:
            print("trick so far: " + ", ".join(f"seat{ s}:{c}" for s, c in view.current_trick))
        print("your hand: " + ", ".join(str(c) for c in view.hand))
        legal = view.legal_actions
        for i, action in enumerate(legal):
            print(f"  [{i}] {A.action_name(action)}")
        while True:
            raw = input("choose> ").strip()
            if raw.isdigit() and 0 <= int(raw) < len(legal):
                return legal[int(raw)]
            print("invalid choice")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Euchre against bots.")
    parser.add_argument("--opponent", choices=["heuristic", "random"], default="heuristic")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    bot: Agent = HeuristicAgent() if args.opponent == "heuristic" else RandomAgent(seed=args.seed)
    seating = [HumanAgent(), bot, bot, bot]

    env = EuchreEnv(seed=args.seed)
    result = env.reset(seed=args.seed)
    while not result.terminated:
        seat = result.current_player
        if seat != HUMAN_SEAT:
            action = bot.act(
                result.observation, result.action_mask, result.info, deterministic=True
            )
            print(f"seat {seat} -> {A.action_name(action)}")
        else:
            action = seating[seat].act(result.observation, result.action_mask, result.info)
        result = env.step(action)

    info = result.info
    your_team = HUMAN_SEAT % 2
    outcome = "won" if info["scoring_team"] == your_team else "lost"
    print(f"\n=== hand over: your team {outcome} | "
          f"trump={info['trump'].name} maker=seat{info['maker']} alone={info['alone']} "
          f"| points={info['points']} to team {info['scoring_team']} "
          f"| tricks {info['tricks_won']} ===")


if __name__ == "__main__":
    main()
