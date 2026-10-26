"""Profile *how* trained agents play, not just how often they win.

Plays each agent (team 0) against the HeuristicAgent (team 1) and reports:

* bidding: how often the agent's team names trump, in which round, alone or not,
  and the call rate binned by hand strength (the heuristic's own strength score,
  which calls at >= 3) -- i.e. how aggressive the learned bidding is;
* outcomes: make / march / euchred rates as makers, and euchre rate as defenders;
* decision agreement: per decision type, how often the agent picks the same
  action the heuristic would have picked in the identical position;
* a few play habits: overtaking a partner who is already winning the trick, and
  leading trump as the making team.

Run::

    python examples/analyze_play.py --hands 3000 \\
        --checkpoint ppo=runs/ppo_vs_heuristic_v1/best.pt \\
        --checkpoint dqn=runs/dqn_vs_heuristic_v1/best.pt
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict

from euchre_bot.agents import HeuristicAgent
from euchre_bot.agents.base import Agent
from euchre_bot.agents.heuristic_agent import _CALL_THRESHOLD, _suit_strength
from euchre_bot.engine import actions as A
from euchre_bot.engine.cards import Suit, is_trump, trick_value
from euchre_bot.engine.rules import partner_of, trick_winner
from euchre_bot.engine.state import Phase
from euchre_bot.env.euchre_env import EuchreEnv

AGENT_TEAM = 0


def _decision_kind(view) -> str:
    if view.phase == Phase.BID_ROUND_1:
        return "bid_r1"
    if view.phase == Phase.BID_ROUND_2:
        return "bid_r2" if A.PASS in view.legal_actions else "bid_r2_stuck"
    if view.phase == Phase.DISCARD:
        return "discard"
    return "lead" if not view.current_trick else "follow"


def _best_strength(view) -> int:
    """Hand strength for the suit this decision is really about."""
    if view.phase == Phase.BID_ROUND_1:
        hand = list(view.hand) + ([view.upcard] if view.seat == view.dealer else [])
        return _suit_strength(hand, view.upcard.suit)
    # The turned-down card is hidden in round 2; the legal calls say which suits remain.
    return max(_suit_strength(view.hand, s) for s in Suit
               if A.call_suit_action(s, alone=False) in view.legal_actions)


def profile(agent: Agent, hands: int, seed: int) -> dict:
    env = EuchreEnv(seed=seed)
    heuristic = HeuristicAgent()
    seating = [agent, heuristic, agent, heuristic]

    agree = defaultdict(lambda: [0, 0])          # kind -> [agreements, decisions]
    calls_by_strength = defaultdict(lambda: [0, 0])  # strength -> [calls, chances]
    habits = Counter()
    outcomes = Counter()

    for _ in range(hands):
        result = env.reset()
        while not result.terminated:
            seat = result.current_player
            obs, mask, info = result.observation, result.action_mask, result.info
            action = seating[seat].act(obs, mask, info, deterministic=True)

            if seat % 2 == AGENT_TEAM:
                view = info["player_view"]
                kind = _decision_kind(view)
                h_action = heuristic.act(obs, mask, info)
                agree[kind][0] += int(action == h_action)
                agree[kind][1] += 1

                if kind in ("bid_r1", "bid_r2"):
                    strength = min(_best_strength(view), 8)
                    calls_by_strength[strength][0] += int(action != A.PASS)
                    calls_by_strength[strength][1] += 1
                if kind == "follow":
                    winner = trick_winner(view.current_trick, view.trump)
                    if winner == partner_of(view.seat):
                        card = A.card_of_action(action)
                        win_card = next(c for s, c in view.current_trick if s == winner)
                        led = view.led_suit
                        habits["partner_winning"] += 1
                        if trick_value(card, view.trump, led) > trick_value(
                            win_card, view.trump, led
                        ):
                            habits["overtook_partner"] += 1
                if kind == "lead" and view.maker is not None and view.maker % 2 == AGENT_TEAM:
                    habits["maker_leads"] += 1
                    if is_trump(A.card_of_action(action), view.trump):
                        habits["maker_led_trump"] += 1
            result = env.step(action)

        info = result.info
        maker_team = info["maker"] % 2
        maker_tricks = info["tricks_won"][maker_team]
        role = "make" if maker_team == AGENT_TEAM else "defend"
        outcomes[f"{role}_hands"] += 1
        if maker_tricks < 3:
            outcomes[f"{role}_euchred"] += 1
        elif maker_tricks == 5:
            outcomes[f"{role}_march"] += 1
        if info["alone"]:
            outcomes[f"{role}_alone"] += 1
        sign = 1 if info["scoring_team"] == AGENT_TEAM else -1
        outcomes[f"{role}_points"] += sign * info["points"]
        outcomes["points"] += sign * info["points"]

    return {"agree": agree, "calls": calls_by_strength, "habits": habits,
            "outcomes": outcomes, "hands": hands}


def _pct(num: float, den: float) -> str:
    return f"{100 * num / den:5.1f}%" if den else "   n/a"


def report(name: str, p: dict) -> None:
    o, h, n = p["outcomes"], p["habits"], p["hands"]
    print(f"\n=== {name}  ({n} hands vs heuristic) ===")
    print(f"points/hand {o['points'] / n:+.3f}")
    mk, df = o["make_hands"], o["defend_hands"]
    print(f"makes trump {_pct(mk, n)} of hands | alone {_pct(o['make_alone'], mk)} of makes")
    print(f"  as maker:    euchred {_pct(o['make_euchred'], mk)}  march {_pct(o['make_march'], mk)}"
          f"  pts/hand {o['make_points'] / max(mk, 1):+.3f}")
    print(f"  as defender: euchre them {_pct(o['defend_euchred'], df)}"
          f"  they march {_pct(o['defend_march'], df)}"
          f"  pts/hand {o['defend_points'] / max(df, 1):+.3f}")
    print(f"call rate by hand strength (heuristic calls at >= {_CALL_THRESHOLD}):")
    print("  " + "  ".join(
        f"{s}{'+' if s == 8 else ''}:{_pct(c, t).strip()}"
        for s, (c, t) in sorted(p["calls"].items()) if t >= 20
    ))
    print("agreement with heuristic's choice:  " + "  ".join(
        f"{k} {_pct(a, t).strip()} (n={t})" for k, (a, t) in sorted(p["agree"].items())
    ))
    print(f"overtakes a winning partner {_pct(h['overtook_partner'], h['partner_winning'])}"
          f" | leads trump as maker {_pct(h['maker_led_trump'], h['maker_leads'])}")


def _load(spec: str) -> tuple[str, Agent]:
    name, path = spec.split("=", 1)
    algo = name.split("_")[0]
    from euchre_bot.algos import A2CAgent, DQNAgent, PPOAgent, ReinforceAgent

    cls = {"reinforce": ReinforceAgent, "a2c": A2CAgent, "ppo": PPOAgent, "dqn": DQNAgent}[algo]
    agent = cls()
    agent.load(path)
    return name, agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[],
                        help="name=path, where name starts with the algo (e.g. ppo_self=...)")
    parser.add_argument("--hands", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    report("heuristic (baseline)", profile(HeuristicAgent(), args.hands, args.seed))
    for spec in args.checkpoint:
        name, agent = _load(spec)
        report(name, profile(agent, args.hands, args.seed))


if __name__ == "__main__":
    main()
