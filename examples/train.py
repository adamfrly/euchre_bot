"""End-to-end training harness: collect self-play, learn, log, and evaluate.

This script is the template you will actually run. Out of the box it works with
the *baseline* agents so you can exercise the full pipeline (and see MLflow
populate) before any algorithm is written -- their ``learn`` is a no-op, so the
curves are flat, which is itself a useful "this is what no learning looks like"
reference. Once you implement an agent in ``euchre_bot.algos``, pass its name and
the same loop trains it for real.

Run::

    python examples/train.py --agent heuristic --iterations 50
    python examples/train.py --agent ppo --iterations 2000 --eval-every 25

Then, in another terminal, browse the runs (serves ./mlruns at :5000)::

    mlflow ui
"""

from __future__ import annotations

import argparse

from euchre_bot.agents import HeuristicAgent, RandomAgent
from euchre_bot.agents.base import Agent
from euchre_bot.env.euchre_env import EuchreEnv
from euchre_bot.training import MetricsLogger, collect_batch, evaluate


def make_agent(name: str) -> Agent:
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent()
    # The learning agents live in euchre_bot.algos and require torch. They start
    # as NotImplementedError stubs -- implement one, then select it here.
    if name == "reinforce":
        from euchre_bot.algos import ReinforceAgent

        return ReinforceAgent()
    if name == "a2c":
        from euchre_bot.algos import A2CAgent

        return A2CAgent()
    if name == "ppo":
        from euchre_bot.algos import PPOAgent

        return PPOAgent()
    if name == "dqn":
        from euchre_bot.algos import DQNAgent

        return DQNAgent()
    raise ValueError(f"unknown agent {name!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="heuristic",
                        choices=["random", "heuristic", "reinforce", "a2c", "ppo", "dqn"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes-per-iter", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-hands", type=int, default=500)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = EuchreEnv(seed=args.seed)
    learner = make_agent(args.agent)

    # Fixed reference opponents -- the only honest measure of progress. We track
    # both so you can see the agent first surpass random, then close on heuristic.
    references = {"vs_random": RandomAgent(seed=123), "vs_heuristic": HeuristicAgent()}

    # Everything you'd want to compare runs by later. Logged once as MLflow params.
    params = {
        "agent": args.agent,
        "iterations": args.iterations,
        "episodes_per_iter": args.episodes_per_iter,
        "eval_hands": args.eval_hands,
        "seed": args.seed,
    }

    with MetricsLogger(
        experiment="euchre-bot",
        run_name=args.run_name or args.agent,
        params=params,
    ) as logger:
        for iteration in range(1, args.iterations + 1):
            batch = collect_batch(env, learner, args.episodes_per_iter)
            metrics = learner.learn(batch)  # no-op for baselines; real for your algos
            logger.log(iteration, metrics, prefix="train/")

            if iteration % args.eval_every == 0 or iteration == args.iterations:
                for tag, opponent in references.items():
                    report = evaluate(learner, opponent, num_hands=args.eval_hands, seed=args.seed)
                    logger.log(iteration, {
                        "win_rate": report.win_rate,
                        "point_diff": report.avg_point_diff,
                    }, prefix=f"eval_{tag}/")


if __name__ == "__main__":
    main()
