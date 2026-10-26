"""End-to-end training harness: collect self-play, learn, log, and evaluate.

This script is the template you will actually run. Out of the box it works with
the *baseline* agents so you can exercise the full pipeline (and see MLflow
populate) before any algorithm is written -- their ``learn`` is a no-op, so the
curves are flat, which is itself a useful "this is what no learning looks like"
reference. Once you implement an agent in ``euchre_bot.algos``, pass its name and
the same loop trains it for real.

Run::

    python examples/train.py --agent heuristic --iterations 50
    python examples/train.py --agent ppo --opponent heuristic --iterations 2000 --eval-every 25

    # league self-play + going-alone exploration, benchmarked against a champion
    python examples/train.py --agent ppo --opponent league --random-scores \
        --alone-explore 0.05 --alone-explore-iters 3000 \
        --champion ppo=runs/ppo_vs_self_v1/best.pt --iterations 6000 --eval-every 50

Every evaluation runs a *gauntlet*: random, the heuristic at bidding thresholds
3-6, and the optional champion checkpoint. ``eval_gauntlet/point_diff`` (mean
points per hand over all but random) is the headline number and picks
``best.pt``; full games to 10 are played every ``--game-eval-every`` iterations.

Then, in another terminal, browse the runs (serves ./mlruns at :5000)::

    mlflow ui
"""

from __future__ import annotations

import argparse
from pathlib import Path

from euchre_bot.agents import HeuristicAgent, RandomAgent
from euchre_bot.agents.base import Agent
from euchre_bot.env.euchre_env import EuchreEnv
from euchre_bot.training import (
    MetricsLogger,
    OpponentPool,
    ScoreBlind,
    collect_batch,
    collect_league_batch,
    evaluate,
    evaluate_games,
)

#: Bidding thresholds for the heuristic variants used as league opponents and in
#: the evaluation gauntlet. 3 is the default HeuristicAgent.
HEURISTIC_THRESHOLDS = (3, 4, 5, 6)


def make_agent(name: str, **hparams) -> Agent:
    """Build an agent by name. ``hparams`` (lr, hidden_dim, ...) go to learning agents."""
    if name == "random":
        return RandomAgent()
    if name == "heuristic":
        return HeuristicAgent()
    # The learning agents live in euchre_bot.algos and require torch.
    from euchre_bot.algos import A2CAgent, DQNAgent, PPOAgent, ReinforceAgent

    classes = {"reinforce": ReinforceAgent, "a2c": A2CAgent, "ppo": PPOAgent, "dqn": DQNAgent}
    if name not in classes:
        raise ValueError(f"unknown agent {name!r}")
    return classes[name](**hparams)


def load_checkpoint(spec: str) -> Agent:
    """Load ``algo=path`` (e.g. ``ppo=runs/ppo_vs_self_v1/best.pt``) as a score-blind agent."""
    algo, path = spec.split("=", 1)
    agent = make_agent(algo)
    agent.load(path)
    # Older checkpoints never saw nonzero game scores; hide the score from them.
    return ScoreBlind(agent)


def heuristic_tag(threshold: int) -> str:
    return "vs_heuristic" if threshold == 3 else f"vs_heuristic_t{threshold}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", default="heuristic",
                        choices=["random", "heuristic", "reinforce", "a2c", "ppo", "dqn"])
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes-per-iter", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-hands", type=int, default=500)
    # Who the learner trains against. "self" is pure self-play (all four seats
    # share the policy). "league" draws an opponent per hand from the current
    # policy, frozen snapshots of it, and heuristic variants -- the most robust
    # option. For REINFORCE the self-play gradient is weak because the zero-sum
    # seats cancel -- a fixed "random"/"heuristic" opponent gives a cleaner signal.
    parser.add_argument("--opponent", choices=["self", "random", "heuristic", "league"],
                        default="self")
    parser.add_argument("--snapshot-every", type=int, default=100,
                        help="league only: iterations between frozen snapshots")
    parser.add_argument("--champion", default=None,
                        help="algo=path of a previous best checkpoint to add to the gauntlet")
    parser.add_argument("--game-eval-every", type=int, default=250,
                        help="iterations between full-game (to 10 points) evaluations")
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--random-scores", action="store_true",
                        help="show random game scores in training so score features get learned")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    # Hyperparameters for the learning agents; unset means "use the agent's default".
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None, help="a2c / ppo only")
    parser.add_argument("--epochs", type=int, default=None, help="ppo only")
    parser.add_argument("--alone-explore", type=float, default=None,
                        help="ppo only: initial probability mixed in for going alone")
    parser.add_argument("--alone-explore-iters", type=int, default=None,
                        help="ppo only: iterations over which --alone-explore decays to 0")
    # Small MLPs on single observations run fastest on one thread, and it keeps
    # several training runs from fighting over cores.
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()

    run_name = args.run_name or f"{args.agent}_vs_{args.opponent}"
    hparams = {
        k: v for k, v in {
            "lr": args.lr,
            "hidden_dim": args.hidden_dim,
            "entropy_coef": args.entropy_coef,
            "epochs": args.epochs,
            "alone_explore": args.alone_explore,
            "alone_explore_updates": args.alone_explore_iters,
        }.items() if v is not None
    }
    learning = args.agent not in ("random", "heuristic")
    if learning:
        import torch

        torch.set_num_threads(args.torch_threads)
        hparams["seed"] = args.seed

    env = EuchreEnv(seed=args.seed, random_scores=args.random_scores)
    learner = make_agent(args.agent, **hparams)
    checkpoint_dir = Path("runs") / run_name
    best_score = float("-inf")

    heuristics = {t: HeuristicAgent(call_threshold=t) for t in HEURISTIC_THRESHOLDS}
    train_opponent = None
    pool = None
    if args.opponent == "random":
        train_opponent = RandomAgent(seed=999)
    elif args.opponent == "heuristic":
        train_opponent = heuristics[3]
    elif args.opponent == "league":
        pool = OpponentPool(list(heuristics.values()), seed=args.seed)

    # Fixed reference opponents -- the only honest measure of progress. The
    # gauntlet score averages every reference except random.
    references: dict[str, Agent] = {"vs_random": RandomAgent(seed=123)}
    references.update({heuristic_tag(t): agent for t, agent in heuristics.items()})
    if args.champion:
        references["vs_champion"] = load_checkpoint(args.champion)
    game_references = {k: v for k, v in references.items()
                       if k in ("vs_heuristic", "vs_heuristic_t5", "vs_champion")}

    # Everything you'd want to compare runs by later. Logged once as MLflow params.
    params = {
        "agent": args.agent,
        "opponent": args.opponent,
        "iterations": args.iterations,
        "episodes_per_iter": args.episodes_per_iter,
        "eval_hands": args.eval_hands,
        "seed": args.seed,
        "random_scores": args.random_scores,
        "champion": args.champion,
        **hparams,
    }

    with MetricsLogger(experiment="euchre-bot", run_name=run_name, params=params) as logger:
        for iteration in range(1, args.iterations + 1):
            if pool is not None:
                if iteration % args.snapshot_every == 0:
                    pool.add_snapshot(learner)
                batch = collect_league_batch(env, learner, args.episodes_per_iter, pool)
            else:
                batch = collect_batch(env, learner, args.episodes_per_iter,
                                      opponent=train_opponent)
            metrics = learner.learn(batch)  # no-op for baselines
            logger.log(iteration, metrics, prefix="train/")

            last = iteration == args.iterations
            if iteration % args.eval_every == 0 or last:
                point_diffs = []
                for tag, opponent in references.items():
                    report = evaluate(learner, opponent, num_hands=args.eval_hands, seed=args.seed)
                    logger.log(iteration, {
                        "win_rate": report.win_rate,
                        "point_diff": report.avg_point_diff,
                    }, prefix=f"eval_{tag}/")
                    if tag != "vs_random":
                        point_diffs.append(report.avg_point_diff)
                score = sum(point_diffs) / len(point_diffs)
                logger.log(iteration, {"point_diff": score}, prefix="eval_gauntlet/")

                # Keep the checkpoint with the best gauntlet score, plus the latest.
                if learning:
                    learner.save(str(checkpoint_dir / "latest.pt"))
                    if score > best_score:
                        best_score = score
                        learner.save(str(checkpoint_dir / "best.pt"))

            if iteration % args.game_eval_every == 0 or last:
                for tag, opponent in game_references.items():
                    games = evaluate_games(learner, opponent, args.eval_games, seed=args.seed)
                    logger.log(iteration, {"win_rate": games.win_rate}, prefix=f"games_{tag}/")


if __name__ == "__main__":
    main()
