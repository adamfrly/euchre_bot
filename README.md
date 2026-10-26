# euchre-bot

A reinforcement-learning playground built around [Euchre](https://en.wikipedia.org/wiki/Euchre).
The game engine, the multi-agent environment, baseline opponents, the self-play
training loop, evaluation, experiment logging, and four learning algorithms
(REINFORCE, A2C, PPO, DQN) are all built and tested.

Variant: standard 4-player Euchre with **going-alone (loners)** and
**stick-the-dealer**. See [`docs/euchre_rules.md`](docs/euchre_rules.md).

## Layout

```
src/euchre_bot/
  engine/      pure Euchre rules — cards, bowers, tricks, scoring (no torch, no numpy)
  env/         turn-based multi-agent EuchreEnv + observation/action encoding
  agents/      the Agent interface + Random and Heuristic baselines
  algos/       REINFORCE / A2C / PPO / DQN learning agents (torch)
  training/    self-play loop, evaluation harness, experiment logger
  play.py      play a hand against the bots from your terminal
examples/      runnable training script with logging + periodic evaluation
tests/         pytest suite for the rules and the environment
docs/          rules, environment API, the implementation guide, monitoring, best practices
```

The dependency arrow points one way: **only `algos` needs deep learning.** You
can run the rules, the environment, and the baselines with just numpy.

## Setup

```bash
# core (engine + env + baselines): numpy only
pip install -e .

# add the deep-learning stack when you start on the algorithms
pip install -e ".[learn]"      # torch

# add experiment monitoring (MLflow); optional — see docs/monitoring.md
pip install -e ".[monitor]"    # mlflow

# dev tools (tests + linter)
pip install -e ".[dev]"
```

(Examples above use `pip`; `uv venv && uv pip install -e ".[dev,learn,monitor]"` works too.)

## Try it now

```bash
pytest                                   # 82 tests, ~3s
python -m euchre_bot.play                # play a hand vs the heuristic bots
python examples/train.py --agent heuristic --iterations 20   # exercise the full pipeline
mlflow ui                                # browse runs at http://localhost:5000
```

The baselines have a no-op `learn`, so those training curves are flat on purpose
— a reference for "what no learning looks like". The learning agents train
through the same loop, e.g. `python examples/train.py --agent ppo`.

## Training the learning agents

`src/euchre_bot/algos/` holds four agents, easiest to hardest:
**REINFORCE → A2C → PPO → DQN**. `reinforce.py` is the most heavily commented
and the best place to start reading; `a2c.py` adds a value baseline (its
`ActorCritic` network is shared with PPO), `ppo.py` adds clipped multi-epoch
updates and GAE, and `dqn.py` is the off-policy, value-based alternative
(Double DQN). [`docs/implementing_algorithms.md`](docs/implementing_algorithms.md)
walks through the design.

```bash
# train the reference REINFORCE against a fixed opponent (clean learning signal)
python examples/train.py --agent reinforce --opponent random --iterations 400

# train PPO directly against the heuristic bot; checkpoints land in runs/<run-name>/
python examples/train.py --agent ppo --opponent heuristic --iterations 5000 \
    --eval-every 50 --eval-hands 1000 --run-name ppo_vs_heuristic
```

The most robust setup so far is **league training**. Each hand, the learner
faces itself, a frozen snapshot of an earlier version, or a heuristic variant.
Score it against a previous best checkpoint like this:

```bash
python examples/train.py --agent ppo --opponent league --random-scores \
    --champion ppo=runs/ppo_vs_self_v1/best.pt --iterations 6000 \
    --eval-every 50 --eval-hands 1000 --run-name ppo_league
```

Every evaluation runs a gauntlet: random, the heuristic at bidding thresholds
3-6, and the optional champion. Full games to 10 points are played every
`--game-eval-every` iterations. `--lr`, `--hidden-dim`, `--entropy-coef`,
`--epochs`, and `--alone-explore` override agent defaults. Each run saves
`latest.pt` and `best.pt` (best `eval_gauntlet/point_diff`) next to its
`metrics.jsonl`.

To see *how* a checkpoint plays, use the analysis script. It reports bidding
aggressiveness by hand strength, euchre rates, and agreement with the
heuristic's choices:

```bash
python examples/analyze_play.py --checkpoint ppo=runs/ppo_league/best.pt
```

You'll know it's working when `eval_vs_random/win_rate` climbs above 0.5 and
then `eval_vs_heuristic/win_rate` follows. **Judge progress by evaluation against
the fixed baselines, never by self-play reward or training loss** — see
[`docs/monitoring.md`](docs/monitoring.md).

## Documentation

| Doc | What's in it |
|---|---|
| [`docs/euchre_rules.md`](docs/euchre_rules.md) | The exact variant and scoring implemented |
| [`docs/environment.md`](docs/environment.md) | Env API, observation vector, action space, rewards |
| [`docs/implementing_algorithms.md`](docs/implementing_algorithms.md) | How the learning agents are built, step by step |
| [`docs/monitoring.md`](docs/monitoring.md) | Observing/analyzing runs with MLflow (install, log, compare, registry) |
| [`docs/rl_best_practices.md`](docs/rl_best_practices.md) | General practices for writing RL projects |
| [`docs/experiments.md`](docs/experiments.md) | Results and findings from the training runs so far |
| [`docs/roadmap.md`](docs/roadmap.md) | **Prioritised next steps**, plus longer-term ideas (PIMC, CFR, hybrid RL) |
