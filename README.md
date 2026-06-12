# euchre-bot

A reinforcement-learning playground built around [Euchre](https://en.wikipedia.org/wiki/Euchre).
The game engine, the multi-agent environment, baseline opponents, the self-play
training loop, evaluation, and experiment logging are all built and tested. The
**learning algorithms themselves are left as documented stubs for you to
implement** — that's the learning project.

Variant: standard 4-player Euchre with **going-alone (loners)** and
**stick-the-dealer**. See [`docs/euchre_rules.md`](docs/euchre_rules.md).

## Layout

```
src/euchre_bot/
  engine/      pure Euchre rules — cards, bowers, tricks, scoring (no torch, no numpy)
  env/         turn-based multi-agent EuchreEnv + observation/action encoding
  agents/      the Agent interface + Random and Heuristic baselines
  algos/       REINFORCE / A2C / PPO / DQN  <-- YOU IMPLEMENT THESE (stubs w/ guidance)
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
pytest                                   # 50 tests, ~0.2s
python -m euchre_bot.play                # play a hand vs the heuristic bots
python examples/train.py --agent heuristic --iterations 20   # exercise the full pipeline
mlflow ui                                # browse runs at http://localhost:5000
```

The baselines have a no-op `learn`, so those training curves are flat on purpose
— a reference for "what no learning looks like". Once you implement an agent,
`python examples/train.py --agent ppo` trains it through the same loop.

## Your task: implement the algorithms

`src/euchre_bot/algos/reinforce.py` is **fully implemented as a worked reference**
— read and run it first. `a2c.py`, `ppo.py`, and `dqn.py` are stubs that reuse
its skeleton; each spells out the math, the update recipe, the Euchre-specific
masking gotchas, and `# TODO(you)` markers. Start with
[`docs/implementing_algorithms.md`](docs/implementing_algorithms.md). Suggested
order: **REINFORCE (done) → A2C → PPO → DQN**.

```bash
# train the reference REINFORCE against a fixed opponent (clean learning signal)
python examples/train.py --agent reinforce --opponent random --iterations 400
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
| [`docs/implementing_algorithms.md`](docs/implementing_algorithms.md) | **Step-by-step guide to the part you build** |
| [`docs/monitoring.md`](docs/monitoring.md) | Observing/analyzing runs with MLflow (install, log, compare, registry) |
| [`docs/rl_best_practices.md`](docs/rl_best_practices.md) | General practices for writing RL projects |
| [`docs/roadmap.md`](docs/roadmap.md) | Future improvements (PIMC final boss, CFR, hybrid RL, variants) |
