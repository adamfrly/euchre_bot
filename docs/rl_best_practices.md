# Best practices for writing RL projects

General lessons, learned the hard way by many people, organised so you can apply
them to this project and the next one. Monitoring has its own page
(`docs/monitoring.md`); this is everything else.

## Architecture: separate the three concerns

Keep these layers apart, with dependencies pointing one way:

```
environment / rules   (no RL, no deep-learning framework)
        ^
        |  observations + rewards
        |
   algorithm           (torch lives here, and only here)
        ^
        |  act / learn
        |
training & evaluation  (framework-agnostic orchestration)
```

This repo enforces it: `engine`, `env`, and `agents` (baselines) have **no torch
dependency** — only `algos` does. The payoff: you can test the game, run
baselines, and profile the environment without a GPU stack; you can swap
algorithms without touching the env; and bugs have an obvious home. A tangled
env-that-imports-your-network is the most common way RL codebases rot.

## Get the environment right *first*

More RL "algorithm bugs" are actually environment bugs. Before you train
anything:

- **Unit-test the rules.** Trick resolution, follow-suit legality, scoring, and
  the bower edge cases each have a test here. A wrong comparison corrupts every
  episode invisibly.
- **Make illegal actions impossible, not merely penalised.** This env raises on
  an illegal action and exposes an action mask. Masking beats reward penalties:
  it shrinks the effective action space and removes a whole class of nonsense
  the agent would otherwise have to learn to avoid.
- **Sanity-check with a random agent.** A full random playthrough must always
  terminate with a valid score. If random rollouts crash or hang, fix that
  before adding a neural network.
- **Know your reward.** Sparse vs dense, zero-sum vs not, per-hand vs per-game —
  these change what the agent optimises. This env is sparse + zero-sum per hand;
  that's a deliberate, documented choice, not an accident.

## Observation and action design

- **Encode relative to the actor.** Positions here are "seats clockwise from
  me", so one policy plays all four seats. Absolute indices would force the
  network to learn four separate strategies.
- **One source of truth for dimensions.** `OBS_DIM` and `NUM_ACTIONS` are
  computed/defined once and imported. Hard-coding `174` in your network is how
  you get silent shape mismatches after a feature change.
- **Give the network what a good player would want:** your hand, public memory
  of played cards, trump, the current trick, table geometry. Don't leak hidden
  information (opponents' hands) — that trains a policy that can't exist at test
  time.

## Algorithms: walk before you run

- **Implement in order of difficulty:** REINFORCE → A2C → PPO → DQN. Each adds
  one idea to the last. Skipping to PPO first means debugging four new things at
  once.
- **Reproduce on a toy first if stuck.** If your PPO won't learn Euchre, point
  the same class at CartPole. If it can't solve CartPole, the bug is in your
  algorithm; if it can, the bug is in your env interface or encoding.
- **Start from known-good hyperparameters** and change one at a time. PPO in
  particular has well-published defaults (clip 0.2, γ 0.99, λ 0.95, lr 3e-4).
  Random hyperparameter flailing is indistinguishable from a real bug.
- **Normalise the learning signal.** Standardise returns/advantages (zero mean,
  unit std) per batch — it's the single highest-leverage variance reduction.
- **Keep an entropy bonus** for policy-gradient methods so exploration doesn't
  die, especially with a mostly-masked action space.

## Evaluation and reproducibility

- **Evaluate against fixed references, deterministically, over many hands.**
  Self-play score is not progress. 500+ hands so the number is signal.
- **Seed everything** (env, agent RNG, torch) and record hyperparameters with
  each run. Reproducibility is a prerequisite for debugging, not a nicety.
- **Checkpoint** so a long run isn't lost and so you can build a pool of past
  opponents for self-play stability.

## Engineering hygiene

- **Pin dependencies** and split optional ones (the `learn` extra here keeps
  torch out of the core). A reproducible environment is part of a reproducible
  experiment.
- **Tests + linter in CI.** Cheap, and they catch the dumb regressions that eat
  afternoons.
- **Small, pure functions** for anything you can test without a network —
  they're where bugs hide and where tests pay off most.
- **Profile before optimising.** In RL the bottleneck is often environment
  stepping or Python overhead, not the GPU. Measure first.

## The mindset

RL debugging is detective work because feedback is delayed, noisy, and
non-stationary. Change one variable at a time, keep a fixed yardstick, watch
your instruments from second one, and trust evaluation curves over your
intuition. Most "the algorithm is broken" moments are a masking bug, a sign
error, or an environment that doesn't do what you think — check those three
before you touch the math.
