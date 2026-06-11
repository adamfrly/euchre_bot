# Monitoring, observing, and analyzing RL runs

RL fails *silently*. Supervised learning tells you it's broken — the loss
diverges, accuracy flatlines at chance. An RL run will happily report a smooth,
decreasing loss while the agent learns nothing useful, because the loss is a
moving target computed against the agent's own changing behaviour. **The loss is
not your scoreboard.** This page is about building the instruments that tell you
what's actually happening.

## The golden rule

> Log scalar metrics every iteration, and judge progress by **evaluation against
> a fixed opponent** — never by the training loss and never by self-play reward.

Self-play reward sits near zero by construction (you beat yourself half the
time). The only honest signal is: *how does the current policy do against a
frozen reference?* That's why `examples/train.py` evaluates against both
`RandomAgent` and `HeuristicAgent` on a schedule and logs `eval_*/win_rate`.

## The tooling in this repo

`euchre_bot.training.MetricsLogger` fans every scalar to three places:

- **stdout** — immediate, so you notice trouble in the first minute;
- **`runs/<name>/metrics.jsonl`** — dependency-free; load with pandas for custom
  plots, statistical tests, or comparing runs;
- **TensorBoard** — if installed, live zoomable curves.

```python
from euchre_bot.training import MetricsLogger
with MetricsLogger(run_name="ppo_v1") as logger:
    for it in range(iters):
        metrics = agent.learn(batch)          # {"policy_loss":..., "entropy":...}
        logger.log(it, metrics, prefix="train/")
        if it % 25 == 0:
            r = evaluate(agent, HeuristicAgent())
            logger.log(it, {"win_rate": r.win_rate}, prefix="eval/")
```

### TensorBoard

The de-facto standard, and a good fit here. Install it (`pip install
tensorboard`, included in the `learn` extra), then:

```
tensorboard --logdir runs
```

and open <http://localhost:6006>. Each scalar name becomes a live plot; runs in
separate `runs/<name>` folders overlay so you can compare hyperparameters
directly. Group names with a `prefix/` (e.g. `train/`, `eval_vs_heuristic/`) and
TensorBoard nests them for you. If TensorBoard isn't installed, `MetricsLogger`
degrades gracefully to stdout + JSONL — nothing breaks.

### Alternatives, briefly

- **Weights & Biases (wandb)** — hosted dashboards, effortless run comparison,
  hyperparameter sweeps. The `log(step, metrics)` shape here ports to
  `wandb.log(metrics, step=step)` almost verbatim. Worth it once you're running
  many experiments.
- **Plain matplotlib over `metrics.jsonl`** — for a publication-quality figure
  or a custom analysis TensorBoard can't express. The JSONL is one flat record
  per line: `pandas.read_json(path, lines=True)` and plot.
- **CSV + a spreadsheet** — never underestimate this for a quick eyeball.

## What to log (and why)

| Metric | Why you watch it |
|---|---|
| `eval/win_rate` vs random, vs heuristic | **The real scoreboard.** Up and to the right = learning. |
| `eval/point_diff` | Finer-grained than win-rate; moves before win-rate does. |
| `train/policy_loss` | Sanity, not success. Should be finite and not exploding. |
| `train/value_loss` (actor-critic) | Critic learning to predict returns; should fall then plateau. |
| `train/entropy` | **Exploration health.** High early, decaying smoothly. A crash to ~0 = premature collapse. |
| `train/approx_kl` (PPO) | Policy step size. Spikes ⇒ too-large updates; tune clip/lr. |
| `train/mean_return` / `mean_advantage` | Distribution of the learning signal; near-constant ⇒ nothing to learn from. |
| grad norm | Catches exploding/vanishing gradients before they wreck a run. |
| episode length / actions-per-hand | Cheap structural sanity check. |

Log **distributions**, not just means, once basics work — TensorBoard
histograms of advantages or action probabilities reveal collapse that a mean
hides.

## Reading the curves: a failure-mode field guide

- **Win-rate flat at ~0.5 vs random, loss looks fine.** Almost always a masking
  bug or a returns sign error. Verify `act` never samples an illegal action and
  that higher return ⇒ higher action probability.
- **Entropy collapses to ~0 in the first few iterations.** Premature
  convergence to a degenerate policy (e.g. always pass). Raise the entropy
  coefficient; lower the learning rate.
- **Win-rate oscillates / saw-tooths in self-play.** Policy chasing its own
  tail against a non-stationary opponent. Play against a pool of frozen past
  checkpoints sometimes; reduce step size.
- **PPO `approx_kl` spikes, performance craters.** Updates too aggressive — the
  clip isn't holding. Lower lr, fewer epochs, smaller batches.
- **Everything is noisy and unreproducible.** Seed the env, the agent's RNG, and
  torch. Evaluate over *many* hands (500+) — a few hands is mostly variance.

## Make runs reproducible

Set seeds (`EuchreEnv(seed=...)`, your agent's RNG, `torch.manual_seed`), record
the hyperparameters in the run name or a small JSON dropped next to
`metrics.jsonl`, and evaluate with enough hands that the number is signal, not
noise. A result you can't reproduce can't be debugged.

## Cheap habits that pay off

- Evaluate against **both** baselines so you see the agent surpass random long
  before it troubles the heuristic — early progress you'd otherwise miss.
- Keep a `runs/` folder per idea and let TensorBoard overlay them; comparing is
  where the learning happens.
- Watch the first 60 seconds of every run. Most broken runs are visibly broken
  immediately; don't wait an hour to discover it.
