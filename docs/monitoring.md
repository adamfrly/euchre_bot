# Monitoring with MLflow: observing and analyzing RL runs

RL fails *silently*. Supervised learning tells you it's broken — the loss
diverges, accuracy flatlines. An RL run will happily report a smooth, decreasing
loss while the agent learns nothing useful, because the loss is a moving target
computed against the agent's own changing behaviour. **The loss is not your
scoreboard.** This page is about building the instruments that tell you what's
actually happening — and this project uses **MLflow** to do it.

## The golden rule (read this first, it outranks every tool)

> Log scalar metrics every iteration, and judge progress by **evaluation against
> a fixed opponent** — never by the training loss and never by self-play reward.

Self-play reward sits near zero by construction (you beat yourself half the
time). The only honest signal is: *how does the current policy do against a
frozen reference?* That's why `examples/train.py` evaluates against both
`RandomAgent` and `HeuristicAgent` on a schedule and logs `eval_*/win_rate`.

---

## Part 1 — Learning MLflow

### Why MLflow

MLflow is a free, open-source experiment-tracking tool. For this project its
appeal is:

- **Local and account-free.** By default it writes to a `./mlruns/` folder on
  your machine. No cloud, no login, no server to provision.
- **Runs keep everything together.** One *run* bundles the hyperparameters, the
  full metric history, and any files (checkpoints) — so three weeks from now you
  can answer "what lr did my best agent use?" in two clicks.
- **Run comparison is the headline feature.** Overlay metric curves from many
  runs and sort a table by `eval/win_rate` — exactly the workflow you need when
  tuning PPO.

### The four concepts you need

| Concept | What it is | In this project |
|---|---|---|
| **Experiment** | A named bucket of runs | `"euchre-bot"` |
| **Run** | One training attempt | one agent + hyperparameter config, e.g. `ppo_v1` |
| **Params** | Inputs you set, logged once | `agent`, `lr`, `gamma`, `clip_eps`, `seed` |
| **Metrics** | Numbers over time, logged per step | `train/policy_loss`, `eval_vs_heuristic/win_rate` |
| **Artifacts** | Files attached to a run | checkpoints (`*.pt`), configs, plots |

(There's a fifth, the **Model Registry**, for versioning your best agent —
covered at the end.)

### Install

```bash
pip install -e ".[monitor]"     # installs mlflow
# or just: pip install mlflow
```

Nothing else to start — the first time you log, MLflow creates `./mlruns/`.

### How this repo wires it: `MetricsLogger`

`euchre_bot.training.MetricsLogger` is a thin wrapper that drives the MLflow run
lifecycle for you (and also echoes to stdout and a `runs/<name>/metrics.jsonl`
mirror). The whole training loop in `examples/train.py` is:

```python
from euchre_bot.training import MetricsLogger, collect_batch, evaluate

with MetricsLogger(experiment="euchre-bot", run_name="ppo_v1",
                   params={"agent": "ppo", "lr": 3e-4, "gamma": 0.99}) as logger:
    for it in range(1, iterations + 1):
        batch = collect_batch(env, learner, episodes_per_iter)
        metrics = learner.learn(batch)                  # {"policy_loss":..., "entropy":...}
        logger.log(it, metrics, prefix="train/")        # -> mlflow.log_metrics(step=it)

        if it % 25 == 0:
            r = evaluate(learner, HeuristicAgent())
            logger.log(it, {"win_rate": r.win_rate}, prefix="eval_vs_heuristic/")
            learner.save("checkpoints/ppo.pt")
            logger.log_artifact("checkpoints/ppo.pt")   # checkpoint travels with the run
```

`MetricsLogger.__init__` calls `mlflow.set_experiment` + `mlflow.start_run` +
`mlflow.log_params`; `.log()` calls `mlflow.log_metrics(..., step=it)`;
`.log_artifact()` calls `mlflow.log_artifact`; and `__exit__` calls
`mlflow.end_run()`. If MLflow isn't installed it degrades to stdout + JSONL.

### The same thing in raw MLflow (so you know what's underneath)

You don't need this — `MetricsLogger` does it — but seeing the bare API makes the
wrapper transparent:

```python
import mlflow

mlflow.set_experiment("euchre-bot")                 # creates ./mlruns on first use
with mlflow.start_run(run_name="ppo_v1"):           # context manager ends the run for you
    mlflow.log_params({"agent": "ppo", "lr": 3e-4, "gamma": 0.99})
    for it in range(1, iterations + 1):
        ...
        mlflow.log_metric("train/policy_loss", loss, step=it)      # one scalar
        mlflow.log_metrics({"train/entropy": ent}, step=it)        # many at once
        if it % 25 == 0:
            mlflow.log_metric("eval_vs_heuristic/win_rate", wr, step=it)
            mlflow.log_artifact("checkpoints/ppo.pt")              # any file
```

Notes that trip people up:
- **`step=` is what makes a curve.** Omit it and you just overwrite a single
  value. Pass your iteration number.
- **Slashes in names are allowed** and become groups in the UI, so
  `eval_vs_heuristic/win_rate` is a feature, not a problem.
- **Params are write-once and stringified**; metrics are numeric and append by
  step. Don't try to log a changing value as a param.

### Browsing runs: the MLflow UI

Run it **from the project root** (so it finds the same store your training run
wrote to):

```bash
mlflow ui            # http://localhost:5000
```

Open it and:
1. Pick the **euchre-bot** experiment in the sidebar.
2. The **run table** lists every run with its params and latest metrics. Sort by
   `eval_vs_heuristic/win_rate` to find your best config instantly.
3. **Tick several runs → Compare** to overlay their metric curves and see a
   side-by-side param diff. This is where hyperparameter tuning actually
   happens.
4. Click a run to see its full metric history, params, and artifacts (your
   checkpoints are downloadable here).

### Where your data lives

**MLflow 3.x** (what you have) defaults to a local **SQLite** store: a
`mlflow.db` file in your working directory for run/metric data, plus an
`./mlartifacts/` directory for artifacts. (Older MLflow 2.x defaulted to a plain
`./mlruns/` file directory — you'll see that name in a lot of tutorials; same
idea, different backend.) All of these are already in `.gitignore`.

Because the store lives in the working directory, **launch `mlflow ui` from the
project root** so it reads the same `mlflow.db` your run wrote. To pin the
location explicitly (handy if you run from elsewhere):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To send runs somewhere specific, set a tracking URI before logging — e.g.
`MetricsLogger(tracking_uri="sqlite:///mlflow.db")` or
`mlflow.set_tracking_uri(...)`. The SQLite backend is also what unlocks the
**Model Registry** below.

### Saving and versioning your best agent (Model Registry)

Logging a checkpoint as an artifact is enough to reproduce a run. If you want a
named, versioned "current best Euchre bot" you can promote, use the registry:

```python
import mlflow
# After training, with a tracking URI that supports it (e.g. sqlite:///mlflow.db):
mlflow.pytorch.log_model(learner.policy, name="model",
                         registered_model_name="euchre-ppo")
# Later, load a specific version back:
policy = mlflow.pytorch.load_model("models:/euchre-ppo/3")
```

This is optional and only worth it once you have an agent good enough to keep.

### A note on `mlflow.autolog()`

MLflow can auto-capture metrics from some frameworks. For a hand-written RL loop
it won't know what your "win rate" is, so **explicit `log_metrics` is the right
call here** — autolog is for the supervised-training paths it understands.

---

## Part 2 — What to log and how to read it (tool-independent)

### What to log (and why)

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

### Reading the curves: a failure-mode field guide

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

### Make runs reproducible

Set seeds (`EuchreEnv(seed=...)`, your agent's RNG, `torch.manual_seed`), log the
hyperparameters as MLflow **params** (so they're attached to the run forever),
and evaluate with enough hands that the number is signal, not noise. A result you
can't reproduce can't be debugged — and "what config was that?" is exactly the
question MLflow params answer.

### Cheap habits that pay off

- Evaluate against **both** baselines so you see the agent surpass random long
  before it troubles the heuristic — early progress you'd otherwise miss.
- Give each run a meaningful `run_name` and let the MLflow compare view overlay
  them; comparing is where the learning happens.
- Watch the first 60 seconds of every run. Most broken runs are visibly broken
  immediately; don't wait an hour to discover it.

---

## Appendix — other free tools (for context)

You're using MLflow, but it helps to know the neighbours. **TensorBoard**
(local, free, native to PyTorch via `torch.utils.tensorboard`) is great for live
scalar/histogram curves but weaker at organizing many runs. **Aim** (local,
fully OSS) is a fast TensorBoard-style viewer with strong run comparison.
**Weights & Biases** (free for personal use, hosted) has the slickest dashboards
and built-in hyperparameter sweeps, at the cost of being a cloud service. All
three accept the same `log(step, metrics)` shape `MetricsLogger` already uses, so
switching later is a small change.
