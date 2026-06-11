# Implementing the learning agents (your part)

This is the guide for the work you reserved for yourself: filling in the agents
in `euchre_bot/algos/`. Everything else — the rules, the environment, the
baselines, the training loop, evaluation, and logging — is built so that the
*only* thing standing between you and a trained Euchre bot is these classes.

Read this once end to end, then keep `algos/reinforce.py` open beside it.

## The contract you implement

Every agent subclasses `euchre_bot.agents.base.Agent` and implements two methods:

```python
def act(self, observation, action_mask, info=None, *, deterministic=False) -> int
def learn(self, batch: list[Trajectory]) -> dict[str, float]
```

- `observation` is a `float32` vector of length `EuchreEnv.observation_dim`.
- `action_mask` is `float32[EuchreEnv.num_actions]`, `1.0` for legal actions.
- A `Trajectory` is `list[Transition]`; a `Transition` is
  `(observation, action_mask, action, reward, done)`. Reward is non-zero only on
  the final transition of each seat's trajectory (rewards are sparse).
- `learn` returns a dict of scalar metrics — hand them to `MetricsLogger` and
  watch the curves (see `docs/monitoring.md`).

Your network is just: **input `observation_dim` → output `num_actions` logits**
(plus a value head for actor-critic methods).

## The one trick you must get right: action masking

At most a handful of the 35 actions are legal in any state. If you let the
policy put probability on illegal actions, it will sample them, the env will
raise, and even if it didn't, you'd be training on nonsense. Mask the logits
*before* building the distribution:

```python
import torch
from torch.distributions import Categorical

logits = self.policy(obs)                      # shape [num_actions]
mask = torch.as_tensor(action_mask, dtype=torch.bool)
logits = logits.masked_fill(~mask, float("-inf"))   # illegal -> -inf
dist = Categorical(logits=logits)                    # softmax ignores -inf
action = dist.probs.argmax() if deterministic else dist.sample()
```

Two rules that save hours of debugging:
1. **Mask consistently.** In `learn`, when you recompute log-probs for stored
   actions, mask with the *stored* `action_mask` from each transition — not a
   freshly computed one. PPO's importance ratio is meaningless otherwise.
2. **For DQN, mask twice.** Mask in `act` (argmax over legal actions only) *and*
   in the TD target (`max` over actions legal in the next state). The base
   `Transition` doesn't store the next state's mask — wiring that through is a
   deliberate design exercise called out in `algos/dqn.py`.

## Turning sparse rewards into returns

Reward arrives once, on the last transition. Convert a trajectory into a return
per step by discounting backwards — write this once, reuse everywhere:

```python
def discounted_returns(rewards, gamma):
    out, running = [], 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        out.append(running)
    out.reverse()
    return out
```

## Suggested order

### 1. REINFORCE (`reinforce.py`) — start here
The minimal policy gradient. Build an MLP policy, implement masked `act`, and in
`learn`: compute returns, normalise them across the batch, recompute masked
log-probs, and step on `-(log_prob * return).mean()`. Success criterion: it
should beat `RandomAgent` (eval win-rate well above 0.5) within a few hundred
iterations. If it doesn't, suspect masking or a returns sign error first.

### 2. A2C (`a2c.py`)
Add a value head and subtract it as a baseline (advantage = return − value), add
a value loss and a small entropy bonus. Lower variance, faster, steadier. The
entropy bonus matters here because the mostly-masked 35-way head can collapse
early.

### 3. PPO (`ppo.py`) — your likely champion
A2C plus a clipped surrogate objective and multiple epochs per batch (reusing
data safely), optionally with GAE advantages. This is the algorithm most likely
to comfortably beat the heuristic. Start from robust defaults (clip 0.2, γ 0.99,
λ 0.95, lr 3e-4, entropy 0.01, a few epochs) and change one thing at a time.

### 4. DQN (`dqn.py`) — the off-policy detour
Value-based, with a replay buffer and target network. Teaches the other half of
RL. Be aware that off-policy + self-play is genuinely harder than the CartPole
tutorials; don't be discouraged if PPO ends up your best player.

## How you'll know it's working

Wire your agent into `examples/train.py` (just pass `--agent ppo`) and watch
`eval_vs_random/win_rate` and `eval_vs_heuristic/win_rate` in TensorBoard. The
canonical healthy story: win-rate vs random climbs first and fast, then
win-rate vs heuristic crosses 0.5 and keeps climbing. Self-play reward hovering
near zero is expected and tells you nothing — always read the *eval* curves.
See `docs/rl_best_practices.md` for the failure-mode catalogue.

## A note on self-play dynamics

Your opponent is a copy of you, improving as you do — the target is
non-stationary. On-policy methods (REINFORCE/A2C/PPO) tolerate this gracefully.
For stability you can later keep a pool of frozen past checkpoints and play
against them sometimes, which prevents the policy from chasing its own tail. Get
a single shared policy working against the baselines first; reach for the
checkpoint pool only if you see win-rate oscillation.
