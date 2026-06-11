"""PPO (Proximal Policy Optimization) -- YOUR IMPLEMENTATION GOES HERE.

PPO is the workhorse for this kind of problem and your likely end-state agent.
It is A2C with two upgrades that make it dramatically more sample-efficient and
stable:

1. **Multiple epochs per batch with a clipped objective.** REINFORCE/A2C use
   each trajectory for exactly one gradient step (strictly on-policy). PPO reuses
   a batch for several epochs while preventing the policy from moving too far via
   the clipped surrogate:

       ratio_t = exp(log_pi_new(a_t) - log_pi_old(a_t))
       L_clip  = mean( min( ratio_t * A_t,
                            clip(ratio_t, 1-eps, 1+eps) * A_t ) )
       loss = -L_clip + value_coef * value_loss - entropy_coef * entropy

   You must store the *old* log-probs (under the policy that generated the data)
   to form the ratio. The simplest place is to capture them at collection time;
   alternatively recompute once before the epoch loop and detach.

2. **Generalized Advantage Estimation (GAE).** A bias/variance knob on the
   advantage:

       delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
       A_t     = delta_t + gamma * lambda * A_{t+1}

   With sparse terminal rewards and short (<=, ~5-11 step) Euchre trajectories,
   plain Monte-Carlo advantages also work; add GAE once the basics train.

Euchre-specific notes
---------------------
* Same masking requirement. Crucially, **mask consistently**: the old and new
  log-probs must come from distributions masked the *same* way, or the ratio is
  garbage. Store the action_mask with each transition (the Transition already
  carries it) and reapply it every epoch.
* Trajectories here are short and episodes cheap, so collect a few hundred hands
  per update and run ~4-10 epochs over them.
* Standard, robust hyperparameters to start: clip eps 0.2, gamma 0.99,
  lambda 0.95, lr 3e-4, entropy 0.01, value_coef 0.5, a couple of minibatches.

References
---------
* Schulman et al. (2017), "Proximal Policy Optimization Algorithms".
* Schulman et al. (2015), "High-Dimensional Continuous Control Using GAE".
* The "37 implementation details of PPO" blog is worth a careful read.
* ``docs/implementing_algorithms.md``.
"""

from __future__ import annotations

import torch  # noqa: F401

from ..agents.base import Agent, Trajectory
from ..env.euchre_env import EuchreEnv


class PPOAgent(Agent):
    def __init__(
        self,
        obs_dim: int = EuchreEnv.observation_dim,
        n_actions: int = EuchreEnv.num_actions,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        epochs: int = 4,
        minibatch_size: int = 256,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        self.n_actions = n_actions
        # TODO(you): build the actor-critic (reuse your A2C network) + optimizer.
        raise NotImplementedError("Build the actor-critic network and optimizer here.")

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        # TODO(you): same masked sampling as A2C/REINFORCE.
        raise NotImplementedError

    def learn(self, batch: list[Trajectory]) -> dict[str, float]:
        # TODO(you):
        #  1. Flatten batch -> arrays of (obs, mask, action, reward, done).
        #  2. Compute values V(s), then returns + advantages (MC first, GAE later).
        #     Normalise advantages.
        #  3. Snapshot old log-probs (masked) under the current policy; detach.
        #  4. For `epochs`, iterate minibatches: recompute new masked log-probs &
        #     values, form the clipped surrogate, value loss, entropy; step.
        #  5. Return metrics (policy_loss, value_loss, entropy, approx_kl, ...).
        raise NotImplementedError
