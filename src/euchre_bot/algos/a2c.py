"""A2C (Advantage Actor-Critic) -- YOUR IMPLEMENTATION GOES HERE.

A2C is REINFORCE plus a learned **value baseline**. Instead of weighting the
log-probability by the raw return, you weight it by the *advantage*
A_t = G_t - V(s_t), where V is a critic trained to predict the return. The
baseline does not bias the gradient but dramatically lowers its variance, so
learning is faster and steadier.

What changes versus REINFORCE
-----------------------------
* The network gains a second head (or a second network): a scalar value V(s).
  A shared-trunk actor-critic is common -- trunk -> {policy logits, value}.
* The loss has three terms:
      policy_loss  = -(log_prob(a_t) * advantage_t.detach()).mean()
      value_loss   =  F.mse_loss(V(s_t), returns_t)
      entropy_bonus= -beta * entropy(pi(.|s_t)).mean()   # encourages exploration
      loss = policy_loss + value_coef * value_loss + entropy_bonus
  Detaching the advantage when it multiplies log_prob is important: the critic
  is trained by `value_loss`, not through the policy term.
* Returns/advantages: start with Monte-Carlo returns (as in REINFORCE) minus
  V(s_t). Later you can upgrade to n-step returns or GAE (see PPO).

Euchre-specific notes
---------------------
* Same masking requirement as REINFORCE -- mask logits before the Categorical.
* The entropy bonus matters here: with a 35-way action space that is mostly
  masked, a confident-but-wrong policy can collapse early. A small entropy
  coefficient (e.g. 0.01) keeps bidding/play exploration alive.

References
---------
* Mnih et al. (2016), "Asynchronous Methods for Deep RL" (the A3C/A2C paper).
* ``docs/implementing_algorithms.md``.
"""

from __future__ import annotations

import torch  # noqa: F401

from ..agents.base import Agent, Trajectory
from ..env.euchre_env import EuchreEnv


class A2CAgent(Agent):
    def __init__(
        self,
        obs_dim: int = EuchreEnv.observation_dim,
        n_actions: int = EuchreEnv.num_actions,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device(device)
        self.n_actions = n_actions
        # TODO(you): build a shared-trunk actor-critic:
        #   trunk: obs_dim -> hidden -> hidden
        #   policy head: hidden -> n_actions (logits)
        #   value head:  hidden -> 1
        # plus an Adam optimizer over all parameters.
        raise NotImplementedError("Build the actor-critic network and optimizer here.")

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        # TODO(you): identical masking + sampling logic as REINFORCE.act.
        raise NotImplementedError

    def learn(self, batch: list[Trajectory]) -> dict[str, float]:
        # TODO(you):
        #  1. Compute discounted returns G_t per trajectory (reuse your code).
        #  2. Forward the net on all observations -> logits, values.
        #  3. advantage = returns - values.detach().
        #  4. policy_loss, value_loss, entropy as in the docstring; combine.
        #  5. Backprop the combined loss; step. Return metrics.
        raise NotImplementedError
