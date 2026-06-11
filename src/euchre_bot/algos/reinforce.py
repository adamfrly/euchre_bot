"""REINFORCE (vanilla policy gradient) -- YOUR IMPLEMENTATION GOES HERE.

This is the gentlest entry point into the algorithms, so it is the most heavily
annotated. Once you have it working and beating ``RandomAgent``, the others are
variations on the same skeleton.

The idea in one paragraph
-------------------------
REINFORCE adjusts the policy so that actions which led to high return become
more likely. For each trajectory it computes the discounted return G_t from
every step, then takes a gradient step on

    loss = - sum_t  log pi(a_t | s_t) * A_t

where A_t is the return G_t (optionally with a baseline subtracted to reduce
variance -- here we use the standard trick of normalising returns to zero mean
/ unit std across the batch). Maximising reward == minimising that negative log
likelihood weighted by return.

The recipe (implement these in `learn`)
----------------------------------------
1. For each trajectory, walk it backwards computing
   G_t = r_t + gamma * G_{t+1}. In this env reward is non-zero only on the last
   transition, so G_t = gamma^(T-1-t) * terminal_reward -- but write the general
   loop; you will reuse it everywhere.
2. Concatenate (G_t) across the batch and normalise:
   returns = (returns - returns.mean()) / (returns.std() + 1e-8).
3. Re-run the policy network on the stored observations to get fresh logits.
4. **Apply the action mask** before forming the distribution (see the note in
   `act`). Build a Categorical, take log_prob(action).
5. loss = -(log_probs * returns).mean(); backprop; optimizer step.

Euchre-specific gotchas
-----------------------
* The network input width is ``EuchreEnv.observation_dim`` and the output width
  is ``EuchreEnv.num_actions`` (35). Most outputs are illegal at any given
  state -- masking is not optional, it is the difference between learning and
  diverging.
* Reward is sparse (one signal per hand) and the trajectory mixes bidding and
  card-play decisions. That is fine for REINFORCE; just discount back from the
  terminal reward.
* Self-play is non-stationary (your opponent is also you, improving). REINFORCE
  has no replay buffer, which side-steps the worst of the off-policy staleness
  issues -- a nice property for a first algorithm.

References
----------
* Sutton & Barto, *Reinforcement Learning: An Introduction*, ch. 13.
* Williams (1992), "Simple statistical gradient-following algorithms".
* ``docs/implementing_algorithms.md`` walks through this file step by step.
"""

from __future__ import annotations

import torch  # noqa: F401  -- you will need this throughout

from ..agents.base import Agent, Trajectory
from ..env.euchre_env import EuchreEnv


class ReinforceAgent(Agent):
    def __init__(
        self,
        obs_dim: int = EuchreEnv.observation_dim,
        n_actions: int = EuchreEnv.num_actions,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        device: str = "cpu",
    ) -> None:
        self.gamma = gamma
        self.device = torch.device(device)
        self.n_actions = n_actions
        # TODO(you): build a small MLP policy network mapping
        #   obs_dim -> hidden -> hidden -> n_actions (raw logits, no softmax),
        # move it to self.device, and create an optimizer (Adam is a fine
        # default at `lr`). Store both on self.
        raise NotImplementedError("Build the policy network and optimizer here.")

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        # TODO(you):
        #  1. Tensorise `observation` and run the policy net to get logits.
        #  2. Mask illegal actions: set logits where action_mask == 0 to -inf,
        #     e.g.  logits = logits.masked_fill(mask == 0, float("-inf")).
        #  3. If `deterministic`, return argmax of the masked logits.
        #     Otherwise sample from Categorical(logits=masked_logits).
        #  4. Return the chosen action as a plain int.
        raise NotImplementedError

    def learn(self, batch: list[Trajectory]) -> dict[str, float]:
        # TODO(you): implement the 5-step recipe in this module's docstring.
        # Return a dict of scalar metrics (e.g. {"loss": ..., "mean_return": ...})
        # so the training loop can log your progress.
        raise NotImplementedError

    # Suggested, once it trains: implement save()/load() with torch.save on the
    # network's state_dict so you can checkpoint and evaluate later.
