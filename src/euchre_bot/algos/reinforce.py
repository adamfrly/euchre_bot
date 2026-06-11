"""REINFORCE (vanilla policy gradient) -- WORKED REFERENCE EXAMPLE.

This is the one algorithm in the project that is **fully implemented**, on
purpose: it's your template. Read it top to bottom, run it (it really does beat
``RandomAgent``), then use it as the skeleton for ``a2c.py`` -> ``ppo.py`` ->
``dqn.py``, which are left as stubs for you. Every non-obvious line is commented.

The idea in one paragraph
-------------------------
REINFORCE adjusts the policy so that actions which led to high return become
more likely. For each trajectory it computes the discounted return G_t from
every step, then takes a gradient step on

    loss = - sum_t  log pi(a_t | s_t) * A_t

where A_t is the return G_t with a baseline subtracted to reduce variance --
here the standard trick of normalising returns to zero mean / unit std across
the batch. Maximising reward == minimising that negative log-likelihood weighted
by return.

How the pieces map to this codebase
-----------------------------------
* The policy network is an MLP: ``observation_dim`` -> hidden -> hidden ->
  ``num_actions`` raw logits.
* ``act`` masks illegal actions to ``-inf`` before sampling -- the single most
  important detail (see the comment there).
* ``learn`` receives a *batch of trajectories* (one per seat per hand, see
  ``training/self_play.py``). Reward is sparse: non-zero only on each
  trajectory's last transition. We discount it back into a per-step return,
  normalise, recompute masked log-probs, and step.
* We do **not** stash tensors during ``act``. Because REINFORCE is on-policy and
  we update immediately after collection, recomputing log-probs from the stored
  ``(observation, action, mask)`` is exact and keeps ``act`` simple. The other
  algorithms reuse this pattern.

References
----------
* Sutton & Barto, *Reinforcement Learning: An Introduction*, ch. 13.
* Williams (1992), "Simple statistical gradient-following algorithms".
* ``docs/implementing_algorithms.md`` walks through this file step by step.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

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
        seed: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.n_actions = n_actions
        self.device = torch.device(device)
        if seed is not None:
            torch.manual_seed(seed)

        # A small MLP that maps an observation to one raw logit per action. No
        # softmax here -- we apply the mask first and let Categorical(logits=...)
        # do the softmax, so illegal actions get exactly zero probability.
        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #

    def _masked_logits(self, observation: np.ndarray, action_mask: np.ndarray) -> torch.Tensor:
        """Run the network and force illegal actions to logit ``-inf``.

        Masking is *the* detail that makes RL on a fixed action space work. The
        network can put weight on any of the 35 actions, but only a few are legal
        in a given state. Setting illegal logits to ``-inf`` makes their softmax
        probability exactly 0, so we never sample them (the env raises on an
        illegal action -- that's a deliberate tripwire) and the gradient flows
        only through legal choices.
        """
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
        logits = self.policy(obs)
        return logits.masked_fill(~mask, float("-inf"))

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        # No gradient needed while collecting experience.
        with torch.no_grad():
            logits = self._masked_logits(observation, action_mask)
            if deterministic:
                # Greedy: best legal action. Used during evaluation.
                return int(torch.argmax(logits).item())
            # Sample from the masked policy. Used during training for exploration.
            return int(Categorical(logits=logits).sample().item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def _discounted_returns(self, rewards: list[float]) -> list[float]:
        """G_t = r_t + gamma * G_{t+1}, computed by one backward pass.

        Rewards here are sparse (only the last element is non-zero), so this just
        spreads the terminal reward back over the hand with a gamma^k decay -- but
        it's written for the general case because every other algorithm reuses it.
        """
        returns: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.gamma * running
            returns.append(running)
        returns.reverse()
        return returns

    def learn(self, batch: list[Trajectory]) -> dict[str, float]:
        # Flatten the batch of trajectories into parallel arrays. We compute the
        # discounted return per step *within* each trajectory, then pool
        # everything for one gradient step.
        observations: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        returns: list[float] = []

        for trajectory in batch:
            if not trajectory:
                continue
            rewards = [t.reward for t in trajectory]
            traj_returns = self._discounted_returns(rewards)
            for transition, g in zip(trajectory, traj_returns, strict=True):
                observations.append(transition.observation)
                masks.append(transition.action_mask)
                actions.append(transition.action)
                returns.append(g)

        if not actions:
            return {}

        obs_t = torch.as_tensor(np.array(observations), dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(np.array(masks), dtype=torch.bool, device=self.device)
        action_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        return_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Baseline by standardising returns across the batch. This doesn't bias
        # the gradient (the baseline is action-independent) but slashes its
        # variance, which is the difference between learning and thrashing.
        return_t = (return_t - return_t.mean()) / (return_t.std() + 1e-8)

        # Recompute logits *with* gradients this time, mask, and score the
        # actions that were actually taken.
        logits = self.policy(obs_t).masked_fill(~mask_t, float("-inf"))
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(action_t)

        # The policy-gradient loss. Negative because optimisers minimise.
        loss = -(log_probs * return_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Metrics worth watching in MLflow (see docs/monitoring.md): a finite,
        # non-exploding loss; entropy that decays smoothly (a crash to ~0 means
        # premature collapse).
        return {
            "loss": float(loss.item()),
            "entropy": float(dist.entropy().mean().item()),
            "transitions": float(len(actions)),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        self.policy.load_state_dict(torch.load(path, map_location=self.device))
