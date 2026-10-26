"""A2C (Advantage Actor-Critic).

A2C is REINFORCE plus a learned **value baseline**. Instead of weighting the
log-probability by the raw return, you weight it by the *advantage*
A_t = G_t - V(s_t), where V is a critic trained to predict the return. The
baseline does not bias the gradient but dramatically lowers its variance, so
learning is faster and steadier.

What changes versus REINFORCE
-----------------------------
* The network gains a second head: a scalar value V(s). We use a shared-trunk
  actor-critic -- trunk -> {policy logits, value} -- defined here as
  :class:`ActorCritic` and reused by PPO.
* The loss has three terms:
      policy_loss  = -(log_prob(a_t) * advantage_t.detach()).mean()
      value_loss   =  F.mse_loss(V(s_t), returns_t)
      entropy_bonus= -beta * entropy(pi(.|s_t)).mean()   # encourages exploration
      loss = policy_loss + value_coef * value_loss + entropy_bonus
  Detaching the advantage when it multiplies log_prob is important: the critic
  is trained by `value_loss`, not through the policy term.
* Returns/advantages: Monte-Carlo returns (as in REINFORCE) minus V(s_t). PPO
  upgrades this to GAE.

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..agents.base import Agent, Trajectory
from ..env.euchre_env import EuchreEnv


class ActorCritic(nn.Module):
    """Shared trunk feeding a policy head (logits) and a value head (scalar)."""

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(obs)
        return self.policy_head(features), self.value_head(features).squeeze(-1)


def discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    """G_t = r_t + gamma * G_{t+1}, computed by one backward pass."""
    returns: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


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
        max_grad_norm: float = 0.5,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = torch.device(device)
        self.n_actions = n_actions
        if seed is not None:
            torch.manual_seed(seed)

        self.net = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            logits, _ = self.net(obs)
            logits = logits.masked_fill(~mask, float("-inf"))
            if deterministic:
                return int(torch.argmax(logits).item())
            return int(Categorical(logits=logits).sample().item())

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #

    def learn(self, batch: list[Trajectory]) -> dict[str, float]:
        observations: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        actions: list[int] = []
        returns: list[float] = []

        for trajectory in batch:
            if not trajectory:
                continue
            traj_returns = discounted_returns([t.reward for t in trajectory], self.gamma)
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

        logits, values = self.net(obs_t)
        dist = Categorical(logits=logits.masked_fill(~mask_t, float("-inf")))
        log_probs = dist.log_prob(action_t)
        entropy = dist.entropy().mean()

        # The critic is the baseline; detach so the policy term can't move it.
        advantages = (return_t - values).detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, return_t)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "transitions": float(len(actions)),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
