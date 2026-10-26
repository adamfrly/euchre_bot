"""PPO (Proximal Policy Optimization).

PPO is the workhorse for this kind of problem and the likely end-state agent.
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

   The *old* log-probs are recomputed once, before the epoch loop, and detached.
   That is exact here because ``learn`` runs right after collection, before any
   update, so the current policy *is* the one that generated the data.

2. **Generalized Advantage Estimation (GAE).** A bias/variance knob on the
   advantage:

       delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
       A_t     = delta_t + gamma * lambda * A_{t+1}

   Computed per seat-trajectory; V(s_{t+1}) is 0 past the final transition.

Euchre-specific notes
---------------------
* Same masking requirement. Crucially, **mask consistently**: the old and new
  log-probs must come from distributions masked the *same* way, or the ratio is
  garbage. Each Transition carries its action_mask and we reapply it every epoch.
* Trajectories here are short and episodes cheap, so collect a few hundred hands
  per update and run ~4-10 epochs over them.
* Standard, robust hyperparameters: clip eps 0.2, gamma 0.99, lambda 0.95,
  lr 3e-4, entropy 0.01, value_coef 0.5, a couple of minibatches.

References
---------
* Schulman et al. (2017), "Proximal Policy Optimization Algorithms".
* Schulman et al. (2015), "High-Dimensional Continuous Control Using GAE".
* The "37 implementation details of PPO" blog is worth a careful read.
* ``docs/implementing_algorithms.md``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from ..agents.base import Agent, Trajectory
from ..env.euchre_env import EuchreEnv
from .a2c import ActorCritic


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
        max_grad_norm: float = 0.5,
        target_kl: float | None = 0.03,
        device: str = "cpu",
        seed: int | None = None,
    ) -> None:
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.device = torch.device(device)
        self.n_actions = n_actions
        if seed is not None:
            torch.manual_seed(seed)

        self.net = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)

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

    def _gae(self, rewards: np.ndarray, values: np.ndarray) -> np.ndarray:
        """GAE over one trajectory whose final transition is terminal."""
        advantages = np.zeros_like(rewards)
        running = 0.0
        next_value = 0.0  # nothing follows the terminal transition
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value - values[t]
            running = delta + self.gamma * self.gae_lambda * running
            advantages[t] = running
            next_value = values[t]
        return advantages

    def learn(self, batch: list[Trajectory]) -> dict[str, float]:
        batch = [trajectory for trajectory in batch if trajectory]
        if not batch:
            return {}

        observations = np.array([t.observation for traj in batch for t in traj])
        masks = np.array([t.action_mask for traj in batch for t in traj])
        actions = [t.action for traj in batch for t in traj]

        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        mask_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        action_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)

        # Snapshot the behaviour policy's log-probs and values before any update.
        with torch.no_grad():
            logits, values = self.net(obs_t)
            old_log_probs = Categorical(
                logits=logits.masked_fill(~mask_t, float("-inf"))
            ).log_prob(action_t)
        values_np = values.cpu().numpy()

        # GAE per trajectory, using that trajectory's slice of the value array.
        advantages_np = np.empty_like(values_np)
        start = 0
        for trajectory in batch:
            end = start + len(trajectory)
            rewards = np.array([t.reward for t in trajectory], dtype=np.float32)
            advantages_np[start:end] = self._gae(rewards, values_np[start:end])
            start = end

        adv_t = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        return_t = adv_t + values  # the critic's regression target
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        n = len(actions)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0}
        updates = 0
        epochs_run = 0
        for _ in range(self.epochs):
            epochs_run += 1
            epoch_kls: list[float] = []
            for idx in torch.randperm(n, device=self.device).split(self.minibatch_size):
                logits, new_values = self.net(obs_t[idx])
                dist = Categorical(logits=logits.masked_fill(~mask_t[idx], float("-inf")))
                new_log_probs = dist.log_prob(action_t[idx])
                entropy = dist.entropy().mean()

                log_ratio = new_log_probs - old_log_probs[idx]
                ratio = log_ratio.exp()
                adv = adv_t[idx]
                surrogate = torch.min(
                    ratio * adv,
                    torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv,
                )
                policy_loss = -surrogate.mean()
                value_loss = 0.5 * (new_values - return_t[idx]).pow(2).mean()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # The low-variance KL estimator from the "37 details" post.
                    approx_kl = float(((ratio - 1) - log_ratio).mean().item())
                    clip_frac = float(((ratio - 1).abs() > self.clip_eps).float().mean().item())
                epoch_kls.append(approx_kl)
                stats["policy_loss"] += float(policy_loss.item())
                stats["value_loss"] += float(value_loss.item())
                stats["entropy"] += float(entropy.item())
                stats["approx_kl"] += approx_kl
                stats["clip_frac"] += clip_frac
                updates += 1

            # Stop early if the policy has drifted too far from the data's policy.
            if self.target_kl is not None and np.mean(epoch_kls) > 1.5 * self.target_kl:
                break

        metrics = {k: v / updates for k, v in stats.items()}
        metrics["epochs"] = float(epochs_run)
        metrics["transitions"] = float(n)
        return metrics

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
