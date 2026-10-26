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
* **Going-alone exploration.** Going alone is only right on very strong hands,
  so early on it looks like a losing action, its probability collapses, and the
  agent never discovers the 4-point lone march. ``alone_explore`` mixes a small,
  annealed probability epsilon of "go alone" (uniform over the legal alone
  calls) into the policy itself:

      pi'(a|s) = (1 - eps) * pi(a|s) + eps * uniform_alone(a|s)

  Because the *same* mixture is used to act and to compute log-probs in
  ``learn``, PPO stays exactly on-policy. Epsilon only changes after an update,
  never between collection and learning. Evaluation (``deterministic=True``)
  ignores the mixture.

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
from ..engine import actions as A
from ..env.euchre_env import EuchreEnv
from .a2c import ActorCritic

ALONE_ACTIONS = [A.ORDER_UP_ALONE] + [A.CALL_SUIT_ALONE_BASE + suit for suit in range(4)]


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
        alone_explore: float = 0.0,
        alone_explore_updates: int = 0,
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
        # Going-alone exploration: start at `alone_explore`, decay linearly to 0
        # over `alone_explore_updates` learn() calls (0 means never decay).
        self.alone_explore = alone_explore
        self.alone_explore_updates = alone_explore_updates
        self.updates_done = 0
        self.device = torch.device(device)
        self.n_actions = n_actions
        if seed is not None:
            torch.manual_seed(seed)

        self.net = ActorCritic(obs_dim, n_actions, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self._alone_mask = torch.zeros(n_actions, dtype=torch.bool, device=self.device)
        self._alone_mask[ALONE_ACTIONS] = True

    @property
    def alone_epsilon(self) -> float:
        if self.alone_explore_updates <= 0:
            return self.alone_explore
        frac = min(1.0, self.updates_done / self.alone_explore_updates)
        return self.alone_explore * (1.0 - frac)

    def _distribution(self, logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
        """The masked policy, mixed with going-alone exploration when it is on."""
        logits = logits.masked_fill(~mask, float("-inf"))
        eps = self.alone_epsilon
        if eps <= 0.0:
            return Categorical(logits=logits)
        alone_legal = (mask & self._alone_mask).float()
        n_alone = alone_legal.sum(-1, keepdim=True)
        # Only mix where some alone call is legal (i.e. during bidding).
        mix = eps * (n_alone > 0).float()
        probs = (1.0 - mix) * torch.softmax(logits, -1) + mix * alone_legal / n_alone.clamp(min=1)
        return Categorical(probs=probs)

    # ------------------------------------------------------------------ #
    # Acting
    # ------------------------------------------------------------------ #

    def act(self, observation, action_mask, info=None, *, deterministic=False) -> int:
        with torch.no_grad():
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            mask = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device)
            logits, _ = self.net(obs)
            if deterministic:
                return int(torch.argmax(logits.masked_fill(~mask, float("-inf"))).item())
            return int(self._distribution(logits, mask).sample().item())

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
            old_log_probs = self._distribution(logits, mask_t).log_prob(action_t)
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
                dist = self._distribution(logits, mask_t[idx])
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
        metrics["alone_epsilon"] = self.alone_epsilon
        alone_taken = torch.isin(action_t, torch.as_tensor(ALONE_ACTIONS, device=self.device))
        metrics["alone_rate"] = float(alone_taken.float().mean().item())
        self.updates_done += 1
        return metrics

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: str) -> None:
        self.net.load_state_dict(torch.load(path, map_location=self.device))
