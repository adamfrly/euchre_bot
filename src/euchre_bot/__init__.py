"""euchre_bot -- a reinforcement-learning playground built around Euchre.

Layout
------
* :mod:`euchre_bot.engine`   -- pure game rules (cards, bowers, tricks, scoring).
* :mod:`euchre_bot.env`      -- the turn-based multi-agent environment + encoding.
* :mod:`euchre_bot.agents`   -- the ``Agent`` interface and baseline players.
* :mod:`euchre_bot.algos`    -- learning agents (REINFORCE/A2C/PPO/DQN) -- *you implement these*.
* :mod:`euchre_bot.training` -- self-play loop and evaluation harness.

The engine/env/agents layers have no deep-learning dependency; only the
algorithms (and your training scripts) import torch. That separation is on
purpose -- see ``docs/rl_best_practices.md``.
"""

from .env import EuchreEnv, StepResult

__version__ = "0.1.0"
__all__ = ["EuchreEnv", "StepResult"]
