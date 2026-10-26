"""Learning agents.

``reinforce`` is the simplest, fully-commented reference; ``a2c`` adds a learned
value baseline, ``ppo`` adds clipped multi-epoch updates and GAE, and ``dqn`` is
the value-based, off-policy alternative. Easiest to hardest:

    reinforce -> a2c -> ppo -> dqn

Each module subclasses :class:`euchre_bot.agents.base.Agent`, so every agent
drops straight into :func:`euchre_bot.training.train` and
:func:`euchre_bot.training.evaluate`. The companion guide is
``docs/implementing_algorithms.md``.

Importing this package requires ``torch`` (install the ``learn`` extra). The
engine/env/agents layers do not, so you can run the rules, the environment, and
the baselines without a deep-learning stack.
"""

from .a2c import A2CAgent
from .dqn import DQNAgent
from .ppo import PPOAgent
from .reinforce import ReinforceAgent

__all__ = ["ReinforceAgent", "A2CAgent", "PPOAgent", "DQNAgent"]
