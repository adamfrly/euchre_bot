"""Learning agents -- **these are stubs for you to implement**.

Each module subclasses :class:`euchre_bot.agents.base.Agent`, so a finished
agent drops straight into :func:`euchre_bot.training.train` and
:func:`euchre_bot.training.evaluate`. Suggested order, easiest to hardest:

    reinforce -> a2c -> ppo -> dqn

Every file contains the algorithm's intuition, the precise update recipe, the
Euchre-specific masking gotchas, and ``# TODO(you)`` markers. The companion
guide is ``docs/implementing_algorithms.md``.

Importing this package requires ``torch`` (install the ``learn`` extra). The
engine/env/agents layers do not, so you can run the rules, the environment, and
the baselines without a deep-learning stack.
"""

from .a2c import A2CAgent
from .dqn import DQNAgent
from .ppo import PPOAgent
from .reinforce import ReinforceAgent

__all__ = ["ReinforceAgent", "A2CAgent", "PPOAgent", "DQNAgent"]
