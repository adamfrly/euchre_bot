"""Learning agents.

``reinforce`` is a **fully-worked reference example** -- read it and run it.
``a2c``, ``ppo``, and ``dqn`` are **stubs for you to implement**, building on the
REINFORCE skeleton. Suggested order, easiest to hardest:

    reinforce (done) -> a2c -> ppo -> dqn

Each module subclasses :class:`euchre_bot.agents.base.Agent`, so a finished
agent drops straight into :func:`euchre_bot.training.train` and
:func:`euchre_bot.training.evaluate`. The stubs contain the algorithm's
intuition, the precise update recipe, the Euchre-specific masking gotchas, and
``# TODO(you)`` markers. The companion guide is
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
