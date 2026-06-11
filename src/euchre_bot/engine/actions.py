"""The discrete action space shared by every phase of a Euchre hand.

A Euchre hand has four kinds of decision -- bid on the up-card, name a suit,
discard, and play a card -- but reinforcement-learning agents are easiest to
write against a *single* fixed action space. We therefore flatten every
decision into one ``Discrete(35)`` space and rely on an **action mask** (see
:mod:`euchre_bot.env.encoding`) to expose only the legal actions at each step.

Action layout (indices are stable and meaningful across the whole hand)::

     0        PASS                         (bidding rounds 1 & 2)
     1        ORDER_UP                      (round 1: accept up-card as trump)
     2        ORDER_UP_ALONE               (round 1: accept and go alone)
     3.. 6    CALL_SUIT[suit]              (round 2: name a suit, with partner)
     7..10    CALL_SUIT_ALONE[suit]        (round 2: name a suit, go alone)
    11..34    CARD[index]                   (discard or play card with that index)

The ``CARD`` block is indexed by :pyattr:`euchre_bot.engine.cards.Card.index`,
so the same action id means "the nine of clubs" whether you are discarding it
after picking up the up-card or playing it to a trick. The phase plus the mask
make the meaning unambiguous.
"""

from __future__ import annotations

from .cards import NUM_CARDS, Card, Suit

PASS = 0
ORDER_UP = 1
ORDER_UP_ALONE = 2

CALL_SUIT_BASE = 3          # CALL_SUIT_BASE + suit  -> name `suit`, with partner
CALL_SUIT_ALONE_BASE = 7    # CALL_SUIT_ALONE_BASE + suit -> name `suit`, alone
CARD_BASE = 11              # CARD_BASE + card.index -> discard/play that card

NUM_ACTIONS = CARD_BASE + NUM_CARDS  # == 35


def call_suit_action(suit: Suit, alone: bool) -> int:
    """Action id for naming ``suit`` in the second bidding round."""
    return (CALL_SUIT_ALONE_BASE if alone else CALL_SUIT_BASE) + int(suit)


def card_action(card: Card) -> int:
    """Action id for discarding or playing ``card``."""
    return CARD_BASE + card.index


def is_card_action(action: int) -> bool:
    return CARD_BASE <= action < NUM_ACTIONS


def card_of_action(action: int) -> Card:
    """Inverse of :func:`card_action`. Raises if ``action`` is not a card action."""
    if not is_card_action(action):
        raise ValueError(f"action {action} is not a card action")
    return Card.from_index(action - CARD_BASE)


def decode_call(action: int) -> tuple[Suit, bool]:
    """Return ``(named_suit, went_alone)`` for a round-2 CALL_SUIT* action."""
    if CALL_SUIT_BASE <= action < CALL_SUIT_BASE + 4:
        return Suit(action - CALL_SUIT_BASE), False
    if CALL_SUIT_ALONE_BASE <= action < CALL_SUIT_ALONE_BASE + 4:
        return Suit(action - CALL_SUIT_ALONE_BASE), True
    raise ValueError(f"action {action} is not a call-suit action")


def action_name(action: int) -> str:
    """Human-readable label for an action id (for logging and debugging)."""
    if action == PASS:
        return "PASS"
    if action == ORDER_UP:
        return "ORDER_UP"
    if action == ORDER_UP_ALONE:
        return "ORDER_UP_ALONE"
    if CALL_SUIT_BASE <= action < CALL_SUIT_BASE + 4:
        return f"CALL_{Suit(action - CALL_SUIT_BASE).name}"
    if CALL_SUIT_ALONE_BASE <= action < CALL_SUIT_ALONE_BASE + 4:
        return f"CALL_{Suit(action - CALL_SUIT_ALONE_BASE).name}_ALONE"
    if is_card_action(action):
        return f"CARD_{card_of_action(action)}"
    return f"UNKNOWN({action})"
