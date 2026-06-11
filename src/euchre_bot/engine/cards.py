"""Card model and trump-aware card logic for Euchre.

Euchre uses a 24-card deck: 9, 10, Jack, Queen, King, Ace in each of the four
suits. The defining quirk of the game is the *bowers*:

* The **right bower** is the Jack of the trump suit. It is the highest card in
  the game.
* The **left bower** is the Jack of the *other suit of the same color* as
  trump (e.g. if Hearts are trump, the left bower is the Jack of Diamonds). It
  counts **as a trump card** -- its *effective* suit becomes the trump suit --
  and ranks just below the right bower.

Almost every subtle bug in a Euchre engine comes from the left bower, because
its *printed* suit and its *effective* (in-play) suit differ once trump is set.
We therefore funnel every suit comparison through :func:`effective_suit` and
never compare ``card.suit`` directly during play.

This module is intentionally free of any game-flow or RL concepts. It is a
small, pure, easily-testable vocabulary that the rest of the package builds on.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class Suit(IntEnum):
    """The four suits, with helpers for the color relationships Euchre cares about."""

    CLUBS = 0
    DIAMONDS = 1
    HEARTS = 2
    SPADES = 3

    @property
    def color(self) -> str:
        """``"black"`` for clubs/spades, ``"red"`` for diamonds/hearts."""
        return "black" if self in (Suit.CLUBS, Suit.SPADES) else "red"

    @property
    def same_color_suit(self) -> Suit:
        """The *other* suit of the same color -- i.e. the left bower's home suit."""
        return {
            Suit.CLUBS: Suit.SPADES,
            Suit.SPADES: Suit.CLUBS,
            Suit.HEARTS: Suit.DIAMONDS,
            Suit.DIAMONDS: Suit.HEARTS,
        }[self]

    @property
    def short(self) -> str:
        return "CDHS"[int(self)]


class Rank(IntEnum):
    """Card ranks. Values are chosen so larger int == stronger *off-trump* card."""

    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14

    @property
    def short(self) -> str:
        return {9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}[int(self)]


#: Ranks present in a Euchre deck, low to high.
RANKS: tuple[Rank, ...] = (Rank.NINE, Rank.TEN, Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)

#: Total distinct cards in the deck. Used to size one-hot encodings.
NUM_CARDS = 24


@dataclass(frozen=True, order=True)
class Card:
    """An immutable playing card.

    ``order=True`` gives a deterministic sort (by rank then suit) which is handy
    for tests and for presenting a hand to a human; it has nothing to do with
    in-game card strength (that depends on trump -- see :func:`trick_value`).
    """

    rank: Rank
    suit: Suit

    @property
    def index(self) -> int:
        """A stable index in ``[0, 24)`` for one-hot encoding.

        Layout: ``suit * 6 + (rank - 9)``. This is a bijection with the deck, so
        :meth:`from_index` inverts it. It is independent of trump, which keeps
        the observation encoding stable across hands.
        """
        return int(self.suit) * len(RANKS) + (int(self.rank) - int(Rank.NINE))

    @classmethod
    def from_index(cls, index: int) -> Card:
        suit, rank_offset = divmod(index, len(RANKS))
        return cls(Rank(rank_offset + int(Rank.NINE)), Suit(suit))

    def __str__(self) -> str:
        return f"{self.rank.short}{self.suit.short}"


def make_deck() -> list[Card]:
    """Return a fresh, ordered 24-card Euchre deck."""
    return [Card(rank, suit) for suit in Suit for rank in RANKS]


# --------------------------------------------------------------------------- #
# Trump-aware predicates. Everything below depends on which suit is trump.
# `trump` is ``None`` only before a suit has been named (during the first
# bidding round we still know the *candidate* suit, so callers pass it in).
# --------------------------------------------------------------------------- #


def is_right_bower(card: Card, trump: Suit) -> bool:
    return card.rank == Rank.JACK and card.suit == trump


def is_left_bower(card: Card, trump: Suit) -> bool:
    return card.rank == Rank.JACK and card.suit == trump.same_color_suit


def effective_suit(card: Card, trump: Suit | None) -> Suit:
    """The suit a card behaves as *in play*.

    Identical to ``card.suit`` except the left bower, whose effective suit is
    trump. This is the function to use for "must follow suit" checks and for
    deciding what suit was led.
    """
    if trump is not None and is_left_bower(card, trump):
        return trump
    return card.suit


def is_trump(card: Card, trump: Suit | None) -> bool:
    return trump is not None and effective_suit(card, trump) == trump


def trick_value(card: Card, trump: Suit, lead_suit: Suit) -> int:
    """Strength of ``card`` for winning the current trick (higher wins).

    The ordering encoded here is the whole point of the bower rules:

    1. Right bower (Jack of trump) -- always highest.
    2. Left bower (Jack of same color) -- counts as trump, second highest.
    3. Any other trump, by rank.
    4. Any card following the led suit, by rank.
    5. Anything else (off-suit discard) -- cannot win the trick.

    The large constant gaps keep the tiers strictly separated regardless of
    rank values.
    """
    if is_right_bower(card, trump):
        return 1000
    if is_left_bower(card, trump):
        return 900
    if is_trump(card, trump):
        return 800 + int(card.rank)
    if effective_suit(card, trump) == lead_suit:
        return 100 + int(card.rank)
    return int(card.rank)
