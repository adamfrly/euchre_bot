"""Mutable game-state container and the per-player view handed to agents.

:class:`GameState` is the full, god's-eye record of a hand -- it knows every
player's cards. The environment owns one and mutates it as the hand proceeds.
Agents must **never** see it directly (that would leak hidden information);
instead the environment derives a :class:`PlayerView` for the player on turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from .cards import Card, Suit


class Phase(Enum):
    """The decision phases of a hand, in the order they occur."""

    BID_ROUND_1 = auto()   # accept or pass on the turned-up card
    BID_ROUND_2 = auto()   # name a different suit, or pass
    DISCARD = auto()       # dealer discards after picking up the up-card
    PLAY = auto()          # play out the five tricks
    DONE = auto()          # hand complete; rewards available


#: Phases in which the agent is choosing among the bidding actions rather than
#: a card. Used by the encoder when building masks.
BIDDING_PHASES = (Phase.BID_ROUND_1, Phase.BID_ROUND_2)


@dataclass
class GameState:
    """Complete hidden state of a single Euchre hand."""

    dealer: int
    hands: list[list[Card]]                 # hands[seat] -> that seat's cards
    upcard: Card                            # card turned up from the kitty
    kitty: list[Card]                       # remaining undealt cards (unused in play)

    phase: Phase = Phase.BID_ROUND_1
    current_player: int = 0

    trump: Suit | None = None
    maker: int | None = None                # seat that named trump
    alone: bool = False
    sitting_out: int | None = None          # partner of a lone maker, if any

    # Bidding bookkeeping.
    passes: int = 0                         # consecutive passes in the current round
    upcard_visible: bool = True             # turned down once round 1 ends

    # Play bookkeeping.
    current_trick: list[tuple[int, Card]] = field(default_factory=list)
    tricks_won: list[int] = field(default_factory=lambda: [0, 0])  # by team
    completed_tricks: int = 0
    played_cards: list[Card] = field(default_factory=list)         # public memory

    # Filled in once phase == DONE.
    scoring_team: int | None = None
    points: int = 0

    def active_players(self) -> int:
        """Number of players dealt into the trick phase (3 if someone is alone)."""
        return 3 if self.alone else 4


@dataclass
class PlayerView:
    """Everything the player on turn is allowed to know.

    This is what baseline/heuristic agents and the human CLI read. RL agents
    consume the flat vector from the encoder instead, but the same information
    is present in both -- the vector is just this view, numericised.
    """

    seat: int
    phase: Phase
    hand: list[Card]
    legal_actions: list[int]

    dealer: int
    trump: Suit | None
    maker: int | None
    alone: bool

    upcard: Card | None                     # None after it is turned down
    current_trick: list[tuple[int, Card]]   # (seat, card) played so far this trick
    led_suit: Suit | None
    played_cards: list[Card]                # all cards seen this hand
    tricks_won: list[int]                   # by team
    scores: list[int]                       # running game score, by team
