"""Pure Euchre rules engine (no RL, no NumPy, no torch).

Import the vocabulary you need directly from the submodules; this package
re-exports the most common names for convenience.
"""

from . import actions
from .cards import (
    NUM_CARDS,
    RANKS,
    Card,
    Rank,
    Suit,
    effective_suit,
    is_left_bower,
    is_right_bower,
    is_trump,
    make_deck,
    trick_value,
)
from .rules import legal_plays, partner_of, score_hand, team_of, trick_winner
from .state import BIDDING_PHASES, GameState, Phase, PlayerView

__all__ = [
    "actions",
    "NUM_CARDS",
    "RANKS",
    "Card",
    "Rank",
    "Suit",
    "effective_suit",
    "is_left_bower",
    "is_right_bower",
    "is_trump",
    "make_deck",
    "trick_value",
    "legal_plays",
    "partner_of",
    "score_hand",
    "team_of",
    "trick_winner",
    "BIDDING_PHASES",
    "GameState",
    "Phase",
    "PlayerView",
]
