"""Pure trick-taking and scoring rules for Euchre.

These functions are stateless: you hand them cards and a trump suit and they
answer a question. All game *flow* (dealing, bidding, turn order) lives in the
environment; keeping the rules pure makes them trivial to unit-test, which
matters because a single wrong comparison silently corrupts every training
episode.

Team convention used throughout the package: seats are ``0, 1, 2, 3`` arranged
clockwise. Partners sit across from each other, so ``team(seat) = seat % 2``.
Seats 0 & 2 are team 0; seats 1 & 3 are team 1.
"""

from __future__ import annotations

from .cards import Card, Suit, effective_suit, trick_value


def team_of(seat: int) -> int:
    """Return the partnership (0 or 1) a seat belongs to."""
    return seat % 2


def partner_of(seat: int) -> int:
    """Return the seat directly across the table."""
    return (seat + 2) % 4


def legal_plays(hand: list[Card], lead_suit: Suit | None, trump: Suit) -> list[Card]:
    """Return the subset of ``hand`` that may legally be played.

    If a suit has been led, the player **must follow suit** (play a card whose
    *effective* suit matches the led suit) when able. With no card of the led
    suit they may play anything. When leading (``lead_suit is None``) every card
    is legal.
    """
    if lead_suit is None:
        return list(hand)
    following = [card for card in hand if effective_suit(card, trump) == lead_suit]
    return following if following else list(hand)


def trick_winner(plays: list[tuple[int, Card]], trump: Suit) -> int:
    """Return the seat that wins a completed trick.

    ``plays`` is a list of ``(seat, card)`` in the order they were played; the
    first entry is the leader and its effective suit defines the led suit.
    """
    if not plays:
        raise ValueError("cannot resolve an empty trick")
    lead_suit = effective_suit(plays[0][1], trump)
    best_seat, best_card = plays[0]
    best_value = trick_value(best_card, trump, lead_suit)
    for seat, card in plays[1:]:
        value = trick_value(card, trump, lead_suit)
        if value > best_value:
            best_seat, best_card, best_value = seat, card, value
    return best_seat


def score_hand(
    maker_team: int, tricks_by_team: dict[int, int], went_alone: bool
) -> tuple[int, int]:
    """Score a completed hand.

    Args:
        maker_team: The partnership that named trump.
        tricks_by_team: Tricks won, keyed by team (the two values sum to 5).
        went_alone: Whether the maker played without their partner.

    Returns:
        ``(scoring_team, points)`` -- exactly one team scores in Euchre.

    Rules implemented:
        * Makers win 3 or 4 tricks ............... 1 point.
        * Makers win all 5 (a "march") ........... 2 points (4 if alone).
        * Makers win fewer than 3 ("euchred") .... 2 points to the defenders.
        * A lone march is the only way to score 4.
    """
    defender_team = 1 - maker_team
    maker_tricks = tricks_by_team.get(maker_team, 0)

    if maker_tricks < 3:
        return defender_team, 2  # euchred
    if maker_tricks == 5:
        return maker_team, (4 if went_alone else 2)  # march
    return maker_team, 1
