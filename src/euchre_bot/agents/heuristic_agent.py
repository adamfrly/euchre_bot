"""A hand-written, rule-of-thumb Euchre player.

This is deliberately simple -- it is a *yardstick*, not a champion. Self-play
win-rate is meaningless on its own (a policy is always ~50% against a copy of
itself), so you measure progress by how often your trained agent beats this
fixed opponent. A good agent should comfortably exceed it; if it cannot beat a
bot that only counts bowers and trump, something is wrong with your training.

The agent reasons over the structured :class:`PlayerView` carried in
``info["player_view"]`` -- it does **not** look at the observation vector. That
is the privilege of scripted agents; a learning agent must work from the vector.

Strategy in one breath: call trump when the hand is strong enough, lead your big
cards, win tricks as cheaply as possible, don't trump your partner's winners,
and throw away your weakest card when you can't win.
"""

from __future__ import annotations

import numpy as np

from ..engine import actions as A
from ..engine.cards import Card, Suit, is_left_bower, is_right_bower, is_trump, trick_value
from ..engine.rules import partner_of, trick_winner
from ..engine.state import Phase, PlayerView
from .base import Agent

# Hand-strength thresholds (in "strength points", see _suit_strength).
_CALL_THRESHOLD = 3
_ALONE_THRESHOLD = 7


def _suit_strength(hand: list[Card], trump: Suit) -> int:
    """A crude 0-ish..10 strength estimate for a hand if ``trump`` were trump."""
    strength = 0
    for card in hand:
        if is_right_bower(card, trump):
            strength += 3
        elif is_left_bower(card, trump):
            strength += 2
        elif is_trump(card, trump):
            strength += 1
        elif card.rank.name == "ACE":
            strength += 1  # off-suit ace: a likely trick
    return strength


class HeuristicAgent(Agent):
    def act(
        self,
        observation: np.ndarray,
        action_mask: np.ndarray,
        info: dict | None = None,
        *,
        deterministic: bool = False,
    ) -> int:
        if info is None or "player_view" not in info:
            raise ValueError("HeuristicAgent needs info['player_view']")
        view: PlayerView = info["player_view"]

        if view.phase == Phase.BID_ROUND_1:
            return self._bid_round_1(view)
        if view.phase == Phase.BID_ROUND_2:
            return self._bid_round_2(view)
        if view.phase == Phase.DISCARD:
            return self._discard(view)
        return self._play(view)

    # -- bidding --------------------------------------------------------- #

    def _bid_round_1(self, view: PlayerView) -> int:
        candidate = view.upcard.suit
        hand = list(view.hand)
        if view.seat == view.dealer:  # the dealer would gain the up-card
            hand = hand + [view.upcard]
        strength = _suit_strength(hand, candidate)
        if A.ORDER_UP_ALONE in view.legal_actions and strength >= _ALONE_THRESHOLD:
            return A.ORDER_UP_ALONE
        if strength >= _CALL_THRESHOLD:
            return A.ORDER_UP
        return A.PASS

    def _bid_round_2(self, view: PlayerView) -> int:
        forbidden = view.upcard.suit if view.upcard is not None else None
        best_suit, best_strength = None, -1
        for suit in Suit:
            if suit == forbidden:
                continue
            strength = _suit_strength(view.hand, suit)
            if strength > best_strength:
                best_suit, best_strength = suit, strength

        can_pass = A.PASS in view.legal_actions  # False when stuck as dealer
        if can_pass and best_strength < _CALL_THRESHOLD:
            return A.PASS
        alone = best_strength >= _ALONE_THRESHOLD
        action = A.call_suit_action(best_suit, alone=alone)
        # Going alone may be disabled; fall back to the partnered call.
        if action in view.legal_actions:
            return action
        return A.call_suit_action(best_suit, alone=False)

    def _discard(self, view: PlayerView) -> int:
        # Throw the weakest card -- lowest trick value, treating each card in its
        # own suit so non-trump low cards rank below any trump.
        trump = view.trump
        weakest = min(view.hand, key=lambda c: trick_value(c, trump, c.suit))
        return A.card_action(weakest)

    # -- card play ------------------------------------------------------- #

    def _play(self, view: PlayerView) -> int:
        trump = view.trump
        legal = [A.card_of_action(a) for a in view.legal_actions]

        if not view.current_trick:
            # Leading: play the strongest card we hold.
            best = max(legal, key=lambda c: trick_value(c, trump, c.suit))
            return A.card_action(best)

        led = view.led_suit
        winning_seat = trick_winner(view.current_trick, trump)
        winning_card = next(c for s, c in view.current_trick if s == winning_seat)
        winning_value = trick_value(winning_card, trump, led)

        if winning_seat == partner_of(view.seat):
            # Partner is winning -- don't waste a good card, dump the weakest.
            weakest = min(legal, key=lambda c: trick_value(c, trump, led))
            return A.card_action(weakest)

        # Opponent is winning: take it as cheaply as possible if we can.
        winners = [c for c in legal if trick_value(c, trump, led) > winning_value]
        if winners:
            cheapest = min(winners, key=lambda c: trick_value(c, trump, led))
            return A.card_action(cheapest)

        # Can't win -- slough the weakest legal card.
        weakest = min(legal, key=lambda c: trick_value(c, trump, led))
        return A.card_action(weakest)
