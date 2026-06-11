"""Turn a :class:`~euchre_bot.engine.state.PlayerView` into arrays an agent can learn from.

The environment hands agents two NumPy arrays each step:

* an **observation** -- a flat ``float32`` vector that numericises everything in
  the player's view (their hand, public history, trump, phase, table geometry);
* an **action mask** -- a ``float32`` vector of length
  :data:`~euchre_bot.engine.actions.NUM_ACTIONS` that is ``1.0`` for legal
  actions and ``0.0`` otherwise.

Everything here is *relative to the player on turn*: positions are encoded as
"how many seats clockwise from me", so the same network can play every seat.
That seat-relative framing is what makes a single shared policy (self-play)
sound -- the network never needs to know its absolute seat number.

Keeping encoding in one small module matters: when you later implement an
agent, the network's input size is exactly :data:`OBS_DIM` and its output size
is :data:`~euchre_bot.engine.actions.NUM_ACTIONS`. If you change the features
here, those dimensions change in lockstep -- there is a single source of truth.
"""

from __future__ import annotations

import numpy as np

from ..engine.actions import NUM_ACTIONS
from ..engine.cards import NUM_CARDS, Card, Suit
from ..engine.state import Phase, PlayerView

#: Order of phases used for the one-hot phase feature.
_PHASES = (Phase.BID_ROUND_1, Phase.BID_ROUND_2, Phase.DISCARD, Phase.PLAY)


def _card_multi_hot(cards: list[Card]) -> np.ndarray:
    vec = np.zeros(NUM_CARDS, dtype=np.float32)
    for card in cards:
        vec[card.index] = 1.0
    return vec


def _suit_one_hot(suit: Suit | None) -> np.ndarray:
    """5-wide: index 0 means "no suit", indices 1-4 are the suits."""
    vec = np.zeros(5, dtype=np.float32)
    vec[0 if suit is None else int(suit) + 1] = 1.0
    return vec


def _rel_seat_one_hot(seat: int | None, me: int, width: int, include_none: bool) -> np.ndarray:
    vec = np.zeros(width, dtype=np.float32)
    if seat is None:
        if include_none:
            vec[0] = 1.0
        return vec
    rel = (seat - me) % 4
    vec[rel + (1 if include_none else 0)] = 1.0
    return vec


def encode_observation(view: PlayerView) -> np.ndarray:
    """Flatten a player's view into a fixed-length ``float32`` vector."""
    me = view.seat
    parts: list[np.ndarray] = []

    # What I hold, and everything that has been played this hand (public memory).
    parts.append(_card_multi_hot(view.hand))
    parts.append(_card_multi_hot(view.played_cards))

    # Cards already on the table this trick, by seat relative to me (1, 2, 3).
    trick = np.zeros(3 * NUM_CARDS, dtype=np.float32)
    for seat, card in view.current_trick:
        rel = (seat - me) % 4
        if rel != 0:  # rel 0 is me; I have not played to this trick yet
            trick[(rel - 1) * NUM_CARDS + card.index] = 1.0
    parts.append(trick)

    # The up-card, plus a flag for whether it is still showing.
    parts.append(_card_multi_hot([view.upcard] if view.upcard is not None else []))
    parts.append(np.array([1.0 if view.upcard is not None else 0.0], dtype=np.float32))

    # Trump, current led suit.
    parts.append(_suit_one_hot(view.trump))
    parts.append(_suit_one_hot(view.led_suit))

    # Table geometry, all relative to me.
    parts.append(_rel_seat_one_hot(view.dealer, me, width=4, include_none=False))
    parts.append(_rel_seat_one_hot(view.maker, me, width=5, include_none=True))

    # Bidding / partnership flags.
    my_team = me % 2
    maker_is_my_team = view.maker is not None and (view.maker % 2) == my_team
    parts.append(np.array([1.0 if view.alone else 0.0], dtype=np.float32))
    parts.append(np.array([1.0 if maker_is_my_team else 0.0], dtype=np.float32))

    # Phase one-hot.
    phase_vec = np.zeros(len(_PHASES), dtype=np.float32)
    if view.phase in _PHASES:
        phase_vec[_PHASES.index(view.phase)] = 1.0
    parts.append(phase_vec)

    # Tricks taken and game score, ordered (my team, other team), lightly scaled.
    other = 1 - my_team
    tricks = np.array([view.tricks_won[my_team], view.tricks_won[other]], dtype=np.float32)
    scores = np.array([view.scores[my_team], view.scores[other]], dtype=np.float32)
    parts.append(tricks / 5.0)
    parts.append(scores / 10.0)

    return np.concatenate(parts)


def action_mask(legal_actions: list[int]) -> np.ndarray:
    """Build a ``float32`` mask: ``1.0`` at every legal action id, else ``0.0``."""
    mask = np.zeros(NUM_ACTIONS, dtype=np.float32)
    mask[legal_actions] = 1.0
    return mask


#: Length of the observation vector. Computed once from a throwaway view so the
#: rest of the codebase (and your networks) can import a single constant.
def _compute_obs_dim() -> int:

    dummy = PlayerView(
        seat=0,
        phase=Phase.PLAY,
        hand=[],
        legal_actions=[],
        dealer=0,
        trump=None,
        maker=None,
        alone=False,
        upcard=None,
        current_trick=[],
        led_suit=None,
        played_cards=[],
        tricks_won=[0, 0],
        scores=[0, 0],
    )
    return int(encode_observation(dummy).shape[0])


OBS_DIM = _compute_obs_dim()
