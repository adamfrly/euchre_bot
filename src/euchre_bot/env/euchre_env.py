"""The Euchre environment: a turn-based, multi-agent, imperfect-information game.

Why this does not subclass ``gymnasium.Env``
---------------------------------------------
Gymnasium models a *single* agent stepping through time. Euchre has four agents
who act one after another, each seeing only their own cards. The closest
standard is PettingZoo's "Agent-Environment-Cycle" (AEC) API, and this class
borrows its spirit: at any moment exactly one seat is *on turn*, you read that
seat's observation, choose an action, and :meth:`step` advances to whoever acts
next. We keep a small, explicit API (rather than depending on PettingZoo) so the
control flow is easy to read while you are learning.

Episode = one hand
-------------------
By default an episode is a single hand: deal -> bid -> five tricks -> score.
Rewards are sparse and arrive only at the end (this is realistic for card games
and is a good forcing function for the credit-assignment machinery in your
algorithms). The terminal :class:`StepResult` carries a reward for *every*
seat; the training loop attributes each seat's reward back over the actions
that seat took.

Variant implemented: standard 4-player Euchre, going-alone (loners), and
stick-the-dealer (on by default). See ``docs/euchre_rules.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..engine import actions as A
from ..engine.cards import Suit, effective_suit, make_deck
from ..engine.rules import (
    legal_plays,
    partner_of,
    score_hand,
    team_of,
    trick_winner,
)
from ..engine.state import GameState, Phase, PlayerView
from . import encoding

NUM_PLAYERS = 4
CARDS_PER_HAND = 5


@dataclass
class StepResult:
    """What :meth:`EuchreEnv.reset` and :meth:`EuchreEnv.step` return.

    On a non-terminal step, ``observation``/``action_mask``/``current_player``
    describe the seat that must act next. On the terminal step they are ``None``
    and ``rewards`` maps every seat to its end-of-hand reward.
    """

    observation: np.ndarray | None
    action_mask: np.ndarray | None
    current_player: int | None
    terminated: bool
    rewards: dict[int, float]
    info: dict[str, Any] = field(default_factory=dict)


class EuchreEnv:
    """A single-hand Euchre environment for self-play reinforcement learning.

    Args:
        stick_the_dealer: If ``True`` (default), the dealer may not pass in the
            second bidding round and is forced to name a suit. If ``False`` and
            all four players pass twice, the hand is thrown in and redealt.
        allow_going_alone: If ``False``, the ``*_ALONE`` actions are masked out.
        reward_mode: ``"zero_sum"`` (default) gives the scoring team ``+points``
            and the other team ``-points``; ``"team"`` gives the scoring team
            ``+points`` and the other team ``0``.
        seed: Optional seed for the internal random generator.
    """

    def __init__(
        self,
        stick_the_dealer: bool = True,
        allow_going_alone: bool = True,
        reward_mode: str = "zero_sum",
        seed: int | None = None,
    ) -> None:
        if reward_mode not in ("zero_sum", "team"):
            raise ValueError(f"unknown reward_mode {reward_mode!r}")
        self.stick_the_dealer = stick_the_dealer
        self.allow_going_alone = allow_going_alone
        self.reward_mode = reward_mode

        self._rng = np.random.default_rng(seed)
        self._scores = [0, 0]          # running game score (placeholder for full-game play)
        self.state: GameState | None = None

    # Static dimensions, handy for building networks.
    observation_dim = encoding.OBS_DIM
    num_actions = A.NUM_ACTIONS

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None, dealer: int | None = None) -> StepResult:
        """Start a new hand and return the first player's observation."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if dealer is None:
            dealer = int(self._rng.integers(NUM_PLAYERS))
        self._scores = [0, 0]
        self._deal(dealer)
        return self._observe()

    def _deal(self, dealer: int) -> None:
        deck = make_deck()
        self._rng.shuffle(deck)  # type: ignore[arg-type]
        # A uniform random deal; we skip the ceremonial 2-3 packet dealing
        # because it does not change the distribution of resulting hands.
        hands = [deck[i * CARDS_PER_HAND : (i + 1) * CARDS_PER_HAND] for i in range(NUM_PLAYERS)]
        for hand in hands:
            hand.sort()
        kitty = deck[NUM_PLAYERS * CARDS_PER_HAND :]
        self.state = GameState(
            dealer=dealer,
            hands=hands,
            upcard=kitty[0],
            kitty=kitty,
            current_player=(dealer + 1) % NUM_PLAYERS,
        )

    # ------------------------------------------------------------------ #
    # Legal actions and observation construction
    # ------------------------------------------------------------------ #

    def legal_actions(self) -> list[int]:
        """The action ids the player on turn may legally choose right now."""
        s = self.state
        assert s is not None

        if s.phase == Phase.BID_ROUND_1:
            acts = [A.PASS, A.ORDER_UP]
            if self.allow_going_alone:
                acts.append(A.ORDER_UP_ALONE)
            return acts

        if s.phase == Phase.BID_ROUND_2:
            acts: list[int] = []
            forbidden = s.upcard.suit  # the suit that was turned down
            dealer_is_stuck = self.stick_the_dealer and s.current_player == s.dealer
            if not dealer_is_stuck:
                acts.append(A.PASS)
            for suit in Suit:
                if suit == forbidden:
                    continue
                acts.append(A.call_suit_action(suit, alone=False))
                if self.allow_going_alone:
                    acts.append(A.call_suit_action(suit, alone=True))
            return acts

        if s.phase == Phase.DISCARD:
            return [A.card_action(card) for card in s.hands[s.dealer]]

        if s.phase == Phase.PLAY:
            playable = legal_plays(s.hands[s.current_player], self._led_suit(), s.trump)
            return [A.card_action(card) for card in playable]

        return []

    def _led_suit(self) -> Suit | None:
        s = self.state
        assert s is not None
        if not s.current_trick:
            return None
        return effective_suit(s.current_trick[0][1], s.trump)

    def _player_view(self) -> PlayerView:
        s = self.state
        assert s is not None
        return PlayerView(
            seat=s.current_player,
            phase=s.phase,
            hand=list(s.hands[s.current_player]),
            legal_actions=self.legal_actions(),
            dealer=s.dealer,
            trump=s.trump,
            maker=s.maker,
            alone=s.alone,
            upcard=s.upcard if s.upcard_visible else None,
            current_trick=list(s.current_trick),
            led_suit=self._led_suit(),
            played_cards=list(s.played_cards),
            tricks_won=list(s.tricks_won),
            scores=list(self._scores),
        )

    def _observe(self) -> StepResult:
        view = self._player_view()
        return StepResult(
            observation=encoding.encode_observation(view),
            action_mask=encoding.action_mask(view.legal_actions),
            current_player=view.seat,
            terminated=False,
            rewards={},
            info={"player_view": view},
        )

    # ------------------------------------------------------------------ #
    # Stepping the environment
    # ------------------------------------------------------------------ #

    def step(self, action: int) -> StepResult:
        """Apply ``action`` for the player on turn and advance the hand."""
        s = self.state
        assert s is not None, "call reset() before step()"
        if s.phase == Phase.DONE:
            raise RuntimeError("hand is over; call reset()")
        if action not in self.legal_actions():
            raise ValueError(
                f"illegal action {A.action_name(action)} in phase {s.phase.name}"
            )

        if s.phase == Phase.BID_ROUND_1:
            self._step_bid_round_1(action)
        elif s.phase == Phase.BID_ROUND_2:
            self._step_bid_round_2(action)
        elif s.phase == Phase.DISCARD:
            self._step_discard(action)
        else:  # Phase.PLAY
            self._step_play(action)

        if s.phase == Phase.DONE:
            return StepResult(
                observation=None,
                action_mask=None,
                current_player=None,
                terminated=True,
                rewards=self._terminal_rewards(),
                info={
                    "scoring_team": s.scoring_team,
                    "points": s.points,
                    "tricks_won": list(s.tricks_won),
                    "maker": s.maker,
                    "alone": s.alone,
                    "trump": s.trump,
                },
            )
        return self._observe()

    def _step_bid_round_1(self, action: int) -> None:
        s = self.state
        assert s is not None
        if action == A.PASS:
            s.passes += 1
            if s.passes == NUM_PLAYERS:
                # Everyone passed: turn the up-card down and open round 2.
                s.phase = Phase.BID_ROUND_2
                s.passes = 0
                s.upcard_visible = False
                s.current_player = (s.dealer + 1) % NUM_PLAYERS
            else:
                s.current_player = (s.current_player + 1) % NUM_PLAYERS
            return

        # ORDER_UP / ORDER_UP_ALONE: the up-card's suit becomes trump and the
        # dealer picks the up-card into hand, then discards.
        s.trump = s.upcard.suit
        s.maker = s.current_player
        s.alone = action == A.ORDER_UP_ALONE
        if s.alone:
            s.sitting_out = partner_of(s.maker)
        s.hands[s.dealer].append(s.upcard)
        s.hands[s.dealer].sort()
        s.upcard_visible = False
        s.phase = Phase.DISCARD
        s.current_player = s.dealer

    def _step_bid_round_2(self, action: int) -> None:
        s = self.state
        assert s is not None
        if action == A.PASS:
            s.passes += 1
            if s.passes == NUM_PLAYERS:
                # Only reachable when stick_the_dealer is off: throw in & redeal.
                self._deal((s.dealer + 1) % NUM_PLAYERS)
                return
            s.current_player = (s.current_player + 1) % NUM_PLAYERS
            return

        suit, alone = A.decode_call(action)
        s.trump = suit
        s.maker = s.current_player
        s.alone = alone
        if alone:
            s.sitting_out = partner_of(s.maker)
        self._begin_play()

    def _step_discard(self, action: int) -> None:
        s = self.state
        assert s is not None
        card = A.card_of_action(action)
        s.hands[s.dealer].remove(card)
        self._begin_play()

    def _begin_play(self) -> None:
        s = self.state
        assert s is not None
        s.phase = Phase.PLAY
        s.current_player = self._next_active((s.dealer) % NUM_PLAYERS)

    def _next_active(self, seat: int) -> int:
        """Next seat clockwise that is actually playing (skips a lone maker's partner)."""
        s = self.state
        assert s is not None
        nxt = (seat + 1) % NUM_PLAYERS
        while nxt == s.sitting_out:
            nxt = (nxt + 1) % NUM_PLAYERS
        return nxt

    def _step_play(self, action: int) -> None:
        s = self.state
        assert s is not None
        card = A.card_of_action(action)
        s.hands[s.current_player].remove(card)
        s.current_trick.append((s.current_player, card))
        s.played_cards.append(card)

        if len(s.current_trick) < s.active_players():
            s.current_player = self._next_active(s.current_player)
            return

        # Trick complete: award it and lead from the winner.
        winner = trick_winner(s.current_trick, s.trump)
        s.tricks_won[team_of(winner)] += 1
        s.completed_tricks += 1
        s.current_trick = []

        if s.completed_tricks == CARDS_PER_HAND:
            self._finish_hand()
        else:
            s.current_player = winner

    def _finish_hand(self) -> None:
        s = self.state
        assert s is not None
        assert s.maker is not None
        scoring_team, points = score_hand(
            maker_team=team_of(s.maker),
            tricks_by_team={0: s.tricks_won[0], 1: s.tricks_won[1]},
            went_alone=s.alone,
        )
        s.scoring_team = scoring_team
        s.points = points
        self._scores[scoring_team] += points
        s.phase = Phase.DONE

    def _terminal_rewards(self) -> dict[int, float]:
        s = self.state
        assert s is not None
        assert s.scoring_team is not None
        rewards: dict[int, float] = {}
        for seat in range(NUM_PLAYERS):
            if team_of(seat) == s.scoring_team:
                rewards[seat] = float(s.points)
            elif self.reward_mode == "zero_sum":
                rewards[seat] = -float(s.points)
            else:
                rewards[seat] = 0.0
        return rewards

    # ------------------------------------------------------------------ #
    # Convenience for human play / debugging
    # ------------------------------------------------------------------ #

    @property
    def current_player(self) -> int | None:
        return None if self.state is None else self.state.current_player

    def render(self) -> str:
        """Return a compact text snapshot of the current state (god's-eye)."""
        s = self.state
        if s is None:
            return "<no hand dealt>"
        lines = [
            f"phase={s.phase.name} dealer={s.dealer} turn={s.current_player} "
            f"trump={s.trump.name if s.trump else '-'} maker={s.maker} alone={s.alone}",
            f"upcard={s.upcard if s.upcard_visible else '-'} "
            f"tricks={s.tricks_won} trick={[(seat, str(c)) for seat, c in s.current_trick]}",
        ]
        for seat in range(NUM_PLAYERS):
            lines.append(f"  seat {seat}: {[str(c) for c in s.hands[seat]]}")
        return "\n".join(lines)
