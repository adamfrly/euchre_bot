"""Tests for trick resolution, follow-suit legality, and scoring."""

from euchre_bot.engine.cards import Card, Rank, Suit
from euchre_bot.engine.rules import (
    legal_plays,
    partner_of,
    score_hand,
    team_of,
    trick_winner,
)


def test_teams_and_partners():
    assert team_of(0) == team_of(2) == 0
    assert team_of(1) == team_of(3) == 1
    assert partner_of(0) == 2
    assert partner_of(1) == 3


def test_must_follow_led_suit_when_able():
    trump = Suit.SPADES
    hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.NINE, Suit.HEARTS), Card(Rank.KING, Suit.CLUBS)]
    legal = legal_plays(hand, lead_suit=Suit.HEARTS, trump=trump)
    assert set(legal) == {Card(Rank.ACE, Suit.HEARTS), Card(Rank.NINE, Suit.HEARTS)}


def test_any_card_when_void_in_led_suit():
    trump = Suit.SPADES
    hand = [Card(Rank.ACE, Suit.HEARTS), Card(Rank.KING, Suit.CLUBS)]
    legal = legal_plays(hand, lead_suit=Suit.DIAMONDS, trump=trump)
    assert set(legal) == set(hand)


def test_left_bower_must_follow_trump_lead():
    # Hearts trump: the J of diamonds is effectively a heart, so when a heart is
    # led the player is NOT void in trump and must follow with it.
    trump = Suit.HEARTS
    left_bower = Card(Rank.JACK, Suit.DIAMONDS)
    hand = [left_bower, Card(Rank.ACE, Suit.CLUBS)]
    legal = legal_plays(hand, lead_suit=Suit.HEARTS, trump=trump)
    assert legal == [left_bower]


def test_left_bower_does_not_count_as_its_printed_suit():
    # Diamonds led, hearts trump: the J of diamonds is a heart now, so the player
    # counts as void in diamonds and may play anything.
    trump = Suit.HEARTS
    left_bower = Card(Rank.JACK, Suit.DIAMONDS)
    hand = [left_bower, Card(Rank.ACE, Suit.CLUBS)]
    legal = legal_plays(hand, lead_suit=Suit.DIAMONDS, trump=trump)
    assert set(legal) == set(hand)


def test_trick_winner_trump_beats_lead():
    trump = Suit.SPADES
    plays = [
        (0, Card(Rank.ACE, Suit.HEARTS)),   # leads hearts
        (1, Card(Rank.NINE, Suit.SPADES)),  # trumps in
        (2, Card(Rank.KING, Suit.HEARTS)),
        (3, Card(Rank.TEN, Suit.HEARTS)),
    ]
    assert trick_winner(plays, trump) == 1


def test_trick_winner_left_bower_beats_offsuit_ace_of_trump_color():
    trump = Suit.HEARTS
    plays = [
        (0, Card(Rank.ACE, Suit.HEARTS)),   # ace of trump leads
        (1, Card(Rank.JACK, Suit.DIAMONDS)),  # left bower outranks ace of trump
        (2, Card(Rank.NINE, Suit.HEARTS)),
        (3, Card(Rank.KING, Suit.HEARTS)),
    ]
    assert trick_winner(plays, trump) == 1


def test_scoring_one_point_for_three_or_four_tricks():
    team, pts = score_hand(maker_team=0, tricks_by_team={0: 3, 1: 2}, went_alone=False)
    assert (team, pts) == (0, 1)


def test_scoring_march_two_points():
    team, pts = score_hand(maker_team=1, tricks_by_team={0: 0, 1: 5}, went_alone=False)
    assert (team, pts) == (1, 2)


def test_scoring_lone_march_four_points():
    team, pts = score_hand(maker_team=0, tricks_by_team={0: 5, 1: 0}, went_alone=True)
    assert (team, pts) == (0, 4)


def test_scoring_euchre_gives_defenders_two():
    team, pts = score_hand(maker_team=0, tricks_by_team={0: 2, 1: 3}, went_alone=False)
    assert (team, pts) == (1, 2)
