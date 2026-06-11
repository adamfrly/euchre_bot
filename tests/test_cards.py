"""Tests for the card vocabulary and the all-important bower logic."""

from euchre_bot.engine.cards import (
    NUM_CARDS,
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


def test_deck_is_24_unique_cards_with_bijective_indices():
    deck = make_deck()
    assert len(deck) == NUM_CARDS
    assert len(set(deck)) == NUM_CARDS
    indices = [c.index for c in deck]
    assert sorted(indices) == list(range(NUM_CARDS))
    for card in deck:
        assert Card.from_index(card.index) == card


def test_same_color_suit():
    assert Suit.HEARTS.same_color_suit == Suit.DIAMONDS
    assert Suit.DIAMONDS.same_color_suit == Suit.HEARTS
    assert Suit.CLUBS.same_color_suit == Suit.SPADES
    assert Suit.SPADES.same_color_suit == Suit.CLUBS


def test_right_and_left_bower_identification():
    trump = Suit.HEARTS
    right = Card(Rank.JACK, Suit.HEARTS)
    left = Card(Rank.JACK, Suit.DIAMONDS)
    assert is_right_bower(right, trump)
    assert not is_left_bower(right, trump)
    assert is_left_bower(left, trump)
    assert not is_right_bower(left, trump)


def test_left_bower_effective_suit_is_trump():
    trump = Suit.HEARTS
    left = Card(Rank.JACK, Suit.DIAMONDS)
    # Printed suit is diamonds, but it plays as a heart (trump).
    assert effective_suit(left, trump) == Suit.HEARTS
    assert is_trump(left, trump)
    # A non-left-bower diamond stays a diamond.
    assert effective_suit(Card(Rank.ACE, Suit.DIAMONDS), trump) == Suit.DIAMONDS


def test_trump_strength_ordering():
    trump = Suit.SPADES
    lead = Suit.SPADES
    right = Card(Rank.JACK, Suit.SPADES)
    left = Card(Rank.JACK, Suit.CLUBS)
    ace_trump = Card(Rank.ACE, Suit.SPADES)
    nine_trump = Card(Rank.NINE, Suit.SPADES)
    values = [trick_value(c, trump, lead) for c in (right, left, ace_trump, nine_trump)]
    assert values == sorted(values, reverse=True)
    # Right beats left beats any other trump.
    assert trick_value(right, trump, lead) > trick_value(left, trump, lead)
    assert trick_value(left, trump, lead) > trick_value(ace_trump, trump, lead)


def test_offsuit_cannot_outrank_led_suit():
    trump = Suit.SPADES
    lead = Suit.HEARTS
    led_card = Card(Rank.NINE, Suit.HEARTS)
    offsuit_ace = Card(Rank.ACE, Suit.DIAMONDS)  # high rank, wrong suit, not trump
    assert trick_value(led_card, trump, lead) > trick_value(offsuit_ace, trump, lead)
