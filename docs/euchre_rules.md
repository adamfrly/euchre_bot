# Euchre rules as implemented

This project implements **standard 4-player Euchre** with **going-alone
(loners)** and **stick-the-dealer**. This page is the spec the engine
(`euchre_bot.engine`) and environment (`euchre_bot.env`) follow. If your house
rules differ, this is the file to reconcile against — the code is small and the
rules live in `engine/rules.py` and `env/euchre_env.py`.

## Players, teams, deck

- Four players in two fixed partnerships. Seats `0,1,2,3` sit clockwise;
  partners are across the table. **Team 0 = seats 0 & 2, Team 1 = seats 1 & 3**
  (`team_of(seat) = seat % 2`).
- 24-card deck: **9, 10, J, Q, K, A** in each of the four suits.
- Each player is dealt 5 cards; the remaining 4 form the kitty, whose top card
  is turned face-up (the **up-card**).

## The bowers (the part everything hinges on)

Once a trump suit is set:

- **Right bower** = Jack of the trump suit — the highest card in the game.
- **Left bower** = Jack of the *other suit of the same color* — it counts **as
  a trump card** (its effective suit becomes trump) and is the second-highest
  card.

Example: hearts are trump → right bower is J♥, left bower is J♦, and J♦ is
treated as a heart for following suit and for winning tricks. The engine routes
every suit decision through `effective_suit(card, trump)` so the left bower is
handled in exactly one place.

**Trump ranking, high to low:** J(trump) > J(same color) > A > K > Q > 10 > 9.
**Off-suit ranking:** A > K > Q > J > 10 > 9 (minus any jack that is a bower).

## Bidding

Bidding goes clockwise starting to the dealer's left.

**Round 1 — the up-card.** Each player may *order it up* (accept the up-card's
suit as trump) or *pass*. If anyone orders up, the **dealer picks the up-card
into their hand and discards one card** face down, and the up-card's suit is
trump. The player who ordered up (or the dealer who "takes it up") makes trump;
their partnership is the **makers**.

**Round 2 — name a suit.** If all four pass, the up-card is turned down. Going
clockwise from the dealer's left again, each player may *name any suit other
than the up-card's suit* as trump, or pass.

- **Stick the dealer (on by default):** in round 2 the dealer **may not pass**
  and must name a suit. With this rule a hand is always played.
- If stick-the-dealer is **off** and all four pass in round 2, the hand is
  thrown in and redealt (the engine redeals with the next dealer).

## Going alone (loners)

When a player makes trump (round 1 or round 2) they may declare **going alone**.
Their partner sits out the entire hand (plays no cards). The other three play as
normal; each trick therefore has three cards instead of four.

## Play

- The player to the dealer's left leads the first trick (skipping a lone maker's
  sitting-out partner if necessary).
- **Follow suit if you can:** you must play a card whose *effective* suit
  matches the led suit when you hold one; otherwise you may play anything.
- A trick is won by the highest trump played, or — if no trump was played — by
  the highest card of the led suit. The winner leads the next trick.
- Five tricks are played.

## Scoring

Exactly one team scores per hand (`score_hand` in `engine/rules.py`):

| Outcome | Points |
|---|---|
| Makers win 3 or 4 tricks | **1** to the makers |
| Makers win all 5 (a *march*) | **2** to the makers |
| Lone maker wins all 5 | **4** to the makers |
| Makers win fewer than 3 (*euchred*) | **2** to the defenders |

A game is traditionally played to 10 points. The environment scores at the
**hand** level (one hand = one episode); stacking hands into a full game to 10
is a small wrapper you can add on top, and the observation already reserves a
slot for the running game score.

## What is intentionally simplified

- **Dealing ritual.** Real Euchre deals in packets of 2 and 3; we deal a uniform
  random 5 cards each. The resulting hand distribution is identical, and nothing
  downstream depends on deal order.
- **Farmer's hand, no-trump, screw-the-dealer variants, 5/6-handed play** are
  not implemented. Hooks exist (`stick_the_dealer`, `allow_going_alone`) and the
  rules are isolated enough that adding a variant is a contained change.
