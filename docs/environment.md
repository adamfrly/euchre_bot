# The environment: API, observations, actions, rewards

`euchre_bot.env.EuchreEnv` is a **turn-based, multi-agent, imperfect-information**
environment. This page is the contract your agents code against.

## Why it isn't a `gymnasium.Env`

Gymnasium models one agent stepping through time. Euchre has four agents acting
in sequence, each seeing only their own cards. The closest standard is
PettingZoo's **Agent–Environment–Cycle (AEC)** API, and this env borrows its
spirit without the dependency: at every step exactly one seat is *on turn*. We
kept the surface tiny and explicit so the control flow is readable while you
learn. If you later want to plug into RLlib/PettingZoo tooling, this is the
adapter boundary.

## The loop

```python
from euchre_bot.env import EuchreEnv

env = EuchreEnv(stick_the_dealer=True, allow_going_alone=True, reward_mode="zero_sum")
result = env.reset(seed=0)
while not result.terminated:
    seat = result.current_player
    action = my_agents[seat].act(result.observation, result.action_mask, result.info)
    result = env.step(action)
# result.rewards is {seat: reward} for all four seats
```

`reset()` and `step()` both return a **`StepResult`**:

| field | meaning |
|---|---|
| `observation` | `float32` vector for the seat on turn (`None` at terminal) |
| `action_mask` | `float32[NUM_ACTIONS]`, `1.0` = legal (`None` at terminal) |
| `current_player` | seat index on turn (`None` at terminal) |
| `terminated` | `True` once the hand is scored |
| `rewards` | `{seat: reward}`, populated only at terminal |
| `info` | `{"player_view": PlayerView}` mid-hand; outcome summary at terminal |

`EuchreEnv.observation_dim` (**174**) and `EuchreEnv.num_actions` (**35**) are
class attributes — use them to size your networks.

## Episode = one hand

An episode runs from deal through bidding and five tricks to scoring. **Rewards
are sparse**: every mid-hand step has reward 0, and the terminal step delivers
one reward per seat. This is realistic for card games and forces your algorithm
to do real credit assignment back through the hand.

- `reward_mode="zero_sum"` (default): scoring team `+points`, other team
  `-points` (the four seats sum to zero).
- `reward_mode="team"`: scoring team `+points`, other team `0`.

## The action space (35 discrete actions)

One flat space spans every phase; the mask exposes only what's legal now. See
`engine/actions.py`.

| ids | meaning |
|---|---|
| `0` | `PASS` (bidding) |
| `1`, `2` | `ORDER_UP`, `ORDER_UP_ALONE` (round 1) |
| `3–6`, `7–10` | name suit (partnered / alone) (round 2) |
| `11–34` | play or discard the card with that index |

The card block is indexed by `Card.index`, so action `11 + card.index` always
means that specific card whether you're discarding it or playing it. Helpers:
`actions.card_action`, `actions.card_of_action`, `actions.action_name`.

**Masking is mandatory.** The env raises `ValueError` on an illegal action — on
purpose, to catch the single most common agent bug. In a neural agent, set
illegal logits to `-inf` before sampling. See `docs/implementing_algorithms.md`.

## The observation vector

Built in `env/encoding.py`, everything is **relative to the seat on turn** so a
single shared policy can play all four seats (this is what makes self-play
sound — the network never sees its absolute seat number). Components:

- own hand (24) and all cards played so far this hand (24);
- the current trick by relative seat (3 × 24);
- the up-card (24) + a "still showing" flag (1);
- trump and led suit, each one-hot over {none, 4 suits} (5 + 5);
- dealer position relative to me (4); maker position relative to me (5);
- "alone" flag (1) and "maker is my team" flag (1);
- phase one-hot (4);
- tricks won and running game score, ordered (my team, other) and scaled (2 + 2).

If you change these features, `OBS_DIM` updates automatically and your network
input width changes with it — there is a single source of truth.

## The structured `PlayerView`

`result.info["player_view"]` is a `PlayerView` dataclass with the same
information in human terms (lists of `Card`, the legal action ids, trump, etc.).
**Scripted agents and the human CLI read this; learning agents should not** —
they must work from the vector + mask, which is exactly what carries no hidden
information.

## Inspecting a hand

`env.render()` returns a god's-eye text snapshot (all four hands) for debugging.
`python -m euchre_bot.play` lets you play a hand against the bots from the
terminal — the fastest way to confirm the rules match your expectations.
