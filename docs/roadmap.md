# Roadmap / future improvements

Ideas captured for later, not yet built. Ordered roughly by value-for-effort.

## 1. PIMC "final boss" agent (Perfect-Information Monte Carlo)

A strong search-based opponent to benchmark the RL agents against. **Not built
yet — deferred on purpose.** Conceptually sound and computationally tractable
(real-time, sub-second per decision).

How it would work:
1. **Sample** many full deals consistent with everything the agent knows — its
   own hand, the up-card, all cards played, and **observed voids** (a player who
   failed to follow suit is provably out of that suit, which constrains the
   sampled worlds). Bias the sampling using bidding signals (the maker likely
   holds trump strength).
2. For each sampled world, **solve the trick-play subgame exactly** as a
   perfect-information ("double-dummy") game — the per-deal tree is tiny
   (branching ≤ 5, heavy follow-suit pruning), so alpha-beta + a transposition
   table solves it in well under a millisecond.
3. **Average** the outcome per candidate action and play the best one. ~100–1000
   sampled worlds per decision is plenty and stays real-time.

Why it fits this repo: the engine already has the perfect-information machinery
a double-dummy solver needs (`trick_winner`, `legal_plays`, `score_hand`), and
`PlayerView` exposes exactly the public info (hand, played cards, voids) to
condition world-sampling on. It would slot in as just another `Agent` and drop
straight into `evaluate(...)` as a fixed benchmark.

Known limitation: PIMC is very strong but **not optimal** — it suffers from
*strategy fusion* (implicitly assumes it can act differently in different worlds
at the same information set) and underweights information-hiding/signaling. Good
enough to crush the heuristic and pressure the RL agents; beatable in principle.

## 2. Game-theoretically optimal play (CFR / Nash)

The "truly optimal" target: solve the full imperfect-information game (including
bidding/signaling) toward a Nash equilibrium with the Counterfactual Regret
Minimization family (MCCFR + state abstraction — the poker-bot lineage). Well-
defined and tractable *with effort*; a substantial engineering project, much
heavier than PIMC, and "optimal" only in the equilibrium sense (best vs a best-
responding opponent, not vs a specific one).

## 3. Search-augmented / hybrid RL (the real ceiling)

Where an RL agent could plausibly match or beat PIMC: pair a learned value/policy
network with decision-time search — AlphaZero-style, or ReBeL/DeepStack-style
subgame solving for imperfect information — or use the RL value net to evaluate
PIMC's sampled worlds and bias world sampling. Each method covers the other's
weakness. This is the most interesting long-term direction and a natural way to
let the learning agents surpass the PIMC final boss.

## Smaller items

- **Full-game wrapper** (play hands to 10 points) on top of the per-hand episode,
  using the observation's reserved game-score slot.
- **Frozen-opponent pool** for self-play stability (sample past checkpoints as
  opponents to avoid the policy chasing its own tail).
- **Additional variants**: no-trump, farmer's hand, screw-the-dealer, 3/6-handed
  — the rules are isolated enough (`engine/rules.py`, env flags) to add cleanly.
