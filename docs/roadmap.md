# Roadmap / future improvements

## Next steps (from the v1/v2 experiments)

Prioritised by what the experiments in [`experiments.md`](experiments.md)
point to. The current best agents are `runs/ppo_league_noalone_v2/best.pt` and
`runs/ppo_league_v2_s1/best.pt`; they are statistically tied.

1. **Give the observation history.** This is the main lever left. Card play
   has not improved past the heuristic in any run: defence, overtaking a
   partner, and follow-suit choices are all unchanged. Add:
   * who played which card in each completed trick, so the agent can infer
     voids and read its partner;
   * the bidding history: who passed, who ordered up or called, and in which
     round;
   * the turned-down card during round 2, which `PlayerView` currently hides
     (`upcard` is ``None`` once it is turned down).

   Changing `OBS_DIM` invalidates old checkpoints. Retrain, and keep the
   previous champion as a score-blind / dimension-adapted gauntlet entry, or
   drop it.
2. **Make going alone learnable by splitting the bid decision.** Replace the
   flat "call suit X alone" actions with a two-step decision: *call?* and then
   *alone?*, each with its own head and entropy bonus. A binary alone head
   cannot collapse the way a single rare action in a 35-way softmax did. The
   mixture exploration in `PPOAgent(alone_explore=...)` collected lone-hand data
   but couldn't move the near-zero base probability (see experiments.md). It is
   worth +1.15 pts on hands strong enough to go alone.
3. **Keep the training setup that worked.** Use `--opponent league
   --random-scores` and the evaluation gauntlet, with `eval_gauntlet/point_diff`
   as the headline metric. Promote the newest best bot to `--champion` each
   round.
4. **Run 2-3 seeds per configuration.** The differences between variants are
   now around 0.02-0.05 pts/hand, about the noise of one run and one eval.
   Re-test the finalists on fresh deals with `evaluate` / `evaluate_games`
   before drawing conclusions.
5. **Then scale up.** Train longer with a wider network. v2 runs plateaued
   around iterations 4,000-5,000 at the current size.

## Longer-term ideas

Captured for later, not yet built. Ordered roughly by value-for-effort.

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

- **Full-game training** (episodes of hands to 10 points, rewarded on game
  outcome). Full-game *evaluation* exists (`evaluate_games`), and
  `--random-scores` teaches the score features, but training is still per hand.
- ~~Frozen-opponent pool~~: done as `training/league.py` (`--opponent league`).
- **Additional variants**: no-trump, farmer's hand, screw-the-dealer, 3/6-handed
  — the rules are isolated enough (`engine/rules.py`, env flags) to add cleanly.
