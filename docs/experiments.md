# Experiment log

Results and findings from training runs, newest first. Prioritised next steps
live at the top of [`roadmap.md`](roadmap.md).

How to read the numbers:

* **pts/hand** is the mean points per hand for the agent's team minus the
  opponent's (`avg_point_diff`). It is the headline metric. A bot that bids
  cautiously can win *fewer* hands yet score *more* points, so hand win rate is
  misleading here.
* **H3-H6** are `HeuristicAgent(call_threshold=3..6)`. H3 is the default and
  calls trump far too loosely: against itself, the calling team is euchred
  38.5% of the time.
* Unless noted, results are re-measured on fresh deals the runs never evaluated
  on (4,000-5,000 hands, 400 games), with every bot playing the same deals.
  At that size, noise is about ±0.02-0.03 pts/hand and ±2.5% on game win rate.
* Play-style profiles come from `examples/analyze_play.py`.

## v2: league training, evaluation gauntlet, going-alone exploration

Three PPO runs, 6,000 iterations x 128 hands each:

* `--opponent league --random-scores`, seed 0 and seed 1, with
  `--alone-explore 0.05 --alone-explore-iters 3000`;
* the same league setup with no going-alone exploration, seed 0.

The champion is v1's `ppo_vs_self_v1/best.pt`.

| Bot (best.pt) | H3 | H4 | H5 | H6 | vs champion | Games won vs H3 / H5 / champion |
|---|---|---|---|---|---|---|
| champion (v1 PPO self-play) | +0.21 | +0.08 | +0.06 | +0.15 | — | 63% / 47% / — |
| league, no alone (it 4,800) | +0.26 | +0.13 | +0.10 | +0.20 | +0.03 | 66% / 51% / 45% |
| league + alone, seed 1 (it 5,650) | +0.25 | +0.11 | +0.09 | +0.20 | +0.03 | 66% / 49% / 48% |
| league + alone, seed 0 (it 5,600) | +0.25 | +0.10 | +0.08 | +0.17 | +0.01 | 68% / 55% / 44% |

(Points per hand, unless marked as games.)

Findings:

* **League training helped, modestly.** All three league bots beat the champion
  against every heuristic, by +0.02 to +0.05 pts/hand. They beat the
  tight-bidding H5, which v1's heuristic-trained bots lost to, by
  -0.14 to -0.17. Head to head with the champion they are about even:
  +0.01 to +0.03 pts/hand, but 44-48% of games. v1 self-play was already fairly
  robust, so the league added less than expected.
* **Going-alone exploration failed.** During exploration the bots went alone,
  but the rate fell to 0% once epsilon reached 0. The cause is the mixture
  design: the policy gradient for an action scales with the *base* policy's
  probability of it. Once pi(alone) collapsed near zero, the explored alone
  calls barely moved the logits, even when they paid off. The alone runs ended
  level with the no-alone run. Going alone is worth chasing: on the
  heuristic's lone hands it averages +2.77 pts, versus +1.62 on the same deals
  played with a partner.
* **Card play has not improved in either round.** Every bot defends the same as
  the heuristic: it euchres a calling opponent about 39% of the time, versus
  40%. It overtakes a partner who is already winning the trick about 22% of the
  time, versus 15% for the heuristic. About 80% of its follow-suit plays match
  the heuristic's choice. All gains so far came from bidding. The league bots
  are the most selective callers yet, euchred 9-14% of the time as makers.
  The likely limit is the observation: it has no record of who played which
  card in past tricks, no bidding history, and no turned-down card in round 2.
* **Bug found:** the heuristic named the turned-down suit in round 2 at
  thresholds above 3, because the up-card is hidden from its view by then. The
  heuristic now reads the callable suits from `legal_actions`, and a regression
  test covers it.

## v1: first pass of A2C / PPO / DQN

5,000 iterations x 128 hands each.

| Bot (best.pt) | Hands won vs H3 | pts/hand vs H3 | vs H5 |
|---|---|---|---|
| PPO self-play | 50.8% | +0.20 | -0.03 |
| PPO trained vs H3 | 50.6% | +0.25 | -0.17 |
| DQN trained vs H3 | 50.0% | +0.20 | -0.14 |
| A2C trained vs H3 | 38.9% | -0.10 | — |
| REINFORCE trained vs H3 (pre-existing run) | ~35% | -0.14 | — |
| baseline: H3 vs itself | 49.7% | 0.00 | -0.23 |

The H5 column was measured on a different, smaller set of deals than the v2
table. The same champion scored -0.03 against H5 there and +0.06 against H5 in
the v2 table.

Head to head, PPO self-play beat both the heuristic-trained PPO (+0.23 pts/hand)
and DQN (+0.24). The heuristic-trained PPO and DQN were even with each other.

Findings:

* **Training against H3 learned to exploit it.** Those bots called trump on only
  18-22% of hands, against H3's 51%. They let H3 overcall, then euchred it. As
  makers they scored +0.7 to +1.0 pts/hand. Against tighter bidders, the edge
  vanished. Self-play generalised best.
* **No bot ever went alone** (0-0.2% of makes).
* **A2C**, with one gradient step per batch, was clearly the weakest and was
  dropped from later rounds.
* DQN's value estimates were noisy: it sometimes called trump on worthless
  hands.
