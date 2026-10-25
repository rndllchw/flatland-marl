# Experiment log

Every experiment we run, in order: the question, how to rerun it, what happened, what we concluded.
Add an entry for each new run. Results that go in the report must come from `scripts/evaluate.py`
output in `results/` or from a run's `eval_log.csv`, never from `train_log.csv` alone (training games
use sampled actions and can make a policy that never moves look half-decent; see E1).

## Conventions

**Seed ranges** (scenario seeds; `make_scenario_pool.py --seed` is the first one):

| Range | Used for |
|---|---|
| 0 – 999,999 | training pools and `make_env` training games |
| 1,000,000 – 1,999,999 | validation pools (tuning and model selection) |
| 2,000,000 and up | frozen test sets (`eval_sets/test`, built once, untouched until configs are frozen on 1 Nov) |

Training seeds (`train.py --seed`) are separate: 0, 1, 2 for final runs. Greedy checks during training
use game seeds 1,000,000 + i (with `source: pool` they only pick which training-pool games to replay).

**Judging a run:** the greedy check (`eval_log.csv`) and `evaluate.py` play the actor's top action;
always compare against `checkpoint` and `heuristic` on the same scenarios; report the share of trains
that arrived next to the score.

## Infrastructure measurements

| Date | What | Result |
|---|---|---|
| 17 Oct | Simulator speed, 6 trains, one process (M2 Max) | ~830 steps/s (checkpoint obs), ~280 steps/s (heuristic) |
| 18 Oct | Reset time, scene 1, 5–10 trains, `line_length=3` | `make_env` 8.25 s vs pool replay 0.36 s (23×) |
| 18 Oct | Pool generation, 64 scenarios, `line_length=3`, 10 workers | 51 s (~8 s per scenario per worker) |
| 18 Oct | 27 example scenarios × 5 policies, 8 workers | 51 s |
| 21 Oct | Pool generation, 2,000 scenarios, scene 1, 5–10 trains, `line_length=2`, 10 workers | 1045 s (17.4 min) |
| 21 Oct | Pool generation, 50 scenarios, same spec, seeds from 1,000,000, 10 workers | 4 s (cause of the speed difference not investigated) |

## E0: baselines on the 27 example scenarios (18 Oct)

- **Question:** how do the reference policies do once trains interact?
- **Rerun:** `python scripts/evaluate.py --policies do_nothing random forward checkpoint heuristic --out results/baselines_example_curriculum --workers 8`
- **Scenarios:** starter kit's 27 examples (scenes 1/4/5 × max route length 2/3/4 × 1/10/25 trains), each with its own breakdowns + departure delays.
- **Outputs:** `results/baselines_example_curriculum/`

| Policy | Score 1 / 10 / 25 trains | Arrived 1 / 10 / 25 |
|---|---|---|
| heuristic | 0.982 / 0.654 / 0.536 | 100% / 81% / 66% |
| checkpoint | 0.982 / 0.396 / 0.351 | 100% / 21% / 16% |
| forward | 0.694 / 0.393 / 0.202 | 11% / 4% / 2% |
| do_nothing | 0.150 / 0.086 / 0.123 | 0% |
| random | 0.000 / 0.000 / 0.002 | 11% / 2% / 3% |

- **Conclusions:** the heuristic beats the checkpoint whenever trains interact; the checkpoint navigates
  alone but fails at coordination (~20% home). No policy serves any intermediate stop (70 episodes have
  them). Every MOVING→STOPPED costs 250 in flatland 4.2.6, even a chosen STOP, so skipping a stop (50)
  is cheaper than serving it (README, "What we know so far").

## Sanity task: one train, scene 1, no stops, no breakdowns

All `make_env(scene="scene_1", n_agents=1, line_length=2)`, training seed 0, 30 iterations × 16 games,
8 workers. Greedy checks played the same 8 games (seeds 1,000,000–1,000,007).

| ID | Date | Config | Change from E1 | Greedy result after 30 iterations | Training log (sampled) |
|---|---|---|---|---|---|
| E1 | 18–19 Oct | `configs/ippo_debug.yaml` | — (fresh weights) | **never departs: 8/8 cancelled**, score 0.000 | score 0.53, halts 51 → 0 |
| E1b | 18 Oct | `configs/ippo_debug_long_horizon.yaml` | γ 0.999, λ 0.99 | not measured (run before the greedy check existed) | score 0.000, entropy collapsed to 0.03 |
| E2a | 19 Oct | `configs/ippo_debug_warm.yaml` | start from organisers' checkpoint | **100% arrived, score 1.000** at 0, 10, 20, 30 | score 1.000, entropy ~0.6 |
| E2b | 19 Oct | `configs/ippo_debug_progress.yaml` | progress shaping 0.01 | never departs (100%) | score 0.58 |
| E2c | 19 Oct | `configs/ippo_debug_progress01.yaml` | progress shaping 0.1 | departs at it 10 (12% never depart), regresses (50%, 88%); no arrivals | score 0.32 |

Rerun any of them with `python scripts/train.py --config configs/<config> --seed 0`.

**Diagnosis D1 (19 Oct):** the E1 actor on 10 unseen one-train games (seeds 5000–5009):

| | E1 actor, greedy | E1 actor, sampled | Checkpoint |
|---|---|---|---|
| Outcome | never departed 10/10 | on the map, never arrived 10/10 | arrived 10/10 |
| Actions | do-nothing 100% | do-nothing 95%, forward 3% | do-nothing 62–88%, forward 9–34% |
| Junctions: chose the closer branch | — | 74% | 100% |

(Observation features 0–3 are relative to the train: left, forward, right, back.)

**Conclusions:**
- The trainer is sound: warm start stays perfect (PPO does not damage a good policy), and shaping
  changes behaviour when strong enough (E2c).
- From scratch, the network learns to stay at the start. Likely cause: while exploring it presses STOP
  often and every STOP while moving costs 250 immediately, whereas waiting at the start costs nothing
  until the episode ends.
- Training-log scores were misleading in E1 (0.53 while the greedy policy never moves), which is why the
  greedy check was added.
- Next levers for from-scratch learning (not pursued yet): lower the initial STOP probability, or offset
  the exploration STOP cost in training. **Decided 25 Oct:** from-scratch training is dropped; both IPPO
  and MAPPO warm-start from the organisers' checkpoint (see "Team decisions" below).

## E3: warm-started IPPO on interacting trains (started 21 Oct)

- **Question:** starting from the organisers' checkpoint, does IPPO fine-tuning get more trains home
  once they interact (5–10 trains)?
- **Config:** `configs/ippo_warm_scene1_a5-10_ll2.yaml` (lr 1e-4, 100 iterations × 32 games, 10 workers,
  greedy check on 16 games every 10 iterations), training seed 0.
- **Training pool:** `pools/scene1_a5-10_ll2` (2,000 scenarios, seeds 0–1999, no disruptions).
- **Validation pool:** `pools/val_scene1_a5-10_ll2` (50 scenarios, seeds 1,000,000–1,000,049).
- **Rerun:**
  ```bash
  python scripts/make_scenario_pool.py --out pools/scene1_a5-10_ll2 --n 2000 --scene scene_1 --n-agents 5 10 --line-length 2 --seed 0 --workers 10
  python scripts/make_scenario_pool.py --out pools/val_scene1_a5-10_ll2 --n 50 --scene scene_1 --n-agents 5 10 --line-length 2 --seed 1000000 --workers 10
  python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2.yaml --seed 0
  python scripts/evaluate.py --scenarios pools/val_scene1_a5-10_ll2 --disruption none \
      --policies checkpoint heuristic actor:runs/ippo_warm_scene1_a5-10_ll2_s0/actor.pt --out results/e3_val --workers 10
  ```
- **Validation baselines** (`results/e3_val_baselines/`, 50 scenarios, no disruptions):

  | Policy | Score | Arrived | Halts per game |
  |---|---|---|---|
  | heuristic | 0.637 | 86% | 6.5 |
  | checkpoint (IPPO's starting point) | 0.519 | 45% | 4.5 |
  | forward | 0.618 | 8% | 1.1 |

  "Forward" outscoring the checkpoint while getting 8% vs 45% home is the score's leniency on
  non-arrival again: judge E3 by arrivals first.
- **Training greedy check** (16 training-pool games, `runs/ippo_warm_scene1_a5-10_ll2_s0/eval_log.csv`):

  | After iteration | 0 | 10 | 20 | 30 | 40 | 60 | 80 | 100 |
  |---|---|---|---|---|---|---|---|---|
  | Score | 0.432 | 0.562 | 0.620 | 0.813 | 0.838 | 0.820 | 0.826 | 0.825 |
  | Arrived | 38% | 50% | 55% | 42% | 53% | 60% | 47% | 41% |
  | Halts per game | 4.6 | 3.6 | 3.2 | 0.7 | 0.3 | 0.9 | 0.6 | 0.4 |

- **Validation** (`results/e3_val_ippo/`, same 50 scenarios as the baselines; differences are paired per
  scenario, 95% bootstrap CI over scenarios):

  | Policy | Score | Arrived | Halts | Score vs checkpoint | Arrived vs checkpoint | Arrived vs heuristic |
  |---|---|---|---|---|---|---|
  | heuristic | 0.637 | 86% | 6.5 | | | |
  | checkpoint | 0.519 | 45% | 4.5 | | | |
  | IPPO it 30 | 0.827 | 46% | 0.7 | +0.31 [+0.25, +0.37] | +1% [−5, +6] | −40% [−47, −33] |
  | IPPO it 60 | 0.811 | 57% | 1.1 | +0.29 [+0.22, +0.36] | **+12% [+4, +19]** | −29% [−35, −23] |
  | IPPO it 100 | 0.794 | 40% | 0.8 | +0.28 [+0.21, +0.34] | −6% [−12, 0] | −46% [−52, −40] |

  Every IPPO snapshot also beats the heuristic on score (+0.16 to +0.19, CIs above 0).
- **Conclusions:**
  - Warm-started IPPO raises the official score a lot (0.52 → ~0.81, above the heuristic's 0.64), but
    **almost entirely by halting less** (4.5 → ~1 per game, 250 each), not by getting more trains home.
    Only the iteration-60 snapshot gets clearly more trains home than its starting point (+12%), and all
    snapshots stay far below the heuristic (86%).
  - This is the 250-per-stop rule again, now at the heart of the research question: **giving way means
    stopping, and stopping costs as much as a collision**. The heuristic gives way (6.5 halts) and gets
    86% home with a lower score; IPPO learns not to stop. The official score therefore rewards *less*
    coordination, which works against measuring whether MAPPO coordinates better.
  - Caveats: one training seed; iteration 60 was picked using these validation scenarios, so its numbers
    are optimistic (needs the frozen test set); CIs cover scenario variation only, not training seeds.
- **Decision this raised:** for the IPPO vs MAPPO comparison, train on the official reward (and accept
  that "don't stop" wins), or train with chosen STOPs not penalised while still *reporting* official
  scores and arrivals. **Decided 25 Oct: the second** (see "Team decisions" below). E3 stays as the
  official-reward comparison.

### E3 repeated with training seeds 1 and 2 (21 Oct)

- **Question:** is seed 0's result (validation arrivals 45% → 57% at iteration 60) reproducible across
  training seeds?
- **Rerun:** `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2.yaml --seed 1` (and `--seed 2`).
- **Code:** seeds 1–2 ran on commit `6540847`; seed 0 ran on `25f05a5` plus uncommitted changes that are
  byte-identical to `6540847` (all code files were last modified on 19 Oct, before seed 0 ran).
- **Outputs (committed):** `results/e3_training/` — per seed: snapshots `actor_10.pt` … `actor_100.pt`,
  `eval_log.csv`, `train_log.csv`, `config.yaml`, `run.json`; plus `greedy_checks_all_seeds.csv`.

**Run health:** all three ran cleanly.

| Seed | Iterations | Errors | Unattributed penalty | Lowest entropy | Largest `approx_kl` | Time |
|---|---|---|---|---|---|---|
| 0 | 100 | none | 0 | 0.57 | 0.010 | 15.4 min |
| 1 | 100 | none | 0 | 0.56 | 0.007 | 17.3 min |
| 2 | 100 | none | 0 | 0.55 | 0.009 | 14.9 min |

**Greedy check, share of trains arrived** (16 fixed training-pool games per check; noisy):

| After iteration | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Seed 0 | 38% | 50% | 55% | 42% | 53% | 49% | 60% | 47% | 47% | 49% | 41% |
| Seed 1 | 38% | 59% | 42% | 39% | 38% | 40% | 51% | 43% | 59% | 68% | 49% |
| Seed 2 | 38% | 49% | 79% | 54% | 66% | 33% | 5% | 39% | 53% | 40% | 23% |
| Seed 2, never departed | 0% | 0% | 0% | 0% | 0% | 8% | 37% | 32% | 0% | 0% | 14% |

Halts per game fell from 4.6 to under 1 in every seed and stayed there.

- **Conclusions:**
  - The training mechanics are reproducible and healthy (no errors, no lost penalties, no entropy collapse).
  - **Arrivals are unstable across seeds and across iterations.** Each seed peaks at a different iteration
    (seed 0: 60% at it 60; seed 1: 68% at it 90; seed 2: 79% at it 20) and falls back; final snapshots range
    from 23% to 49% arrived on these games.
  - Seed 2 temporarily relapses into "stay at the start" (37% of trains never depart at it 60): the
    failure mode of E1 can return mid-training even from a warm start.
  - "Stop halting" is learned reliably by every seed; arriving is not.
  - Seed 0's +12% validation result should not be treated as established until all seeds are evaluated
    on the validation pool.
- **Validation of all seeds (21 Oct):** all 30 snapshots on the 50 validation scenarios, no disruptions
  (`results/e3_val_seeds/`: `episodes.csv`, `trains.csv`, `run.json`, and `summary_by_snapshot.csv` = one row
  per seed × iteration). Seed 0's snapshots reproduce the earlier `e3_val_ippo` results scenario for scenario.
  Rule fixed before looking: main number = each seed's iteration-100 snapshot; secondary = each seed's best
  snapshot by validation arrivals. Differences are paired per scenario, 95% bootstrap CI over scenarios.

  **Validation arrivals by iteration:**

  | After iteration | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Seed 0 | 49% | 64% | 46% | 55% | 50% | 57% | 45% | 46% | 47% | 40% |
  | Seed 1 | 55% | 49% | 43% | 40% | 41% | 55% | 49% | 62% | 65% | 53% |
  | Seed 2 | 46% | 75% | 52% | 63% | 36% | 9% | 28% | 52% | 42% | 21% |

  (Checkpoint 45%, heuristic 86% on the same scenarios. Seed 2 never-departed: 12%, 34%, 31% at iterations
  50–70 and 20% at 100.)

  **Main result, final snapshots (unbiased):**

  | | Score | Arrived | Halts | vs checkpoint: score | vs checkpoint: arrived | vs heuristic: arrived |
  |---|---|---|---|---|---|---|
  | Seed 0, it 100 | 0.794 | 40% | 0.8 | +0.28 [+0.21, +0.33] | −6% [−12, 0] | |
  | Seed 1, it 100 | 0.844 | 53% | 0.7 | +0.33 [+0.26, +0.39] | +8% [+1, +15] | |
  | Seed 2, it 100 | 0.702 | 21% | 0.6 | +0.18 [+0.11, +0.26] | −24% [−32, −15] | |
  | **Mean of seeds** | **0.780** (sd 0.072) | **38%** (sd 16%) | | **+0.26 [+0.20, +0.32]** | **−7% [−14, −1]** | −48% [−53, −42] |

  **Secondary, best snapshot per seed by validation arrivals (optimistic):** iterations 20, 90, 20.
  Mean score 0.796 (sd 0.084), arrived **68%** (sd 6%): vs checkpoint arrived +22% [+16, +28], vs heuristic
  −18% [−23, −12]. Chosen from 10 snapshots on these same scenarios, so this overstates what a fresh test
  set would show.
- **Conclusions (all seeds):**
  - **After full training, IPPO gets *fewer* trains home than the checkpoint it started from**
    (38% vs 45%, −7% [−14, −1]), while the official score rises by +0.26: the score gain is fewer halts.
  - **Arrivals peak early and then erode:** two of three seeds are best at iteration 20, and the longer
    training continues on the official reward, the further it moves towards "don't stop", at the cost of
    getting trains home. Consistent with the 250-per-stop incentive.
  - Seed 0's earlier "+12% at iteration 60" was one seed at one snapshot; it does not hold up as a general result.
  - Seed-to-seed spread is large (arrivals sd 16% at iteration 100): conclusions need all three seeds.
  - The best early snapshots (68% arrived) are promising but selected on validation; only a frozen test set
    can confirm them.
- **Possible next steps:** smaller learning rate or fewer PPO epochs per iteration (stability); snapshot
  selection by validation; train with chosen STOPs unpenalised (decided 25 Oct; the swings between "go"
  and "stay put" are what the 250-per-stop penalty encourages).

## Team decisions (25 Oct)

| Decision | What it means for every later run |
|---|---|
| **Warm start for both arms** | IPPO and MAPPO both start from the organisers' `checkpoint.pt`. From-scratch training (E1–E2) is not part of the comparison. |
| **Stop rule: train without the chosen-STOP penalty, report official** | The training reward does not charge the 250 for a halt that follows the train's own STOP action; halts forced by being blocked still cost 250. Every reported number uses the official reward, with arrivals first and the score next to it. Same training reward for IPPO and MAPPO. E3 (trained on the official reward) is the comparison that shows what the rule does to learning. |

Implemented 25 Oct as two config switches (both default off, so earlier configs rerun unchanged), used
together from E4 on and identical for IPPO and MAPPO:

| Switch | What it does |
|---|---|
| `reward.exempt_chosen_stops: true` | the stop rule above; `train_log.csv` column `exempted_halts` = chosen STOPs not charged, per game |
| `ppo.value_norm: true` | the critic learns returns standardised by their running mean/std (Yu et al., 2022); `train_log.csv` columns `return_mean`, `return_std` |
| `reward.chosen_stop_cost: 25` (26 Oct, from E5) | with the exemption on, a chosen STOP costs 25 instead of 250 (below 50, so serving a station still beats skipping it) |
| `reward.waiting_cost: 0.5` (26 Oct, from E5) | per step, for every train that may travel (past its earliest departure) but has not arrived |

## E4: E3 with chosen STOPs unpenalised and value normalisation (25 Oct): failed

- **Question:** with giving way no longer charged in training, do arrivals stop eroding, and do they beat
  E3 and the checkpoint?
- **Config:** `configs/ippo_warm_scene1_a5-10_ll2_nostop.yaml` = E3's config plus `reward.exempt_chosen_stops`
  and `ppo.value_norm`; same pool, budget and hyperparameters. Training seeds 0, 1, 2, commit `bafc37c`.
- **Rerun:** `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_nostop.yaml --seed 0` (and 1, 2)
- **Outputs (committed):** `results/e4_training/` (same layout as `results/e3_training/`).

**Run health:** mechanically clean, like E3.

| Seed | Iterations | Errors | Unattributed penalty | Lowest entropy | Largest `approx_kl` | Time (wall) |
|---|---|---|---|---|---|---|
| 0 | 100 | none | 0 | 0.57 | 0.008 | 30.6 min |
| 1 | 100 | none | 0 | 0.57 | 0.009 | 32.5 min |
| 2 | 100 | none | 0 | 0.54 | 0.015 | 28.9 min |

(About twice E3's wall time: as the policy degrades, games run to the time limit.)

**Greedy check, share of trains arrived** (16 fixed training-pool games per check; E3 for comparison):

| After iteration | 0 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| E4 seed 0 | 38% | 53% | **67%** | 61% | 41% | 32% | 20% | 2% | 3% | 0% | 0% |
| E4 seed 1 | 38% | 55% | **65%** | 63% | 61% | 41% | 35% | 27% | 27% | 27% | 26% |
| E4 seed 2 | 38% | 51% | 50% | **76%** | 58% | 41% | 26% | 9% | 14% | 2% | 3% |
| E4 seed 0, never departed | 0% | 0% | 0% | 3% | 28% | 59% | 67% | 57% | 76% | 100% | 82% |
| E4 seed 2, never departed | 0% | 0% | 0% | 0% | 3% | 21% | 56% | 53% | 27% | 31% | 45% |
| E3 seeds 0 / 1 / 2 at it 100 | | | | | | | | | | | 41% / 49% / 23% |

(Seed 1 never-departed stayed at 0–4%: its trains departed but stood still on the map.)

**Halts per training game (sampled actions):**

| After iteration | 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Seed 0: halts / of which chosen STOPs (exempted) | 5 / 0 | 5 / 1 | 63 / 59 | 45 / 43 | 74 / 73 | 145 / 145 | 196 / 195 | 231 / 231 | 205 / 205 | 215 / 215 | 244 / 243 |
| Seed 1 | 5 / 0 | 3 / 0 | 71 / 64 | 50 / 48 | 55 / 54 | 110 / 109 | 152 / 152 | 200 / 200 | 245 / 244 | 282 / 282 | 268 / 267 |
| Seed 2 | 5 / 0 | 5 / 0 | 7 / 5 | 60 / 56 | 45 / 45 | 61 / 61 | 143 / 142 | 196 / 194 | 291 / 290 | 288 / 288 | 332 / 332 |

(E3: halts fell from 4.6 to under 1 per game.)

- **Conclusions:**
  - **Removing the chosen-STOP charge made stopping free, and every seed learned to stop and wait.** Training
    games reach 240–330 halts per game, almost all chosen; by iteration 100 greedy arrivals are 0–26%, below
    E3 (23–49%) and the checkpoint (38% on these games). Seeds 0 and 2 slide into trains never departing (the
    E1 failure mode); seed 1's trains depart and then stand still.
  - **Cause:** once the 250 is gone, nothing charges a train *immediately* for standing still. Lateness and
    not arriving are charged as lumps at arrival or at the episode's end, often hundreds of steps later, and
    with γ = 0.99 a penalty 300 steps away weighs about 5%. So stopping looks nearly free to the critic.
    Taking away an unfair penalty left no immediate cost for wasting time.
  - **Before the slide, E4 peaks about as high as E3:** 65–76% arrived at iterations 20–30, vs E3's peaks of
    60%, 68% and 79% (iterations 60, 90, 20). All three E4 seeds peak in the first 30 iterations; the
    collapse comes after.
  - Both switches changed at once. Exempted halts make up nearly all halts, which points at the exemption,
    but value normalisation is not strictly ruled out.
  - No validation evaluation: the failure is clear from the greedy checks.
- **Next (decided 26 Oct): E5 = option 3**: keep the exemption but charge a small cost for a chosen STOP, plus
  a small cost for every step a train has not yet arrived, so waiting costs something immediately.

## E5: E4 plus a small chosen-STOP cost and a waiting cost (25–26 Oct)

- **Question:** does giving way stay cheap without trains learning to stop and wait (E4)?
- **Config:** `configs/ippo_warm_scene1_a5-10_ll2_stopcost.yaml` = E4's config plus `reward.chosen_stop_cost: 25`
  and `reward.waiting_cost: 0.5` (option 3, decided 26 Oct). Same pool, budget and hyperparameters. Training
  seeds 0, 1, 2, commit `1d221b1`.
- **Rerun:**
  ```bash
  python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_stopcost.yaml --seed 0   # and 1, 2
  python scripts/evaluate.py --scenarios pools/val_scene1_a5-10_ll2 --disruption none \
      --policies actor:results/e5_training/seed_0/actor_10.pt ... actor:results/e5_training/seed_2/actor_100.pt \
      --out results/e5_val_seeds --workers 10        # all 30 snapshots; full list in results/e5_val_seeds/run.json
  ```
- **Outputs (committed):** `results/e5_training/` (networks + logs, same layout as E3/E4) and `results/e5_val_seeds/`
  (`episodes.csv`, `trains.csv`, `run.json`, `summary_by_snapshot.csv`).

**Run health:** clean, like E3 and E4.

| Seed | Iterations | Errors | Unattributed penalty | Entropy (min–max) | Largest `approx_kl` | Time (wall) |
|---|---|---|---|---|---|---|
| 0 | 100 | none | 0 | 0.57–0.67 | 0.008 | 29.9 min |
| 1 | 100 | none | 0 | 0.56–0.68 | 0.008 | 30.6 min |
| 2 | 100 | none | 0 | 0.54–0.68 | 0.008 | 27.9 min |

**The E4 failure is gone.** Halts per training game fall from about 5 (first 10 iterations) to 1.3–1.7 (last 10),
of which about 1 is a chosen STOP; E4 climbed to 240–330. Arrivals in the training games (sampled actions,
mean of the last 10 iterations, ~320 games) rise in every seed: **57% / 62% / 60%**, from ~45%; E3 ended at
39% / 46% / 37%.

**Validation** (50 scenarios, no disruptions; same rule as E3, fixed before looking: main number = each seed's
iteration-100 snapshot; secondary = each seed's best snapshot by validation arrivals; differences paired per
scenario, 95% bootstrap CI over scenarios). Arrivals by iteration:

| After iteration | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| Seed 0 | 52% | 58% | 57% | 61% | 66% | 63% | 65% | 38% | 64% | 12% |
| Seed 1 | 58% | 62% | 56% | 47% | 69% | 53% | 61% | 35% | 75% | 76% |
| Seed 2 | 52% | 59% | 56% | 49% | 72% | 47% | 62% | 72% | 72% | 40% |
| **E5 mean of seeds** | 54% | 59% | 56% | 53% | 69% | 55% | 63% | 48% | 70% | 43% |
| E3 mean of seeds | 50% | 62% | 47% | 52% | 42% | 40% | 40% | 53% | 51% | 38% |

(Checkpoint 45%, heuristic 86%. Mean over all 30 snapshots: E5 57%, E3 48%. Never departed: seed 0 14% at
it 80 and **71% at it 100**; seed 1 29% at it 80; otherwise 0–5%.)

**Main result, final snapshots (unbiased):**

| | Score | Arrived | Halts | vs checkpoint: score | vs checkpoint: arrived |
|---|---|---|---|---|---|
| Seed 0, it 100 | 0.307 | 12% | 0.0 | −0.21 [−0.27, −0.16] | −34% [−42, −26] |
| Seed 1, it 100 | 0.843 | **76%** | 1.7 | +0.32 [+0.26, +0.39] | **+31% [+23, +38]** |
| Seed 2, it 100 | 0.699 | 40% | 1.0 | +0.18 [+0.13, +0.23] | −5% [−11, 0] |
| **Mean of seeds** | **0.616** (sd 0.277) | **43%** (sd 32%) | 0.9 | +0.10 [+0.05, +0.15] | −3% [−9, +3] |

Mean of seeds vs heuristic: score −0.02 [−0.07, +0.03], arrived −43% [−48, −38]. vs E3 (iteration 100, mean of
seeds): score −0.16 [−0.19, −0.14], arrived **+5% [+2, +7]**.

**Secondary, best snapshot per seed by validation arrivals (optimistic):** iterations 50, 100, 80. Mean score
0.835 (sd 0.014), arrived **72%** (sd 5%): vs checkpoint +26% [+20, +33], vs heuristic −14% [−20, −9], vs E3's
final snapshots +33% [+30, +37]. (E3's best snapshots: 68%.) Chosen from 10 snapshots on these same scenarios.

- **Conclusions:**
  - **The costs fix E4.** With a chosen STOP at 25 and waiting at 0.5 per step, trains no longer learn to stop
    and wait, and arrivals no longer erode with training: on validation E5 is above E3 at 8 of 10 snapshots
    (mean over all snapshots 57% vs 48%), and in the training games every seed ends higher than it started.
  - **Training is now unstable rather than misdirected.** Seed 0's final snapshot relapsed into trains never
    departing (71%) after sitting at 61–66% for most of training; seeds 0 and 1 also dipped at iteration 80
    (14% / 29% never departed) and recovered. A single stretch of updates can flip a good policy into "stay
    put" (also seen in E3 seed 2 and all of E4). Because of seed 0, the unbiased final mean (43%) is only
    slightly above E3 (+5%) and level with the checkpoint, with a very large spread across seeds (sd 32%).
  - **Best result so far:** seed 1's final snapshot, 76% arrived (+31% vs checkpoint), still below the
    heuristic (86%). The best-by-validation snapshots average 72%, but that number is optimistic.
  - **Giving way is still rare:** about 1 chosen STOP per game against the heuristic's 6.5 halts. The gain
    comes from not drifting into "don't move", not yet from coordination.
  - The official score is lower than E3's (0.62 vs 0.78) because E3 scored by never stopping; judge by arrivals.
  - Caveats: 16-game greedy checks during training are too noisy to pick snapshots; CIs cover scenario
    variation only, not training seeds.
- **Next:** stabilise training before MAPPO (team discussion 26 Oct).
