# flatland-marl

CS4246/5446 Group 20: IPPO vs MAPPO for multi-agent train rescheduling on the
[Flatland ECML 2026 challenge](https://flatland-association.github.io/flatland-book/challenges/ecml2026.html).

## Setup

Requires **Python 3.12** and git. Everything runs on CPU.

```bash
# 1. Clone the repo
git clone https://github.com/rndllchw/flatland-marl.git
cd flatland-marl

# 2. Create the venv and install pinned dependencies (also installs our `flatmarl` package)
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Clone the starter kit and heuristic baseline (pinned commits) into third_party/
bash scripts/setup_third_party.sh
```

After pulling new changes, re-run `pip install -r requirements.txt` if `requirements.txt` changed.

Check it worked by running the smoke test. It runs the starter-kit PPO checkpoint and the
deadlock-avoidance heuristic on the sample scenario and compares their scores with the reference
values:

```bash
python scripts/smoke_test.py
```

Expected output (speed will vary by machine):

```
ppo_checkpoint       score 0.730 (expected 0.730)  arrived 6/6 (expected 6)  stops served 0%  halts while moving 0  ... OK
deadlock_heuristic   score 0.513 (expected 0.513)  arrived 6/6 (expected 6)  stops served 0%  halts while moving 4  ... OK
smoke test PASSED
```

Run the unit tests (~30 s) before pushing changes to `src/`:

```bash
pytest tests/
```

## Creating environments

**Always build environments through `flatmarl.envs`**, never with flatland's `RailEnvPersister`
directly. Scenario files silently restore their own breakdown settings and random state;
`flatmarl.envs` handles this, and `tests/test_envs.py` checks it.

```python
from flatmarl.envs import make_env, load_scenario

# Random scenarios on the competition map (training). Every reset samples new routes/timetables.
env = make_env(scene="scene_1", n_agents=10, line_length=3, disruption="breakdowns", seed=0)

# A saved scenario (evaluation). Every reset replays it exactly, breakdowns included.
env = load_scenario("path/to/scenario.pkl", disruption="none", seed=0)

obs, info = env.reset()                          # obs: {train_id: 41 numbers (36 features + 5 action mask)}
while True:
    actions = {t: 2 for t in obs}                # your policy here (0 nothing, 1 left, 2 forward, 3 right, 4 stop)
    obs, rewards, dones, info = env.step(actions)
    if dones["__all__"]:
        break
```

| Argument | Meaning |
|---|---|
| `scene` | `"scene_1"`–`"scene_4"` are map regions; `"scene_5"` is the whole map |
| `n_agents` | number of trains: an int, or `(lo, hi)` re-drawn at every reset |
| `line_length` | max stations per route incl. start and end (2 = no intermediate stops) |
| `disruption` | `"none"`, `"breakdowns"` (rate 1/540, 20–50 steps), `"breakdowns+delays"` (+ departure delays at stops) |
| `seed` | makes everything reproducible. Use a different seed per training worker |

**Parallel workers: send a recipe, not an env.** Envs can't be pickled (the starter kit's route
generator and flatland's reward memory contain inner functions), so a worker process can't be handed
a finished env. Send it a recipe and let it build its own:

```python
import functools
recipe = functools.partial(make_env, scene="scene_1", n_agents=10, seed=worker_id)  # picklable
env = recipe()                                                                      # inside the worker
```

This is also what SB3's `SubprocVecEnv` and RLlib's env creators expect. `tests/test_envs.py` checks it
works in a separate process.

Rewards are always the official `ECML2026Rewards` (a recording version that returns identical
rewards, see below), and every reset gets a fresh one so penalties never leak between episodes.

## Measuring an episode

After an episode finishes on any env from `flatmarl.envs`, one call gives every metric. Nothing
needs to be recorded by hand during the episode.

```python
from flatmarl.metrics import episode_metrics

m = episode_metrics(env)     # after dones["__all__"] is True
m.score                      # official normalised score
m.completion_rate, m.mean_lateness, m.stop_service_rate, m.halts_while_moving, m.gini_penalty
m.trains[3].penalty_by_type  # where one train's penalty came from
m.as_row()                   # everything as one flat dict (a CSV row)
```

| Episode metric | Meaning |
|---|---|
| `score` | official normalised score (0–1) |
| `completion_rate` / `cancelled_rate` | share of trains that arrived / never departed |
| `on_time_rate`, `mean_lateness` | among arrived trains (lateness = steps late, early counts as 0) |
| `stop_service_rate` | intermediate stops served / required |
| `halts_while_moving` | times a moving train came to a halt, **whether it chose STOP or was blocked** (250 each, see below) |
| `malfunctions`, `malfunctions_before_departure` | trains stuck on the map (breakdowns **and** departure delays: flatland uses one state for both) / breakdowns before departure |
| `p90_lateness`, `max_lateness`, `gini_penalty` | fairness: is the damage concentrated on a few trains? (Gini 0 = equal) |
| `penalty_by_type` | total penalty split into flatland's 8 types (below) |

| Penalty type | Meaning |
|---|---|
| `TARGET_LATE_ARRIVAL` | arrived late at destination: 1 per step |
| `INTERMEDIATE_NOT_SERVED` | passed an intermediate stop without stopping: 50 each |
| `INTERMEDIATE_LATE_ARRIVAL` / `INTERMEDIATE_EARLY_DEPARTURE` | late at / left too early from a stop: 0.5 per step |
| `TARGET_NOT_REACHED` | started but never arrived: at least 100 |
| `CANCELLATION` | never departed: 5 × shortest travel time |
| `COLLISION` | a moving train came to a halt, by choosing STOP **or** by being blocked: 250 |
| `TARGET_NOT_REACHED_MINIMUM_PENALTY` | declared by flatland 4.2.6 but never assigned: always 0 |

## Evaluating policies

`scripts/evaluate.py` runs policies over a folder of saved scenarios and writes one CSV row per
episode. Default scenarios: the starter kit's 27 examples (unzipped by `setup_third_party.sh`).

```bash
# Baseline table (5 policies x 27 scenarios, each scenario's own disruptions), 8 in parallel
python scripts/evaluate.py --policies do_nothing random forward checkpoint heuristic \
    --out results/baselines_example_curriculum --workers 8

# An experiment condition on a frozen test set, 3 breakdown patterns, a trained actor
python scripts/evaluate.py --scenarios eval_sets/test --disruption breakdowns --seeds 0 1 2 \
    --policies heuristic actor:runs/ippo_s0/actor.pt --out results/test_breakdowns
```

| Option | Meaning |
|---|---|
| `--policies` | `do_nothing`, `forward`, `random`, `heuristic`, `checkpoint`, `actor:<path>` (any network with the starter-kit `ActorCritic` shape, saved as `{"model": state_dict}`; plays its top action), or `actor-sampled:<path>` (same network, each action drawn from its policy as in training; **the main number for trained actors since 26 Oct**, use with `--seeds 0 1 2`) |
| `--scenarios` | folder of `.pkl` scenarios, or one `.pkl` |
| `--disruption` | `file` (each scenario's own, default), `none`, `breakdowns`, `breakdowns+delays` |
| `--seeds` | one episode per seed (breakdown pattern + random / `actor-sampled` draws); omitted = the file's stored randomness. With `--disruption none` they change only the draws |
| `--workers` | episodes in parallel; results are identical to `--workers 1` |
| `--overwrite` | needed to replace existing results in `--out` |

Output in `--out`: `episodes.csv` (one row per episode), `trains.csv` (one row per train per
episode), `run.json` (command, git commit, versions, scenario list). Policies are defined in
`src/flatmarl/policies.py`; `make_policy(name)` gives a fresh (policy, observation builder) pair.

## Scenario pools (fast training resets)

Inventing routes with intermediate stops takes 7–21 s per reset. Generate a pool once, then train on
replays (measured: 8.25 s → 0.36 s per reset, 23× faster, scene 1, 5–10 trains, `line_length=3`).

```bash
# ~8 s per scenario per worker; re-running resumes (existing files are skipped)
python scripts/make_scenario_pool.py --out pools/scene1_a5-10_ll3 --n 2000 \
    --scene scene_1 --n-agents 5 10 --line-length 3 --seed 0 --workers 10
```

```python
from flatmarl.envs import make_pool_env
env = make_pool_env("pools/scene1_a5-10_ll3", disruption="breakdowns", seed=worker_id)
obs, info = env.reset()      # a random scenario from the pool, with a fresh seeded breakdown pattern
```

Pool files are saved with no disruptions; the breakdown setting is chosen when the pool is used.
`pools/` is gitignored (about 50 KB per scenario). The same script builds evaluation sets. **Keep seed
ranges disjoint:** training pools from seed 0, validation pools from 1,000,000, frozen test sets from
2,000,000 (see `EXPERIMENTS.md`).

## Training (IPPO)

```bash
python scripts/train.py --config configs/ippo_debug.yaml --seed 0
python scripts/evaluate.py --policies actor:runs/ippo_debug_s0/actor.pt checkpoint --out results/ippo_debug
```

To compare a run's snapshots on validation with the rule used since E3 (final snapshot = main number, best
snapshot = optimistic secondary, paired bootstrap CIs over scenarios), evaluate them all and run
`scripts/val_analysis.py <evaluate output folder> --ref ... --write-summary` (examples in `EXPERIMENTS.md`).

A config sets the env (`source: make_env` or `source: pool`), `critic` (`local` = IPPO; `central` =
MAPPO, see below), reward scale, PPO settings and the training budget; unknown keys are
rejected. Each run writes `runs/<name>_s<seed>/`: `actor.pt` (plus snapshots), `train_log.csv` (one row
per iteration: official score, arrivals, stops served, halts, PPO stats), `eval_log.csv`, `config.yaml`,
`run.json`. Same config + seed gives identical weights, whatever the number of workers.

**Judge progress by `eval_log.csv`, not `train_log.csv`** (and judge results by validation with
`evaluate.py` + `scripts/val_analysis.py`). The quick check plays the same games each time
(`train.eval_episodes`, default 8), at iteration 0 and at every save: `train.eval_mode: greedy` (default)
plays the top action; `sampled` draws actions from the policy, like the main evaluation since 26 Oct (use it
for new configs; greedy checks misread E3–E5, see `EXPERIMENTS.md`). `train.init_checkpoint` and `env.pool`
paths are relative to the repo root.

**Training budget** (decided 26 Oct): `train.iterations` × `train.episodes_per_iteration` training games, the
same for IPPO and MAPPO; final runs use 200 × 32. `train_log.csv` records `decisions` and `agent_steps`
(env steps × trains) per iteration, with running totals, so the report can state both.

| File | Role | Extension point |
|---|---|---|
| `src/flatmarl/models.py` | actor (the starter kit's `ActorCritic`), masked action sampling; `CentralCritic` + `policy_and_values` for MAPPO | — |
| `src/flatmarl/obs.py` | `global_state`: the 40-number summary of all trains the central critic sees (training only) | — |
| `src/flatmarl/rollout.py` | play one episode; records only real decisions; credits every reward to the train's latest decision | — |
| `src/flatmarl/ppo.py` | GAE advantages, clipped PPO update (actor + central critic for MAPPO); optional value-target normalisation (`ppo.value_norm`, running mean/std of returns) | — |
| `src/flatmarl/reward.py` | official reward × scale; optional `progress` shaping (potential-based, towards the destination); optional `exempt_chosen_stops`, `chosen_stop_cost`, `waiting_cost` | more shaping terms (Member 4) |
| `scripts/train.py`, `configs/` | config + seed → run folder | curriculum schedule (Member 4) |

**Status of learning (19 Oct), one-train sanity task, seed 0, 30 iterations, greedy check:**

| Config | Greedy result after 30 iterations |
|---|---|
| `ippo_debug` (fresh weights) | never departs (100%); the training log's 0.53 came from sampled actions |
| `ippo_debug_warm` (organisers' checkpoint) | **100% arrived, score 1.000 throughout**: PPO keeps a good policy good |
| `ippo_debug_progress` (fresh, progress 0.01) | never departs (100%) |
| fresh, progress 0.1 | departs at iteration 10 (12% never depart), then drifts back (88%); no arrivals |

Diagnosis: a fresh network picks STOP often while exploring, and every STOP while moving costs 250
immediately, while waiting at the start costs nothing until the episode ends. So it learns to stay off
the map. Warm start sidesteps this; learning from scratch needs more work. Every run, with how to rerun
it, is in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Official reward vs training reward

There are two rewards. The **official reward** (`ECML2026Rewards`: 1 per step late, 50 per missed stop,
250 per halt while moving, ...) is how results are graded: every reported number uses it, so results
stay comparable with the baselines and the organisers' scores. The **training reward** is what PPO
learns from; we may change it, as long as learning from it produces policies that do well on the
official reward.

```
env.step() ──▶ official reward per train
                 ├──▶ metrics.py  (RecordingRewards) ──▶ score, arrivals, penalty breakdown   ← what we REPORT
                 └──▶ reward.py   (RewardShaper)     ──▶ training reward                      ← what PPO LEARNS FROM
                                                        = (official, with a chosen STOP charged
                                                           reward.chosen_stop_cost instead of 250 if
                                                           reward.exempt_chosen_stops, minus
                                                           reward.waiting_cost per step not yet arrived)
                                                          × reward.scale (+ reward.progress shaping)
```

The training reward is computed only inside `rollout.py`; `evaluate.py` and `episode_metrics` never see it.

| Problem with learning from the official reward | Fix in the training reward | Changes which policy is best? |
|---|---|---|
| Numbers too big: −250 makes the critic's targets large and learning unstable | `reward.scale` (default 0.01): −250 becomes −2.5 | No: every reward shrinks equally |
| Feedback too late: most penalties arrive at arrival or episode end, hundreds of steps after the decisions that caused them | `reward.progress`: a small reward for each cell closer to the destination | No: it is potential-based shaping (Ng et al., 1999), the same signal given earlier |
| Rewards the wrong behaviour: a chosen STOP costs 250, so policies learn not to give way and to skip stations | `reward.exempt_chosen_stops` (decided 25 Oct): a halt in a step where the train chose STOP is not charged; halts forced by being blocked still cost 250. Logged per game as `exempted_halts` in `train_log.csv` | **Yes**: a stated design choice in the report |
| Stopping must not become free: with the 250 removed, trains learned to stop and wait (E4), because lateness is charged only at arrival or episode end | `reward.chosen_stop_cost` (a chosen STOP costs this instead of 250; keep it below 50) and `reward.waiting_cost` (per step for every train that may travel but has not arrived) | **Yes**: stated in the report with the stop rule |

**Rules:**

1. Report official numbers only (`episode_metrics`, `evaluate.py`), with arrivals next to the score. Never
   compare runs by training reward.
2. IPPO and MAPPO train on exactly the same training reward.
3. Every shaping term is a config switch that defaults to off, so each run's `config.yaml` records which
   training reward produced it.
4. A change that alters which policy is best is reported as a design choice, with results still measured
   on the official reward.

## For teammates: how to plug in

`envs.py`, `metrics.py` and `evaluate.py` are the shared foundation. The trainer (IPPO / MAPPO) sits
between them: it trains on games from `make_env` and hands back a network file that `evaluate.py`
runs with `actor:<path>`, with **no changes to `policies.py`**.

```
make_env(...) ──► trainer (rollout → PPO update → repeat) ──► actor.pt ──► evaluate.py --policies actor:actor.pt
episode_metrics(env) ──► logging during training (score, arrivals, stops served, halts while moving)
```

| Member | Builds | Uses |
|---|---|---|
| 2: IPPO | `models.py`, `rollout.py`, `ppo.py`, `train.py`; saves `actor.pt` | `make_env` (one recipe per worker), 41-number observations, `episode_metrics` |
| 3: MAPPO | `obs.py` (summary of all trains for the critic) + central critic; a `critic: local/central` flag in the same trainer | same as Member 2 |
| 4: curriculum + reward | `reward.py` (shaping on top of the official reward) + curriculum schedule | `make_env`'s `scene`, `n_agents=(lo, hi)`, `line_length`; `stop_service_rate` to see if shaping works |
| 5: evaluation | `make_eval_sets.py` (`make_env` → `reset()` → `RailEnvPersister.save`), final runs, analysis | `evaluate.py --scenarios eval_sets/... --disruption none/breakdowns --seeds ...`; `trains.csv` for fairness |

**Three rules. Breaking any of them fails silently, not with an error:**

1. **The actor is the starter kit's `ActorCritic`, saved as `{"model": state_dict}`.** Loading requires
   every layer name to match, so train that class rather than a lookalike. IPPO: its built-in value head
   is the local critic. MAPPO: the actor's value head stays in place (a file without it is rejected) and
   the central critic is a separate module saved as `critic_<i>.pt`, never inside `actor.pt`. Always pass `checkpoint_path` explicitly: with no argument,
   `ActorCritic()` looks for `./submission/checkpoint.pt` relative to wherever you run it, and silently
   starts from random weights, or from the organisers' weights if run inside the starter kit.

   ```python
   from reinforcement_learning.my_policy import ActorCritic
   from flatmarl.policies import STARTER_KIT_CHECKPOINT

   actor = ActorCritic(checkpoint_path=None)                     # fresh weights
   # actor = ActorCritic(checkpoint_path=str(STARTER_KIT_CHECKPOINT))   # or warm start from the organisers'
   ...train...
   torch.save({"model": actor.state_dict()}, "runs/ippo_s0/actor.pt")  # evaluate with actor:runs/ippo_s0/actor.pt
   ```

2. **Keep the observation unchanged** (`make_env`'s default: `FastTreeObsBuilder(max_depth=3,
   with_action_mask=True)`, 36 features + action mask at positions 36–40). `actor:<path>` always feeds that
   observation. If you change it, add your own entry to `src/flatmarl/policies.py` pairing your network with
   your observation builder.

3. **Parallel workers get a recipe, not an env, and each gets its own seed:**
   `functools.partial(make_env, ..., seed=worker_id)`. Same seed means identical games in every worker.

**Settled:** our own PPO (MAPPO = a different critic input, not a different library), and scenario
pools for training with intermediate stops (see above). Both can still be revisited as a team.

## Repo layout and status

```
scripts/setup_third_party.sh   ✅ clones starter kit @54edacd and flatland-baselines @1e9d4a6 into third_party/
scripts/smoke_test.py          ✅ quick end-to-end check against reference scores
src/flatmarl/envs.py           ✅ environment factory: make_env, load_scenario, disruption presets
tests/test_envs.py             ✅ 20 tests for envs.py
src/flatmarl/metrics.py        ✅ per-episode metrics (score, arrivals, lateness, missed stops, halts while moving, fairness)
tests/test_metrics.py          ✅ 21 tests for metrics.py
src/flatmarl/policies.py       ✅ named baseline policies + actor:<path> + actor-sampled:<path>
scripts/evaluate.py            ✅ run policies over saved scenarios -> episodes.csv, trains.csv, run.json
tests/test_evaluate.py         ✅ 11 tests for evaluate.py (incl. actor-sampled, memory)
scripts/val_analysis.py        ✅ snapshot comparison on validation (E3 rule, paired bootstrap); tests/test_val_analysis.py
results/baselines_example_curriculum/  ✅ first baseline table (5 policies x 27 scenarios)
eval_sets/test/                ✅ frozen test sets T1–T3 (200 scenarios; built with make_scenario_pool.py, see EXPERIMENTS.md E8)
src/flatmarl/envs.py           ✅ + ScenarioPoolEnv / make_pool_env (replay a random pooled scenario)
scripts/make_scenario_pool.py  ✅ pre-generate scenario pools (parallel, resumable)
tests/test_pool.py             ✅ 7 tests for pools
src/flatmarl/{models,rollout,ppo,reward}.py, scripts/train.py, configs/   ✅ IPPO framework (learning: see Training)
tests/test_training.py         ✅ 26 tests for the training framework
EXPERIMENTS.md                 ✅ experiment log: every run, how to rerun it, results, conclusions
results/e3_training/           ✅ E3 trained networks (every 10 iterations) and logs, seeds 0–2
results/e3_val_seeds/          ✅ E3 validation of all 30 snapshots (+ summary_by_snapshot.csv)
results/e4_training/           ✅ E4 networks and logs (failed run: free STOPs → stop and wait)
results/e5_training/           ✅ E5 networks and logs, seeds 0–2
results/e5_val_seeds/          ✅ E5 validation of all 30 snapshots, greedy (+ summary_by_snapshot.csv)
results/{e3,e5}_val_sampled/   ✅ E3 and E5 validation, sampled actions x 3 seeds (+ summary_by_snapshot.csv)
results/val_sampled_baselines/ ✅ checkpoint on validation, sampled actions x 3 seeds
results/e6_training/           ✅ E6 (IPPO arm) networks every 10 iterations to 200, logs, seeds 0–2
results/e6_val_sampled/        ✅ E6 validation of all 60 snapshots, sampled x 3 seeds (+ summary_by_snapshot.csv)
results/e7_training/           ✅ E7 (MAPPO arm) actors + critics every 10 iterations to 200, logs, seeds 0–2
results/e7_val_sampled/        ✅ E7 validation of all 60 snapshots, sampled x 3 seeds (+ summary_by_snapshot.csv)
results/e8_test/               ✅ E8: IPPO vs MAPPO vs baselines on the test sets (+ summary_arrivals.csv, summary_metrics.csv)
scripts/run_e8.sh, scripts/e8_analysis.py   ✅ E8 evaluation driver and analysis
src/flatmarl/obs.py + models.CentralCritic   ✅ MAPPO (critic: central); 5 tests in test_training.py
```

## What we know so far

**Baselines on the 27 example scenarios** (`results/baselines_example_curriculum/`, each scenario's own
breakdowns + departure delays):

| Policy | Score: 1 / 10 / 25 trains | Arrived: 1 / 10 / 25 trains |
|---|---|---|
| Deadlock-avoidance heuristic | 0.982 / **0.654** / **0.536** | 100% / **81%** / **66%** |
| Starter-kit PPO checkpoint | 0.982 / 0.396 / 0.351 | 100% / 21% / 16% |
| Forward only | 0.694 / 0.393 / 0.202 | 11% / 4% / 2% |
| Do nothing | 0.150 / 0.086 / 0.123 | 0% |
| Random | 0.000 / 0.000 / 0.002 | 11% / 2% / 3% |

- With more than one train, the heuristic clearly beats the RL checkpoint, which gets only ~20% of
  trains home (consistent with the organisers' full-suite result, 10.73 vs 9.42).
- **No policy serves a single intermediate stop** in any of the 70 episodes that have them.
- **Always report arrivals next to the score.** A train that departs but never arrives costs about
  100 penalty points, often small relative to the episode length, so "forward only" scores 0.694 on
  single-train scenarios while getting 1 in 9 trains home.

On the 6-train sample scenario alone the ranking flips (checkpoint 0.730 > heuristic 0.513), which is
why a single scenario is not a benchmark.

**Stopping a moving train always costs 250, even by choice.** Flatland 4.2.6 (and 4.3.0) charges the
COLLISION penalty on every MOVING→STOPPED transition, because the speed it penalises is read before
braking is applied (`rail_env.py`, line 516), although the reward code's comment says otherwise. Since
the competition ran on 4.2.5/4.2.6, official scores work the same way. So **serving an intermediate stop
costs 250 while skipping it costs 50**: the official reward makes skipping stops rational, which explains
why no baseline serves any. Worth discussing in the report (reward design vs. real-world goals).

**Warm-started IPPO on 5–10 interacting trains (E3 in `EXPERIMENTS.md`, 3 training seeds, 50 unseen
validation scenarios):** after 100 iterations the official score rises from 0.52 to 0.78 (+0.26, 95% CI
+0.20 to +0.32, above the heuristic's 0.64), but the share of trains that arrive *falls* from 45% to 38%
(−7%, CI −14% to −1%), against the heuristic's 86%. The score gain is fewer halts. Arrivals peak early
(two of three seeds are best at iteration 20, 68% arrived on average) and erode with further training:
giving way means stopping, and stopping costs 250, so the official score rewards *less* coordination.
**Decided 25 Oct:** from here on, both IPPO and MAPPO warm-start from the checkpoint and train without the
250 for a chosen STOP (forced halts still cost 250); all reported numbers stay official. Networks and logs:
`results/e3_training/`; validation results: `results/e3_val_seeds/`.

**Training without the 250 for a chosen STOP (E4, E5 in `EXPERIMENTS.md`):** making a chosen STOP free (E4)
backfired: every seed learned to stop and wait (240–330 halts per training game; final arrivals 0–26%),
because lateness and not arriving are only charged at the end. Charging a chosen STOP 25 and every step not
yet arrived 0.5 (E5) fixed that. Networks and logs: `results/e5_training/`.

**Evaluate trained actors with sampled actions (26 Oct).** Trained actors learned to depart with about 50%
probability per step, which staggers trains when actions are sampled. Taking the top action turns that into
"all trains depart at once" or "none ever departs", which looked like training collapse (E5 seed 0: 12% greedy,
61% sampled). With sampled actions (`actor-sampled:<path>`, 3 seeds) on 50 validation scenarios, final
snapshots: **E5 63% arrived** (sd 2% over training seeds), E3 42%, checkpoint 47%, heuristic 86%; **E5 vs E3
+21% [+18, +24]**, E5 vs checkpoint +15% [+10, +20]. E3's decline with training is real; E5's reward fixes it.
Sampled is now the main number, greedy secondary. Results: `results/{e3,e5}_val_sampled/`,
`results/val_sampled_baselines/`.

**MAPPO (26 Oct, option A).** `critic: central` adds a branch to the critic: it sees the actor's trunk
features plus `obs.global_state` (mean of the 36 observation features over active trains + shares of trains
arrived / on the map / broken down + elapsed time), and its output is added to the actor's value head. Its
last layer starts at zero, so MAPPO's critic starts exactly equal to IPPO's; the only difference between the
arms is what the critic can see. The actor, warm start, reward, budget and evaluation are identical, and the
saved `actor.pt` never contains the critic (decentralised execution). `critic: local` is unchanged
(checked bit-identical against the previous commit). IPPO = E6, MAPPO = E7 (`EXPERIMENTS.md`).

**IPPO arm settled (E6, 26 Oct).** E5's setup trained for 200 iterations: arrivals level off at about 64–66% from
iteration ~120. Final snapshots, sampled validation: **64% arrived** (sd 2% over seeds), +16% [+11, +21] vs the
checkpoint, +22% [+19, +24] vs E3, −22% vs the heuristic; +1% over E5's iteration 100, so 200 iterations is
enough. Its first 100 iterations reproduce E5 bit for bit. Results: `results/e6_training/`, `results/e6_val_sampled/`.

**MAPPO vs IPPO on validation (E7 vs E6, 26 Oct).** Identical setups except the critic's input. Final snapshots,
sampled validation (50 scenarios, 3 training seeds each): **MAPPO 70% arrived vs IPPO 64%, +6% [+4, +8]**; every
MAPPO seed above every IPPO seed; better in 40 of 50 scenarios; same official score; slightly less lateness. Averaged
over the last 10 snapshots the gap is +2.4%, so the advantage is consistent but modest (about +2 to +6 points).
MAPPO does not give way more (1.5 vs 1.3 halts per game); the heuristic still leads (86%). Results:
`results/e7_training/`, `results/e7_val_sampled/`.

**Main result: frozen test sets (E8, 26 Oct; analysis pre-registered).** 200 unseen scenarios: T1 (scene 1, 5–10
trains, 100 scenarios), T2 (15–20 trains), T3 (scene 4, never trained on), each with and without breakdowns; final
snapshots, sampled actions, 3 training seeds per arm. **Primary (T1, no disruptions): MAPPO 66.9% vs IPPO 61.5%
arrived, +5.4% [+4.0, +6.8]**, every MAPPO seed above every IPPO seed. MAPPO is ahead in every setting (+3.8% to
+7.2%), with less lateness and the same score and fairness. In dense traffic (T2) MAPPO matches the heuristic on
arrivals (−2.5% [−9.3, +4.5]); on T1/T3 the heuristic leads by 13–17 points. With greedy (top-action) execution
IPPO collapses (23–26%) while MAPPO keeps 47–73%. Tables: `EXPERIMENTS.md` (E8), `results/e8_test/summary_*.csv`.

## Notes

- `flatland-rl` must stay at **4.2.6** (competition requirement). Don't upgrade it to 4.3.x.
- `third_party/`, `.venv/` and `runs/` are gitignored. Never commit course PDFs.
- **Flatland leaks memory across episodes** (its `lru_cache`s keep every env alive, ~45 MB each). Any long-lived
  process that plays many episodes must call `flatmarl.envs.clear_flatland_caches()` after each one, as
  `evaluate.py` and `rollout.py` do; their worker processes are also replaced every 50 episodes.
