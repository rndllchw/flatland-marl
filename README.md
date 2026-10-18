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
ppo_checkpoint       score 0.730 (expected 0.730)  arrived 6/6 (expected 6)  stops served 0%  forced halts 0  ... OK
deadlock_heuristic   score 0.513 (expected 0.513)  arrived 6/6 (expected 6)  stops served 0%  forced halts 4  ... OK
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
m.completion_rate, m.mean_lateness, m.stop_service_rate, m.forced_halts, m.gini_penalty
m.trains[3].penalty_by_type  # where one train's penalty came from
m.as_row()                   # everything as one flat dict (a CSV row)
```

| Episode metric | Meaning |
|---|---|
| `score` | official normalised score (0–1) |
| `completion_rate` / `cancelled_rate` | share of trains that arrived / never departed |
| `on_time_rate`, `mean_lateness` | among arrived trains (lateness = steps late, early counts as 0) |
| `stop_service_rate` | intermediate stops served / required |
| `forced_halts` | times a train was blocked while moving (250 each) |
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
| `COLLISION` | blocked while moving: 250 |
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
| `--policies` | `do_nothing`, `forward`, `random`, `heuristic`, `checkpoint`, or `actor:<path>` (any network with the starter-kit `ActorCritic` shape, saved as `{"model": state_dict}`) |
| `--scenarios` | folder of `.pkl` scenarios, or one `.pkl` |
| `--disruption` | `file` (each scenario's own, default), `none`, `breakdowns`, `breakdowns+delays` |
| `--seeds` | one episode per seed (breakdown pattern + random policy); omitted = the file's stored randomness |
| `--workers` | episodes in parallel; results are identical to `--workers 1` |
| `--overwrite` | needed to replace existing results in `--out` |

Output in `--out`: `episodes.csv` (one row per episode), `trains.csv` (one row per train per
episode), `run.json` (command, git commit, versions, scenario list). Policies are defined in
`src/flatmarl/policies.py`; `make_policy(name)` gives a fresh (policy, observation builder) pair.

## Repo layout and status

```
scripts/setup_third_party.sh   ✅ clones starter kit @54edacd and flatland-baselines @1e9d4a6 into third_party/
scripts/smoke_test.py          ✅ quick end-to-end check against reference scores
src/flatmarl/envs.py           ✅ environment factory: make_env, load_scenario, disruption presets
tests/test_envs.py             ✅ 20 tests for envs.py
src/flatmarl/metrics.py        ✅ per-episode metrics (score, arrivals, lateness, missed stops, forced halts, fairness)
tests/test_metrics.py          ✅ 21 tests for metrics.py
src/flatmarl/policies.py       ✅ named baseline policies + actor:<path>
scripts/evaluate.py            ✅ run policies over saved scenarios -> episodes.csv, trains.csv, run.json
tests/test_evaluate.py         ✅ 7 tests for evaluate.py
results/baselines_example_curriculum/  ✅ first baseline table (5 policies x 27 scenarios)
scripts/make_eval_sets.py      ⬜ build frozen val/test scenario sets
src/flatmarl/{models,obs,rollout,reward,ppo}.py, scripts/train.py   ⬜ PPO trainer (IPPO / MAPPO)
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

## Notes

- `flatland-rl` must stay at **4.2.6** (competition requirement). Don't upgrade it to 4.3.x.
- `third_party/`, `.venv/` and `runs/` are gitignored. Never commit course PDFs.
