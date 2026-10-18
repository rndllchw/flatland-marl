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
ppo_checkpoint       score 0.730 (expected 0.730)  arrived 6/6 (expected 6)   ... OK
deadlock_heuristic   score 0.513 (expected 0.513)  arrived 6/6 (expected 6)   ... OK
smoke test PASSED
```

Run the unit tests (~15 s) before pushing changes to `src/`:

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

Rewards are always the official `ECML2026Rewards`.

## Repo layout and status

```
scripts/setup_third_party.sh   ✅ clones starter kit @54edacd and flatland-baselines @1e9d4a6 into third_party/
scripts/smoke_test.py          ✅ quick end-to-end check against reference scores
src/flatmarl/envs.py           ✅ environment factory: make_env, load_scenario, disruption presets
tests/test_envs.py             ✅ 17 tests for envs.py
src/flatmarl/metrics.py        ⬜ per-episode metrics (score, arrivals, delay, missed stops, blocked moves, fairness)
scripts/evaluate.py            ⬜ run a policy over saved scenarios -> results CSV
scripts/make_eval_sets.py      ⬜ build frozen val/test scenario sets
src/flatmarl/{models,obs,rollout,reward,ppo}.py, scripts/train.py   ⬜ PPO trainer (IPPO / MAPPO)
```

## What we know so far

On the starter-kit sample scenario (6 trains, no breakdowns), **neither baseline serves a single
intermediate stop**:

| Policy | Score | Arrived | Where the penalty comes from |
|---|---|---|---|
| Starter-kit PPO checkpoint | 0.730 | 6/6, all early | only missed stops (50 each) |
| Deadlock-avoidance heuristic | 0.513 | 6/6 | missed stops + 4 "blocked while moving" penalties (250 each) |

A policy that reliably stops at stations without getting blocked would beat both here.

## Notes

- `flatland-rl` must stay at **4.2.6** (competition requirement). Don't upgrade it to 4.3.x.
- `third_party/`, `.venv/` and `runs/` are gitignored. Never commit course PDFs.
