# E4 training runs: warm-started IPPO, chosen STOPs unpenalised + value normalisation, seeds 0, 1, 2

Experiment E4 in [`EXPERIMENTS.md`](../../EXPERIMENTS.md): E3's setup (IPPO fine-tuned from the organisers'
checkpoint on scene 1, 5–10 trains, no intermediate stops, no breakdowns) with `reward.exempt_chosen_stops`
and `ppo.value_norm` switched on. Copied from `runs/` (gitignored) so the trained networks and logs are kept
in the repo. **This run failed** (trains learned to stop and wait); kept as evidence for the report.

| Seed | Training command | Code (git commit) | Date (UTC) |
|---|---|---|---|
| 0 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_nostop.yaml --seed 0` | `bafc37c` | 2026-10-25 14:31 |
| 1 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_nostop.yaml --seed 1` | `bafc37c` | 2026-10-25 15:02 |
| 2 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_nostop.yaml --seed 2` | `bafc37c` | 2026-10-25 15:34 |

Training pool: `pools/scene1_a5-10_ll2` (rebuild with the command in `EXPERIMENTS.md`, E3).

## Files

Same layout as `results/e3_training/`:

| File | Contents |
|---|---|
| `seed_<s>/actor_<i>.pt` | the network after iteration `i` (10, 20, …, 100), as `{"model": state_dict}`. `actor_100.pt` has the same weights as the run's `actor.pt` (checked) |
| `seed_<s>/eval_log.csv` | greedy check: the network's top action on 16 fixed training-pool games, at iterations 0, 10, …, 100 |
| `seed_<s>/train_log.csv` | one row per iteration: official results of the sampled training games, PPO statistics, plus `exempted_halts` (chosen STOPs not charged, per game) and `return_mean` / `return_std` (value normalisation) |
| `seed_<s>/config.yaml` | the full config used, defaults filled in |
| `seed_<s>/run.json` | command, seed, git commit, versions, date |
| `greedy_checks_all_seeds.csv` | the three `eval_log.csv` files in one table, with a `seed` column |

No validation evaluation was run: the greedy checks already show the failure.
