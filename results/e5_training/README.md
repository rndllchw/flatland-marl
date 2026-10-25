# E5 training runs: warm-started IPPO, chosen STOP costs 25, waiting costs 0.5 per step, seeds 0, 1, 2

Experiment E5 in [`EXPERIMENTS.md`](../../EXPERIMENTS.md): E4's setup (IPPO fine-tuned from the organisers'
checkpoint on scene 1, 5–10 trains, no intermediate stops, no breakdowns; chosen STOPs exempt from the 250;
value normalisation) plus `reward.chosen_stop_cost: 25` and `reward.waiting_cost: 0.5`. Copied from `runs/`
(gitignored) so the trained networks and logs are kept in the repo.

| Seed | Training command | Code (git commit) | Date (UTC) |
|---|---|---|---|
| 0 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_stopcost.yaml --seed 0` | `1d221b1` | 2026-10-25 16:13 |
| 1 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_stopcost.yaml --seed 1` | `1d221b1` | 2026-10-25 16:43 |
| 2 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_stopcost.yaml --seed 2` | `1d221b1` | 2026-10-25 17:13 |

Training pool: `pools/scene1_a5-10_ll2` (rebuild with the command in `EXPERIMENTS.md`, E3).

## Files

Same layout as `results/e3_training/`:

| File | Contents |
|---|---|
| `seed_<s>/actor_<i>.pt` | the network after iteration `i` (10, 20, …, 100), as `{"model": state_dict}`. Evaluate with `--policies actor:results/e5_training/seed_<s>/actor_<i>.pt`. `actor_100.pt` has the same weights as the run's `actor.pt` (checked) |
| `seed_<s>/eval_log.csv` | greedy check: the network's top action on 16 fixed training-pool games, at iterations 0, 10, …, 100 |
| `seed_<s>/train_log.csv` | one row per iteration: official results of the sampled training games, PPO statistics, `exempted_halts`, `return_mean` / `return_std` |
| `seed_<s>/config.yaml` | the full config used, defaults filled in |
| `seed_<s>/run.json` | command, seed, git commit, versions, date |
| `greedy_checks_all_seeds.csv` | the three `eval_log.csv` files in one table, with a `seed` column |

Validation results for all 30 snapshots on 50 unseen scenarios: `results/e5_val_seeds/` (per-snapshot summary:
`summary_by_snapshot.csv`). Best single network so far: `seed_1/actor_100.pt` (76% arrived on validation).
