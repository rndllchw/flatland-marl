# E3 training runs: warm-started IPPO, seeds 0, 1, 2

Experiment E3 in [`EXPERIMENTS.md`](../../EXPERIMENTS.md): IPPO fine-tuned from the organisers' checkpoint on
scene 1, 5–10 trains, no intermediate stops, no breakdowns. Copied from `runs/` (gitignored) so the trained
networks and logs are kept in the repo.

| Seed | Training command | Code (git commit) | Date (UTC) |
|---|---|---|---|
| 0 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2.yaml --seed 0` | `25f05a5` + uncommitted changes, identical to the code later committed as `6540847` | 2026-10-21 01:18 |
| 1 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2.yaml --seed 1` | `6540847` | 2026-10-21 06:48 |
| 2 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2.yaml --seed 2` | `6540847` | 2026-10-21 07:20 |

Training pool: `pools/scene1_a5-10_ll2` (rebuild with the command in `EXPERIMENTS.md`, E3).

## Files

| File | Contents |
|---|---|
| `seed_<s>/actor_<i>.pt` | the network after iteration `i` (10, 20, …, 100), as `{"model": state_dict}`. Evaluate with `--policies actor:results/e3_training/seed_<s>/actor_<i>.pt`. `actor_100.pt` is the final network (same weights as the run's `actor.pt`, not copied) |
| `seed_<s>/eval_log.csv` | greedy check: the network's top action on 16 fixed training-pool games, at iterations 0, 10, …, 100 |
| `seed_<s>/train_log.csv` | one row per iteration: official results of that iteration's sampled training games, plus PPO statistics |
| `seed_<s>/config.yaml` | the full config used, defaults filled in |
| `seed_<s>/run.json` | command, seed, git commit, versions, date |
| `greedy_checks_all_seeds.csv` | the three `eval_log.csv` files in one table, with a `seed` column |

Greedy checks use training-pool games and only 16 of them, so they are noisy. Validation results for all
30 snapshots on 50 unseen scenarios are in `results/e3_val_seeds/` (per-snapshot summary:
`summary_by_snapshot.csv`).
