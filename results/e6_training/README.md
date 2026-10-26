# E6 training runs: the IPPO arm (E5's setup, 200 iterations), seeds 0, 1, 2

Experiment E6 in [`EXPERIMENTS.md`](../../EXPERIMENTS.md): IPPO fine-tuned from the organisers' checkpoint on
scene 1, 5–10 trains, no intermediate stops, no breakdowns; chosen STOPs cost 25 instead of 250, waiting costs 0.5
per step, value normalisation; 200 iterations × 32 games (the final budget). Quick checks sample their actions.
Copied from `runs/` (gitignored) so the trained networks and logs are kept in the repo. **This is the IPPO arm of
the IPPO vs MAPPO comparison.**

| Seed | Training command | Code (git commit) | Date (UTC) |
|---|---|---|---|
| 0 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_stopcost_200.yaml --seed 0` | `7944171` | 2026-10-25 20:17 |
| 1 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_stopcost_200.yaml --seed 1` | `7944171` | 2026-10-25 21:43 |
| 2 | `python scripts/train.py --config configs/ippo_warm_scene1_a5-10_ll2_stopcost_200.yaml --seed 2` | `7944171` | 2026-10-25 23:03 |

Training pool: `pools/scene1_a5-10_ll2` (rebuild with the command in `EXPERIMENTS.md`, E3).

## Files

| File | Contents |
|---|---|
| `seed_<s>/actor_<i>.pt` | the network after iteration `i` (10, 20, …, 200), as `{"model": state_dict}`. Evaluate with `--policies actor-sampled:results/e6_training/seed_<s>/actor_<i>.pt --seeds 0 1 2`. `actor_200.pt` has the same weights as the run's `actor.pt` (checked); `actor_100.pt` is bit-identical to `results/e5_training/seed_<s>/actor_100.pt` |
| `seed_<s>/eval_log.csv` | quick check with **sampled** actions on 16 fixed training-pool games, at iterations 0, 10, …, 200 |
| `seed_<s>/train_log.csv` | one row per iteration: official results of the sampled training games, PPO statistics, `exempted_halts`, `return_mean` / `return_std`, `agent_steps` / `agent_steps_total` |
| `seed_<s>/config.yaml` | the full config used, defaults filled in |
| `seed_<s>/run.json` | command, seed, git commit, versions, date |
| `sampled_checks_all_seeds.csv` | the three `eval_log.csv` files in one table, with a `seed` column |

Validation results for all 60 snapshots on 50 unseen scenarios (sampled, 3 seeds): `results/e6_val_sampled/`
(per-snapshot summary: `summary_by_snapshot.csv`). Final snapshots: 64% arrived (sd 2% over seeds).
