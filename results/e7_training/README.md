# E7 training runs: the MAPPO arm (E6 + central critic), seeds 0, 1, 2

Experiment E7 in [`EXPERIMENTS.md`](../../EXPERIMENTS.md): identical to E6 (IPPO fine-tuned from the organisers'
checkpoint on scene 1, 5–10 trains, no intermediate stops, no breakdowns; chosen STOPs cost 25, waiting 0.5 per
step, value normalisation; 200 iterations × 32 games; sampled quick checks) except `critic: central`: the value
head gets a branch that also sees a summary of all trains (`flatmarl.obs.global_state`). Copied from `runs/`
(gitignored) so the trained networks and logs are kept in the repo. **This is the MAPPO arm of the comparison.**

| Seed | Training command | Code (git commit) | Date (UTC) |
|---|---|---|---|
| 0 | `python scripts/train.py --config configs/mappo_warm_scene1_a5-10_ll2_stopcost_200.yaml --seed 0` | `b6949c1` | 2026-10-26 07:04 |
| 1 | `python scripts/train.py --config configs/mappo_warm_scene1_a5-10_ll2_stopcost_200.yaml --seed 1` | `b6949c1` | 2026-10-26 07:22 |
| 2 | `python scripts/train.py --config configs/mappo_warm_scene1_a5-10_ll2_stopcost_200.yaml --seed 2` | `b6949c1` | 2026-10-26 07:40 |

Training pool: `pools/scene1_a5-10_ll2` (rebuild with the command in `EXPERIMENTS.md`, E3).

## Files

| File | Contents |
|---|---|
| `seed_<s>/actor_<i>.pt` | the actor after iteration `i` (10, 20, …, 200), as `{"model": state_dict}` only (no critic inside: decentralised execution). Evaluate with `--policies actor-sampled:results/e7_training/seed_<s>/actor_<i>.pt --seeds 0 1 2`. `actor_200.pt` has the same weights as the run's `actor.pt` (checked) |
| `seed_<s>/critic_<i>.pt` | the central critic's weights at the same iterations, as `{"critic": state_dict}` (training only; never needed to evaluate) |
| `seed_<s>/eval_log.csv` | quick check with **sampled** actions on 16 fixed training-pool games, at iterations 0, 10, …, 200 |
| `seed_<s>/train_log.csv` | one row per iteration: official results of the sampled training games, PPO statistics, `exempted_halts`, `return_mean` / `return_std`, `agent_steps` / `agent_steps_total` |
| `seed_<s>/config.yaml` | the full config used, defaults filled in |
| `seed_<s>/run.json` | command, seed, git commit, versions, date |
| `sampled_checks_all_seeds.csv` | the three `eval_log.csv` files in one table, with a `seed` column |

Validation results for all 60 snapshots on 50 unseen scenarios (sampled, 3 seeds): `results/e7_val_sampled/`
(per-snapshot summary: `summary_by_snapshot.csv`). Final snapshots: 70% arrived (sd 2% over seeds), +6% [+4, +8]
vs E6 (IPPO).
