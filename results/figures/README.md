# Report and slide figures

All made by `python scripts/make_figures.py` (under 2 s, committed CSVs only; `--only <name>` for one).
Each figure is a `.pdf` (report) and a `.png` (slides, 200 dpi); each table a `.md` and a `.tex` (booktabs).
Colours are fixed across figures: MAPPO blue, IPPO orange, heuristic aqua, checkpoint grey; baselines are hollow
rings in `test_results`.

| File | Shows | Data | Caption must say |
|---|---|---|---|
| `test_results` | **Main result.** (a) arrivals per policy in the 6 test settings; (b) MAPPO − IPPO with 95% CI, primary setting in bold | `e8_test/summary_arrivals.csv` (sampled rows) | Sampled actions; per scenario mean over 3 training × 3 evaluation seeds, then over scenarios; CI paired over scenarios (bootstrap). T1 none is the pre-registered primary outcome |
| `failures` | Where each policy's trains end: lost / stuck / other / arrived (T1, no disruptions) | `e8_explore/failures_{summary,trains}.csv` | **Exploratory, not pre-registered.** Shares of all trains pooled, so arrivals differ slightly from `test_results` (per-episode means): IPPO 60.8 vs 61.5%, MAPPO 65.9 vs 66.9%, heuristic 79.6 vs 81.0%. Lost = moving in the last 50 steps and drove ≥ its shortest route; stuck = same cell for the last ≥ 50 steps |
| `validation_curves` | IPPO (E6) vs MAPPO (E7) validation arrivals over 200 iterations, per seed and mean; heuristic for reference | `e6/e7_val_sampled/summary_by_snapshot.csv`, `val_sampled_baselines`, `e3_val_baselines` | Validation set (50 scenarios), sampled; iteration 0 = the starter-kit checkpoint both arms are warm-started from; 32 games per iteration |
| `reward_story` | E3 vs E5: (a) arrivals, (b) official score over 100 iterations | `e3/e5_val_sampled/summary_by_snapshot.csv` | E5 differs from E3 in the training reward (a chosen STOP costs 25 instead of flatland's 250; waiting costs 0.5 per step) **and** in value normalisation (both introduced in E4); E4 (failed) is described in the text only |
| `greedy_vs_sampled` | (a) T1 test arrivals with sampled vs top-action play per network; (b) mean P(go) when ready to depart | `e8_test/summary_arrivals.csv`, `e8_explore/departures.csv` | Panel (b) is from 20 **validation** scenarios with sampled play (not the test set); exploratory |
| `fairness_table` | T1 no disruptions: arrivals, score, mean / p90 lateness, halts, penalty Gini; MAPPO − IPPO with CI | `e8_test/summary_metrics.csv` | Lateness only over trains that arrived (arms averaged over slightly different trains); Gini 0 = penalties shared equally. Note is in the file |
| `metrics_all_settings` | Arrivals, score, p90 lateness, Gini in all 6 settings (appendix) | `e8_test/summary_metrics.csv` | As `fairness_table` |

Videos for the slides are separate: `results/e8_explore/videos/` (`scripts/make_videos.py`).
