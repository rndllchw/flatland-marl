# Report and slide figures

All made by `python scripts/make_figures.py` (under 2 s, committed CSVs only; `--only <name>` for one).
Each figure is a `.pdf` (report) and a `.png` (slides, 200 dpi); each table a `.md` and a `.tex` (booktabs).
Colours are fixed across figures: MAPPO blue, IPPO orange, heuristic aqua, checkpoint grey; baselines are hollow
rings in `test_results`.

| File | Shows | Data | Caption must say |
|---|---|---|---|
| `test_results` | **Main result.** (a) arrivals per policy in the 6 test settings; (b) MAPPO − IPPO with 95% CI, primary setting in bold | `e8_test/summary_arrivals.csv` (sampled rows) | Sampled actions; per scenario mean over 3 training × 3 evaluation seeds, then over scenarios; CI paired over scenarios (bootstrap). T1 none is the pre-registered primary outcome |
| `failures` | Where each policy's trains end: lost / stuck / other / arrived (T1, no disruptions) | `e8_explore/failures_{summary,trains}.csv` | **Exploratory, not pre-registered.** Shares of all trains pooled, so arrivals differ slightly from `test_results` (per-episode means): IPPO 60.8 vs 61.5%, MAPPO 65.9 vs 66.9%, heuristic 79.6 vs 81.0%. Lost = moving in the last 50 steps and drove ≥ its shortest route; stuck = same cell for the last ≥ 50 steps |
| `validation_curves` | IPPO vs MAPPO validation arrivals over 400 iterations, per seed and mean; dotted lines at the networks E8 (200) and E9 (400) tested; heuristic for reference | `e6/e7_val_sampled` (10–200) + `e9_{ippo,mappo}_val_sampled` (210–400) `summary_by_snapshot.csv`, `val_sampled_baselines`, `e3_val_baselines` | Validation set (50 scenarios), sampled; iteration 0 = the starter-kit checkpoint both arms are warm-started from; 32 games per iteration; iterations 1–200 of the E9 runs are bit-identical to E6/E7, so the curves are one run each. End labels: mean at 200 → 400 |
| `budget_effect` | **E9 (secondary).** (a) each arm's test arrivals at iteration 200 (light dot) and 400 (filled), per setting; (b) MAPPO − IPPO at 200 (grey) and 400 (blue) with 95% CI; right column: change in the gap with CI | `e9_test/summary_arrivals.csv` (sampled rows) | Planned after E8's results were known, so secondary to E8 (pre-registered); iteration-200 values are E8's. Same averaging and paired bootstrap as `test_results`; gap change = [MAPPO − IPPO at 400] − [at 200] per scenario. Takeaway: IPPO catches up, MAPPO flat; 3 training seeds per arm |
| `dense_collapse` | **E10 (secondary) + E10 exploratory.** One column per arm. (a, b) validation arrivals of each dense-trained run (15–20 trains) per training seed, from the checkpoint at iteration 0, heuristic for reference; (c, d) share of trains by steps waited after their earliest departure, at iterations 200 and 400 | `e10_{ippo,mappo}_val_sampled/summary_by_snapshot.csv`, `e10_val_baselines/episodes.csv`, `e10_explore/collapse_trains.csv` | No arm mean on purpose (it would average collapsed and healthy runs). Validation: 50 scenarios × 3 evaluation seeds, every 20 iterations. Bars: replay of D1, no disruptions, evaluation seed 0 (all 21,084 trains match the stored results), exploratory. Takeaway: 5 of 6 runs collapse, and when they do 36–53% of trains wait more than 300 steps; MAPPO seed 1 is the only run that does not, and it is E10's whole MAPPO − IPPO gap |
| `reward_story` | E3 vs E5: (a) arrivals, (b) official score over 100 iterations | `e3/e5_val_sampled/summary_by_snapshot.csv` | E5 differs from E3 in the training reward (a chosen STOP costs 25 instead of flatland's 250; waiting costs 0.5 per step) **and** in value normalisation (both introduced in E4); E4 (failed) is described in the text only |
| `greedy_vs_sampled` | (a) T1 test arrivals with sampled vs top-action play per network; (b) mean P(go) when ready to depart | `e8_test/summary_arrivals.csv`, `e8_explore/departures.csv` | Panel (b) is from 20 **validation** scenarios with sampled play (not the test set); exploratory |
| `fairness_table` | T1 no disruptions: arrivals, score, mean / p90 lateness, halts, penalty Gini; MAPPO − IPPO with CI | `e8_test/summary_metrics.csv` | Lateness only over trains that arrived (arms averaged over slightly different trains); Gini 0 = penalties shared equally. Note is in the file |
| `metrics_all_settings` | Arrivals, score, p90 lateness, Gini in all 6 settings (appendix) | `e8_test/summary_metrics.csv` | As `fairness_table` |

**Slide versions:** `python scripts/make_figures.py --slides` writes the same figures with text 1.45× larger (figures
1.3× taller; `dense_collapse` also 1.3× wider) as PNGs to `results/figures/slides/`; tables are skipped. Same data, same
layout; the report files are not touched.

Videos for the slides are separate: `results/e8_explore/videos/` (`scripts/make_videos.py`).
