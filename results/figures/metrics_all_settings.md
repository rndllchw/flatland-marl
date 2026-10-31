All six test settings, sampled actions (appendix).

| Setting: metric | IPPO | MAPPO | MAPPO − IPPO [95% CI] | Heuristic |
|---|---|---|---|---|
| T1, no disruptions (primary): Trains arrived (%) | 61.5 | 66.9 | +5.4 [+4.0, +6.8] | 81.0 |
| T1, no disruptions (primary): Official score | 0.830 | 0.823 | -0.006 [-0.016, +0.004] | 0.592 |
| T1, no disruptions (primary): p90 lateness (steps) | 4.9 | 3.1 | -1.8 [-3.0, -0.7] | 8.7 |
| T1, no disruptions (primary): Gini of train penalties | 0.611 | 0.622 | +0.012 [-0.005, +0.027] | 0.498 |
| T1, breakdowns: Trains arrived (%) | 60.9 | 65.9 | +5.0 [+3.9, +6.1] | 78.9 |
| T1, breakdowns: Official score | 0.767 | 0.782 | +0.015 [+0.005, +0.025] | 0.575 |
| T1, breakdowns: p90 lateness (steps) | 5.3 | 4.4 | -0.9 [-2.0, +0.1] | 10.0 |
| T1, breakdowns: Gini of train penalties | 0.617 | 0.626 | +0.009 [-0.007, +0.025] | 0.494 |
| T2 (15–20 trains), no disruptions: Trains arrived (%) | 54.5 | 58.9 | +4.5 [+2.7, +6.2] | 61.5 |
| T2 (15–20 trains), no disruptions: Official score | 0.732 | 0.734 | +0.002 [-0.011, +0.016] | 0.469 |
| T2 (15–20 trains), no disruptions: p90 lateness (steps) | 7.8 | 4.5 | -3.2 [-4.9, -1.7] | 11.2 |
| T2 (15–20 trains), no disruptions: Gini of train penalties | 0.628 | 0.642 | +0.015 [+0.006, +0.024] | 0.530 |
| T2 (15–20 trains), breakdowns: Trains arrived (%) | 53.7 | 57.6 | +3.8 [+2.2, +5.5] | 58.6 |
| T2 (15–20 trains), breakdowns: Official score | 0.604 | 0.635 | +0.031 [+0.019, +0.043] | 0.453 |
| T2 (15–20 trains), breakdowns: p90 lateness (steps) | 11.8 | 9.3 | -2.4 [-4.5, -0.4] | 15.9 |
| T2 (15–20 trains), breakdowns: Gini of train penalties | 0.604 | 0.614 | +0.010 [-0.002, +0.022] | 0.517 |
| T3 (unseen region), no disruptions: Trains arrived (%) | 72.2 | 77.6 | +5.3 [+3.3, +7.3] | 94.8 |
| T3 (unseen region), no disruptions: Official score | 0.828 | 0.831 | +0.004 [-0.013, +0.018] | 0.800 |
| T3 (unseen region), no disruptions: p90 lateness (steps) | 7.0 | 5.6 | -1.4 [-3.4, +0.6] | 6.0 |
| T3 (unseen region), no disruptions: Gini of train penalties | 0.647 | 0.648 | +0.001 [-0.024, +0.026] | 0.631 |
| T3 (unseen region), breakdowns: Trains arrived (%) | 70.0 | 77.2 | +7.2 [+5.2, +9.2] | 94.4 |
| T3 (unseen region), breakdowns: Official score | 0.797 | 0.822 | +0.025 [+0.007, +0.042] | 0.778 |
| T3 (unseen region), breakdowns: p90 lateness (steps) | 8.7 | 7.0 | -1.7 [-3.9, +0.7] | 9.2 |
| T3 (unseen region), breakdowns: Gini of train penalties | 0.661 | 0.654 | -0.007 [-0.032, +0.015] | 0.635 |

Per scenario: mean over 3 training x 3 evaluation seeds (heuristic: deterministic); then mean over scenarios. Difference: paired over scenarios, 95% bootstrap CI. Lateness only over trains that arrived, so the arms are averaged over slightly different trains. Gini: 0 = penalties shared equally, 1 = one train takes all.
