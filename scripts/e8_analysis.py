"""E8: IPPO (E6) vs MAPPO (E7) on the frozen test sets, all pre-registered outcomes (EXPERIMENTS.md, E8).

Reads results/e8_test/<set>/<none|breakdowns>/{e6_sampled, e7_sampled, e6_greedy, e7_greedy, baselines} and writes:
    results/e8_test/summary_arrivals.csv   one row per set x disruption x mode (sampled / greedy): arrivals of
                                           each arm (mean and per training seed), MAPPO - IPPO with 95% CI, and
                                           each arm vs the checkpoint (sampled) and the heuristic
    results/e8_test/summary_metrics.csv    sampled only: score, lateness, p90 lateness, halts, penalty Gini,
                                           never-departed, per arm, with MAPPO - IPPO and 95% CI

Per scenario, each arm is the mean over its 3 training seeds and 3 evaluation seeds; differences are paired over
scenarios with a 95% bootstrap CI (scripts/val_analysis.paired_ci: 10,000 resamples, seed 0). The primary outcome
is the T1 / none / sampled row of summary_arrivals.csv (same number as scripts/val_analysis.py gives).

    python scripts/e8_analysis.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import val_analysis as va  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e8_test"
SETS = ("t1_scene1_a5-10_ll2", "t2_scene1_a15-20_ll2", "t3_scene4_a5-10_ll2")
CHECKPOINT_SAMPLED = "actor-sampled:third_party/ecml2026-starterkit/reinforcement_learning/checkpoint.pt"
METRICS = ["score", "completion_rate", "mean_lateness", "p90_lateness", "halts_while_moving", "gini_penalty",
           "cancelled_rate"]


def arm(folder: Path):
    e = va.load_snapshots(folder)
    return e, e.groupby("scenario")[METRICS].mean()


def baseline(folder: Path, policy: str) -> pd.DataFrame:
    e = pd.read_csv(folder / "episodes.csv", dtype={"scenario": str})
    return e[e.policy == policy].groupby("scenario")[METRICS].mean()


def diff(a: pd.DataFrame, b: pd.DataFrame, column: str) -> tuple[float, float, float]:
    """Paired over scenarios where both have a value (lateness is undefined when no train arrived)."""
    a, b = a[[column]].dropna(), b[[column]].dropna()
    shared = a.index.intersection(b.index)
    return va.paired_ci(a.loc[shared], b.loc[shared], column)


def main():
    arrivals, metrics = [], []
    for s in SETS:
        for d in ("none", "breakdowns"):
            folder = OUT / s / d
            checkpoint = baseline(folder / "baselines", CHECKPOINT_SAMPLED)
            checkpoint_greedy = baseline(folder / "baselines", "checkpoint")
            heuristic = baseline(folder / "baselines", "heuristic")
            for mode in ("sampled", "greedy"):
                (e6, ippo), (e7, mappo) = arm(folder / f"e6_{mode}"), arm(folder / f"e7_{mode}")
                row = {"set": s, "disruption": d, "mode": mode,
                       "ippo_arrived": ippo.completion_rate.mean(), "mappo_arrived": mappo.completion_rate.mean(),
                       "ippo_seeds": "/".join(f"{x:.3f}" for x in e6.groupby("seed").completion_rate.mean()),
                       "mappo_seeds": "/".join(f"{x:.3f}" for x in e7.groupby("seed").completion_rate.mean()),
                       "checkpoint_sampled_arrived": checkpoint.completion_rate.mean(),
                       "checkpoint_greedy_arrived": checkpoint_greedy.completion_rate.mean(),
                       "heuristic_arrived": heuristic.completion_rate.mean()}
                for name, (a, b) in {"mappo_minus_ippo": (mappo, ippo), "ippo_minus_checkpoint": (ippo, checkpoint),
                                     "mappo_minus_checkpoint": (mappo, checkpoint),
                                     "ippo_minus_heuristic": (ippo, heuristic),
                                     "mappo_minus_heuristic": (mappo, heuristic)}.items():
                    row[name], row[f"{name}_lo"], row[f"{name}_hi"] = diff(a, b, "completion_rate")
                arrivals.append(row)
                if mode == "sampled":
                    for c in METRICS:
                        m, lo, hi = diff(mappo, ippo, c)
                        metrics.append({"set": s, "disruption": d, "metric": c, "ippo": ippo[c].mean(),
                                        "mappo": mappo[c].mean(), "heuristic": heuristic[c].mean(),
                                        "mappo_minus_ippo": m, "lo": lo, "hi": hi})
    arrivals, metrics = pd.DataFrame(arrivals), pd.DataFrame(metrics)
    arrivals.to_csv(OUT / "summary_arrivals.csv", index=False)
    metrics.to_csv(OUT / "summary_metrics.csv", index=False)

    pct = lambda x: f"{x:+.1%}"  # noqa: E731
    print("Arrived: IPPO | MAPPO | MAPPO - IPPO [95% CI] | heuristic | MAPPO - heuristic")
    for r in arrivals.itertuples():
        print(f"  {r.set[:2]} {r.disruption:10s} {r.mode:7s} {r.ippo_arrived:6.1%} | {r.mappo_arrived:6.1%} | "
              f"{pct(r.mappo_minus_ippo)} [{pct(r.mappo_minus_ippo_lo)}, {pct(r.mappo_minus_ippo_hi)}] | "
              f"{r.heuristic_arrived:5.1%} | {pct(r.mappo_minus_heuristic)} "
              f"[{pct(r.mappo_minus_heuristic_lo)}, {pct(r.mappo_minus_heuristic_hi)}]")
    print(f"wrote {OUT / 'summary_arrivals.csv'} and {OUT / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
