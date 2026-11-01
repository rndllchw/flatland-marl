"""E9: both arms trained to 400 iterations, all pre-registered outcomes (EXPERIMENTS.md, E9; secondary to E8).

Reads results/e9_test/<set>/<none|breakdowns>/{ippo,mappo}_sampled (and {ippo,mappo}_greedy for T1 none), made by
scripts/run_e9.sh, and the iteration-200 arms and baselines from results/e8_test (same scenarios and seeds). Writes:
    results/e9_test/summary_arrivals.csv    one row per set x disruption x mode: arrivals of each arm at 200 and 400,
                                            and with 95% CI: MAPPO - IPPO at 400 (primary on T1 / none / sampled),
                                            each arm 400 - 200 (outcome 1), the change in the gap (outcome 2), each
                                            arm at 400 - heuristic (outcome 3); other rows are outcomes 4 and 7
    results/e9_test/summary_metrics.csv     sampled only (outcome 5): score, lateness, p90 lateness, halts, Gini, ...
                                            per arm at 200 and 400, MAPPO - IPPO at 400 and each arm 400 - 200
    results/e9_test/summary_validation.csv  outcome 6: validation arrivals, mean of iterations 151-200 and 351-400

Per scenario, each arm is the mean over its 3 training seeds and the evaluation seeds; differences are paired over
scenarios with a 95% bootstrap CI (scripts/val_analysis.paired_ci: 10,000 resamples, seed 0), as in E8. The change
in the gap is [MAPPO(400) - IPPO(400)] - [MAPPO(200) - IPPO(200)] per scenario, bootstrapped the same way.

    python scripts/e9_analysis.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import val_analysis as va  # noqa: E402
from e8_analysis import CHECKPOINT_SAMPLED, METRICS, SETS, baseline, diff  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e9_test"
E8 = ROOT / "results" / "e8_test"
E8_FOLDERS = {"ippo": "e6", "mappo": "e7"}   # iteration-200 arms in results/e8_test
E9_FOLDERS = {"ippo": "ippo", "mappo": "mappo"}
GREEDY = {("t1_scene1_a5-10_ll2", "none")}   # E9 evaluates greedy play here only
VALIDATION = {"ippo": ("e6_val_sampled", "e9_ippo_val_sampled"), "mappo": ("e7_val_sampled", "e9_mappo_val_sampled")}


def arm(folder: Path, iteration: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Episodes and per-scenario means; stops unless every policy is a snapshot of the expected iteration."""
    e = va.load_snapshots(folder)
    if set(e.iteration) != {iteration}:
        raise ValueError(f"{folder}: expected iteration {iteration} only, found {sorted(set(e.iteration))}")
    return e, e.groupby("scenario")[METRICS].mean()


def same_design(old: pd.DataFrame, new: pd.DataFrame, where: Path):
    """The paired comparison needs the same scenarios, training seeds and evaluation seeds at 200 and 400."""
    key = lambda e: set(zip(e.scenario, e.seed, e.eval_seed))  # noqa: E731
    if key(old) != key(new):
        raise ValueError(f"{where}: scenarios / seeds differ between the iteration-200 and new runs")


def analyse(new_root: Path = OUT, new_folders: dict = E9_FOLDERS, new_iteration: int = 400,
            old_root: Path = E8, old_folders: dict = E8_FOLDERS, old_iteration: int = 200):
    arrivals, metrics = [], []
    for s in SETS:
        for d in ("none", "breakdowns"):
            heuristic = baseline(old_root / s / d / "baselines", "heuristic")
            checkpoint = baseline(old_root / s / d / "baselines", CHECKPOINT_SAMPLED)
            for mode in ("sampled", "greedy") if (s, d) in GREEDY else ("sampled",):
                episodes, means = {}, {}
                for name in ("ippo", "mappo"):
                    eo, mo = arm(old_root / s / d / f"{old_folders[name]}_{mode}", old_iteration)
                    en, mn = arm(new_root / s / d / f"{new_folders[name]}_{mode}", new_iteration)
                    same_design(eo, en, new_root / s / d / f"{new_folders[name]}_{mode}")
                    episodes[name] = en
                    means[name, 200], means[name, 400] = mo, mn
                gap = {it: (means["mappo", it][["completion_rate"]] - means["ippo", it][["completion_rate"]])
                       for it in (200, 400)}
                row = {"set": s, "disruption": d, "mode": mode,
                       **{f"{name}_{it}": means[name, it].completion_rate.mean()
                          for name in ("ippo", "mappo") for it in (200, 400)},
                       **{f"{name}_400_seeds": "/".join(f"{x:.3f}" for x in
                                                        episodes[name].groupby("seed").completion_rate.mean())
                          for name in ("ippo", "mappo")},
                       "heuristic": heuristic.completion_rate.mean(),
                       "checkpoint_sampled": checkpoint.completion_rate.mean()}
                for label, (a, b) in {"mappo_minus_ippo_400": (means["mappo", 400], means["ippo", 400]),
                                      "mappo_minus_ippo_200": (means["mappo", 200], means["ippo", 200]),
                                      "ippo_400_minus_200": (means["ippo", 400], means["ippo", 200]),
                                      "mappo_400_minus_200": (means["mappo", 400], means["mappo", 200]),
                                      "gap_change": (gap[400], gap[200]),
                                      "ippo_400_minus_heuristic": (means["ippo", 400], heuristic),
                                      "mappo_400_minus_heuristic": (means["mappo", 400], heuristic)}.items():
                    row[label], row[f"{label}_lo"], row[f"{label}_hi"] = diff(a, b, "completion_rate")
                arrivals.append(row)
                if mode == "sampled":
                    for c in METRICS:
                        m = {"set": s, "disruption": d, "metric": c, "heuristic": heuristic[c].mean(),
                             **{f"{name}_{it}": means[name, it][c].mean()
                                for name in ("ippo", "mappo") for it in (200, 400)}}
                        for label, (a, b) in {"mappo_minus_ippo_400": (means["mappo", 400], means["ippo", 400]),
                                              "ippo_400_minus_200": (means["ippo", 400], means["ippo", 200]),
                                              "mappo_400_minus_200": (means["mappo", 400], means["mappo", 200])
                                              }.items():
                            m[label], m[f"{label}_lo"], m[f"{label}_hi"] = diff(a, b, c)
                        metrics.append(m)
    return pd.DataFrame(arrivals), pd.DataFrame(metrics)


def validation_windows() -> pd.DataFrame:
    """Outcome 6: validation arrivals per arm and training seed, mean of iterations 151-200 and 351-400."""
    rows = []
    for name, folders in VALIDATION.items():
        v = pd.concat([pd.read_csv(ROOT / "results" / f / "summary_by_snapshot.csv") for f in folders])
        for seed, g in v.groupby("seed"):
            early = g[(g.iteration > 150) & (g.iteration <= 200)].completion_rate
            late = g[(g.iteration > 350) & (g.iteration <= 400)].completion_rate
            assert len(early) == len(late) == 5, f"{name} seed {seed}: expected 5 snapshots per window"
            rows.append({"arm": name, "seed": seed, "iterations_151_200": early.mean(),
                         "iterations_351_400": late.mean(), "change": late.mean() - early.mean()})
    v = pd.DataFrame(rows)
    means = v.groupby("arm", as_index=False)[["iterations_151_200", "iterations_351_400", "change"]].mean()
    return pd.concat([v, means.assign(seed="mean")], ignore_index=True)


def main():
    arrivals, metrics = analyse()
    validation = validation_windows()
    arrivals.to_csv(OUT / "summary_arrivals.csv", index=False)
    metrics.to_csv(OUT / "summary_metrics.csv", index=False)
    validation.to_csv(OUT / "summary_validation.csv", index=False)

    ci = lambda r, c: f"{getattr(r, c):+.1%} [{getattr(r, c + '_lo'):+.1%}, {getattr(r, c + '_hi'):+.1%}]"  # noqa
    print("Arrived: IPPO 200 -> 400 | MAPPO 200 -> 400 | MAPPO - IPPO at 400 | gap change | IPPO / MAPPO 400 - 200")
    for r in arrivals.itertuples():
        print(f"  {r.set[:2]} {r.disruption:10s} {r.mode:7s} {r.ippo_200:6.1%} -> {r.ippo_400:6.1%} | "
              f"{r.mappo_200:6.1%} -> {r.mappo_400:6.1%} | {ci(r, 'mappo_minus_ippo_400')} | {ci(r, 'gap_change')} | "
              f"{ci(r, 'ippo_400_minus_200')} / {ci(r, 'mappo_400_minus_200')}")
    print("Validation (outcome 6), mean arrivals of iterations 151-200 -> 351-400:")
    for r in validation[validation.seed == "mean"].itertuples():
        print(f"  {r.arm:5s} {r.iterations_151_200:.1%} -> {r.iterations_351_400:.1%} ({r.change:+.1%})")

    p = arrivals[(arrivals.set == SETS[0]) & (arrivals.disruption == "none") & (arrivals["mode"] == "sampled")].iloc[0]
    print("Pre-registered reading (T1, no disruptions, sampled):")
    print(f"  E9 primary, MAPPO - IPPO at 400: {ci(p, 'mappo_minus_ippo_400')}")
    improving = ("yes" if p.mappo_400_minus_200_lo > 0 else "no, it got worse" if p.mappo_400_minus_200_hi < 0
                 else "no (CI includes 0)")
    print(f"  MAPPO keeps improving: {improving} (400 - 200 {ci(p, 'mappo_400_minus_200')})")
    gap = ("widened" if p.gap_change_lo > 0 else "narrowed" if p.gap_change_hi < 0 else "neither (CI includes 0)")
    print(f"  gap: {gap} ({ci(p, 'gap_change')})")
    print(f"wrote {OUT / 'summary_arrivals.csv'}, {OUT / 'summary_metrics.csv'} and {OUT / 'summary_validation.csv'}")


if __name__ == "__main__":
    main()
