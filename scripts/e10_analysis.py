"""E10: both arms trained on 15-20 trains, all pre-registered outcomes (EXPERIMENTS.md, E10; secondary to E8).

Step 1, after validation and before the test run (early-stop rule and outcome 8):
    python scripts/e10_analysis.py --validation-only
reads results/e10_{ippo,mappo}_val_sampled (summary_by_snapshot.csv) and results/e10_val_baselines and writes
    results/e10_test/summary_validation.csv   per validation iteration: each arm's mean and per-seed arrivals, and
                                              the checkpoint (sampled) and heuristic on the same pool
    results/e10_test/learning_speed_{seeds,headstart}.csv   outcome 8, the measures of
                                              scripts/explore_learning_speed.py (validation every 20 iterations here,
                                              so the head start is read at MAPPO's iterations 100 and 200 only)
Early stop: if at iteration 200 both arms' validation mean is below the checkpoint's, stop and report.

Step 2, after scripts/run_e10.sh:
    python scripts/e10_analysis.py
also reads results/e10_test/d1_scene1_a15-20_ll2/<none|breakdowns>/{ippo,mappo}_{200,400}_sampled,
{ippo,mappo}_400_greedy (none only) and baselines, and results/e10_test/t1_scene1_a5-10_ll2/none/
{ippo,mappo}_400_sampled, and writes
    results/e10_test/summary_arrivals.csv   one row per disruption x mode on D1: arrivals of each arm at 200 and 400
                                            and, with 95% CI, MAPPO - IPPO at 400 (primary on none / sampled;
                                            outcome 4 on breakdowns, 6 greedy) and at 200 (1), each arm 400 - 200
                                            (2), each arm at 400 - heuristic and - checkpoint in the same mode (3)
    results/e10_test/summary_metrics.csv    sampled (outcome 5): score, lateness, p90 lateness, halts, Gini, ...
    results/e10_test/summary_transfer.csv   outcome 7 (exploratory): T1 none, dense-trained arms vs E9's arms at 400
                                            (results/e9_test, same scenarios and seeds)

Per scenario, each arm is the mean over its 3 training seeds and the evaluation seeds; differences are paired over
scenarios with a 95% bootstrap CI (scripts/val_analysis.paired_ci: 10,000 resamples, seed 0), as in E8 and E9.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import explore_learning_speed as ls  # noqa: E402
from e8_analysis import CHECKPOINT_SAMPLED, METRICS, SETS, baseline, diff  # noqa: E402
from e9_analysis import arm, same_design  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e10_test"
E8 = ROOT / "results" / "e8_test"
E9 = ROOT / "results" / "e9_test"
D1, T1 = "d1_scene1_a15-20_ll2", SETS[0]
ARMS = ("ippo", "mappo")
VALIDATION = {a: f"e10_{a}_val_sampled" for a in ARMS}
VAL_BASELINES = ROOT / "results" / "e10_val_baselines"
TRAINING = {a: f"e10_{a}_training" for a in ARMS}
VAL_ITERATIONS = list(range(20, 401, 20))
CHECKPOINT = {"sampled": CHECKPOINT_SAMPLED, "greedy": "checkpoint"}


def default_name(name: str, iteration: int, mode: str) -> tuple[str, int]:
    """Folder of one arm under <set>/<disruption>/, and the iteration all its policies must be."""
    return f"{name}_{iteration}_{mode}", iteration


def analyse(root: Path = OUT, test_set: str = D1, name=default_name):
    arrivals, metrics = [], []
    for d in ("none", "breakdowns"):
        folder = root / test_set / d
        heuristic = baseline(folder / "baselines", "heuristic")
        for mode in ("sampled", "greedy") if d == "none" else ("sampled",):
            checkpoint = baseline(folder / "baselines", CHECKPOINT[mode])
            iterations = (200, 400) if mode == "sampled" else (400,)
            episodes, means = {}, {}
            for a in ARMS:
                for it in iterations:
                    sub, expected = name(a, it, mode)
                    episodes[a, it], means[a, it] = arm(folder / sub, expected)
                if mode == "sampled":
                    same_design(episodes[a, 200], episodes[a, 400], folder / name(a, 400, mode)[0])
            row = {"set": test_set, "disruption": d, "mode": mode,
                   **{f"{a}_{it}": means[a, it].completion_rate.mean() for a in ARMS for it in iterations},
                   **{f"{a}_400_seeds": "/".join(f"{x:.3f}" for x in
                                                 episodes[a, 400].groupby("seed").completion_rate.mean())
                      for a in ARMS},
                   "heuristic": heuristic.completion_rate.mean(), "checkpoint": checkpoint.completion_rate.mean()}
            pairs = {"mappo_minus_ippo_400": (means["mappo", 400], means["ippo", 400]),
                     **{f"{a}_400_minus_heuristic": (means[a, 400], heuristic) for a in ARMS},
                     **{f"{a}_400_minus_checkpoint": (means[a, 400], checkpoint) for a in ARMS}}
            if mode == "sampled":
                pairs |= {"mappo_minus_ippo_200": (means["mappo", 200], means["ippo", 200]),
                          **{f"{a}_400_minus_200": (means[a, 400], means[a, 200]) for a in ARMS}}
            for label, (x, y) in pairs.items():
                row[label], row[f"{label}_lo"], row[f"{label}_hi"] = diff(x, y, "completion_rate")
            arrivals.append(row)
            if mode == "sampled":
                for c in METRICS:
                    m = {"set": test_set, "disruption": d, "metric": c, "heuristic": heuristic[c].mean(),
                         **{f"{a}_{it}": means[a, it][c].mean() for a in ARMS for it in iterations}}
                    m["mappo_minus_ippo_400"], m["lo"], m["hi"] = diff(means["mappo", 400], means["ippo", 400], c)
                    metrics.append(m)
    return pd.DataFrame(arrivals), pd.DataFrame(metrics)


def transfer(new_root: Path = OUT, new_name: str = "{arm}_400_sampled", old_root: Path = E9,
             old_name: str = "{arm}_sampled") -> pd.DataFrame:
    """Outcome 7: T1 none, sampled; dense-trained arms (new) against the arms trained on 5-10 trains (old), both
    at iteration 400. The heuristic comes from results/e8_test (same scenarios)."""
    heuristic = baseline(E8 / T1 / "none" / "baselines", "heuristic")
    episodes, means = {}, {}
    for a in ARMS:
        for which, root, pattern in (("new", new_root, new_name), ("old", old_root, old_name)):
            episodes[which, a], means[which, a] = arm(root / T1 / "none" / pattern.format(arm=a), 400)
        same_design(episodes["old", a], episodes["new", a], new_root / T1 / "none" / new_name.format(arm=a))
    row = {"set": T1, "disruption": "none", "mode": "sampled",
           **{f"{a}_{w}": means[w, a].completion_rate.mean() for w in ("new", "old") for a in ARMS},
           "heuristic": heuristic.completion_rate.mean()}
    pairs = {"mappo_minus_ippo_new": (means["new", "mappo"], means["new", "ippo"]),
             "mappo_minus_ippo_old": (means["old", "mappo"], means["old", "ippo"]),
             **{f"{a}_new_minus_old": (means["new", a], means["old", a]) for a in ARMS}}
    for label, (x, y) in pairs.items():
        row[label], row[f"{label}_lo"], row[f"{label}_hi"] = diff(x, y, "completion_rate")
    return pd.DataFrame([row])


def validation(root: Path = ROOT / "results", baselines: Path = VAL_BASELINES) -> pd.DataFrame:
    """Validation arrivals per iteration: each arm's mean over training seeds, each seed, and the two references."""
    table = pd.DataFrame(index=pd.Index(VAL_ITERATIONS, name="iteration"))
    for a in ARMS:
        v = pd.read_csv(root / VALIDATION[a] / "summary_by_snapshot.csv")
        c = v.pivot(index="iteration", columns="seed", values="completion_rate")
        if list(c.index) != VAL_ITERATIONS or list(c.columns) != [0, 1, 2] or c.isna().any().any():
            raise ValueError(f"{VALIDATION[a]}: expected seeds 0-2 at iterations 20, 40, ..., 400")
        table[a] = c.mean(axis=1)
        for s in c.columns:
            table[f"{a}_seed_{s}"] = c[s]
    e = pd.read_csv(baselines / "episodes.csv", dtype={"scenario": str})
    table["checkpoint_sampled"] = e[e.policy == CHECKPOINT_SAMPLED].completion_rate.mean()
    table["heuristic"] = e[e.policy == "heuristic"].completion_rate.mean()
    return table.reset_index()


def early_stop(v: pd.DataFrame) -> bool:
    """Pre-registered: stop if at iteration 200 both arms' validation mean is below the checkpoint's."""
    r = v.set_index("iteration").loc[200]
    return bool(r.ippo < r.checkpoint_sampled and r.mappo < r.checkpoint_sampled)


def learning_speed() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outcome 8: explore_learning_speed's measures (thresholds 60 / 65 / 67.5%, means over iterations 10-200 and
    210-400, head start); validation is every 20 iterations, so MAPPO's levels are read at 100 and 200."""
    return ls.learning_speed(arms={"IPPO": ((VALIDATION["ippo"],), TRAINING["ippo"]),
                                   "MAPPO": ((VALIDATION["mappo"],), TRAINING["mappo"])},
                             levels_at=(100, 200))


def reading(lo: float, hi: float) -> str:
    """The interpretation of the primary, fixed in advance."""
    if lo > 0:
        return "the central critic helps more in dense traffic (MAPPO ahead, CI excludes 0)"
    if hi < 0:
        return "MAPPO is behind IPPO in dense traffic (CI below 0)"
    return "no detectable difference in dense traffic either (CI includes 0)"


def ci(r, c: str) -> str:
    return f"{getattr(r, c):+.1%} [{getattr(r, c + '_lo'):+.1%}, {getattr(r, c + '_hi'):+.1%}]"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--validation-only", action="store_true", help="step 1: before the test run")
    args = parser.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    v = validation()
    seeds, head = learning_speed()
    v.to_csv(OUT / "summary_validation.csv", index=False)
    seeds.to_csv(OUT / "learning_speed_seeds.csv", index=False)
    head.to_csv(OUT / "learning_speed_headstart.csv", index=False)
    r = v.set_index("iteration")
    print("Validation arrivals (dense pool, sampled, mean of 3 training seeds):")
    print("  iteration   IPPO   MAPPO")
    for it in (20, 100, 200, 300, 400):
        print(f"  {it:9d} {r.loc[it, 'ippo']:6.1%} {r.loc[it, 'mappo']:6.1%}")
    print(f"  checkpoint (sampled) {r.checkpoint_sampled.iloc[0]:.1%}, heuristic {r.heuristic.iloc[0]:.1%}")
    stop = early_stop(v)
    print(f"Early-stop rule at iteration 200: {'STOP - both arms below the checkpoint' if stop else 'continue'}")
    print("Outcome 8, learning speed (exploratory):")
    print(seeds.astype(object).fillna("not reached").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(head.astype(object).fillna("not reached").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.validation_only:
        print(f"wrote {OUT / 'summary_validation.csv'} and {OUT / 'learning_speed_*.csv'}")
        return

    arrivals, metrics = analyse()
    t1 = transfer()
    arrivals.to_csv(OUT / "summary_arrivals.csv", index=False)
    metrics.to_csv(OUT / "summary_metrics.csv", index=False)
    t1.to_csv(OUT / "summary_transfer.csv", index=False)

    print("D1 arrived: IPPO 200 -> 400 | MAPPO 200 -> 400 | MAPPO - IPPO at 400 | at 200")
    for x in arrivals.itertuples():
        early = (f"{x.ippo_200:6.1%} -> {x.ippo_400:6.1%} | {x.mappo_200:6.1%} -> {x.mappo_400:6.1%}"
                 if x.mode == "sampled" else f"{'':9s} {x.ippo_400:6.1%} | {'':9s} {x.mappo_400:6.1%}")
        later = ci(x, "mappo_minus_ippo_200") if x.mode == "sampled" else ""
        print(f"  {x.disruption:10s} {x.mode:7s} {early} | {ci(x, 'mappo_minus_ippo_400')} | {later}")
    p = arrivals[(arrivals.disruption == "none") & (arrivals["mode"] == "sampled")].iloc[0]
    b = arrivals[arrivals.disruption == "breakdowns"].iloc[0]
    g = arrivals[arrivals["mode"] == "greedy"].iloc[0]
    print("Pre-registered reading (D1, no disruptions, sampled):")
    print(f"  E10 primary, MAPPO - IPPO at 400: {ci(p, 'mappo_minus_ippo_400')}")
    print(f"  reading: {reading(p.mappo_minus_ippo_400_lo, p.mappo_minus_ippo_400_hi)}")
    print(f"  (1) at 200: {ci(p, 'mappo_minus_ippo_200')}")
    print(f"  (2) 400 - 200: IPPO {ci(p, 'ippo_400_minus_200')}, MAPPO {ci(p, 'mappo_400_minus_200')}")
    print(f"  (3) vs heuristic ({p.heuristic:.1%}): IPPO {ci(p, 'ippo_400_minus_heuristic')}, "
          f"MAPPO {ci(p, 'mappo_400_minus_heuristic')}")
    print(f"      vs checkpoint ({p.checkpoint:.1%}): IPPO {ci(p, 'ippo_400_minus_checkpoint')}, "
          f"MAPPO {ci(p, 'mappo_400_minus_checkpoint')}")
    print(f"  (4) breakdowns: {ci(b, 'mappo_minus_ippo_400')}")
    m = metrics[metrics.disruption == "none"].set_index("metric")
    print("  (5) MAPPO - IPPO at 400: " + ", ".join(
        f"{c} {m.loc[c, 'mappo_minus_ippo_400']:+.3f} [{m.loc[c, 'lo']:+.3f}, {m.loc[c, 'hi']:+.3f}]"
        for c in ("score", "p90_lateness", "gini_penalty")))
    print(f"  (6) greedy: IPPO {g.ippo_400:.1%}, MAPPO {g.mappo_400:.1%}, MAPPO - IPPO {ci(g, 'mappo_minus_ippo_400')}")
    t = t1.iloc[0]
    print(f"  (7) T1 none, exploratory: dense-trained IPPO {t.ippo_new:.1%} / MAPPO {t.mappo_new:.1%} vs E9 "
          f"{t.ippo_old:.1%} / {t.mappo_old:.1%}; dense - E9: IPPO {ci(t, 'ippo_new_minus_old')}, "
          f"MAPPO {ci(t, 'mappo_new_minus_old')}; dense MAPPO - IPPO {ci(t, 'mappo_minus_ippo_new')}")
    print(f"wrote summary_{{validation,arrivals,metrics,transfer}}.csv and learning_speed_*.csv in {OUT}")


if __name__ == "__main__":
    main()
