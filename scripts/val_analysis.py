"""Compare training snapshots on validation, with the rule used since E3 (fixed before looking at results).

Input: an evaluate.py output folder whose policies are snapshots named .../seed_<s>/actor_<i>.pt (greedy
`actor:` or sampled `actor-sampled:`). Rule:
    main number  = each training seed's iteration-100 (final) snapshot, averaged over seeds
    secondary    = each seed's best snapshot by validation arrivals (optimistic: chosen on these scenarios)
Differences are paired per scenario; 95% bootstrap CI over scenarios (10,000 resamples, seed 0). With
several evaluation seeds (e.g. sampled actions with --seeds 0 1 2), each scenario's result is first
averaged over them. CIs cover scenario variation only, not training seeds.

    python scripts/val_analysis.py results/e5_val_sampled \\
        --ref checkpoint=results/e3_val_baselines:checkpoint heuristic=results/e3_val_baselines:heuristic \\
        --ref-final "E3 sampled"=results/e3_val_sampled --write-summary

--ref NAME=FOLDER:POLICY        one policy from an evaluate.py folder (e.g. a baseline)
--ref-final NAME=FOLDER         another snapshot folder's final snapshots, averaged over its seeds
--write-summary                 also write FOLDER/summary_by_snapshot.csv (one row per seed x iteration)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["score", "completion_rate", "halts_while_moving", "cancelled_rate"]
FINAL = 100


def load_snapshots(folder: Path) -> pd.DataFrame:
    e = pd.read_csv(folder / "episodes.csv", dtype={"scenario": str})
    found = e.policy.str.extract(r"seed_(\d+)/actor_(\d+)\.pt")
    if found.isna().any().any():
        raise ValueError(f"{folder}: every policy must be a snapshot .../seed_<s>/actor_<i>.pt")
    e["eval_seed"] = e["seed"]
    e["seed"], e["iteration"] = found[0].astype(int), found[1].astype(int)
    return e


def summary(e: pd.DataFrame) -> pd.DataFrame:
    """One row per training seed x iteration: means over all episodes (scenarios x evaluation seeds)."""
    return (e.groupby(["seed", "iteration"])
             .agg(scenarios=("scenario", "nunique"), score=("score", "mean"), completion_rate=("completion_rate", "mean"),
                  cancelled_rate=("cancelled_rate", "mean"), mean_lateness=("mean_lateness", "mean"),
                  halts_while_moving=("halts_while_moving", "mean"), gini_penalty=("gini_penalty", "mean"))
             .reset_index())


def per_scenario(e: pd.DataFrame, picks: dict) -> pd.DataFrame:
    """Per scenario: mean over the picked snapshot of each training seed (and over evaluation seeds)."""
    rows = pd.concat([e[(e.seed == s) & (e.iteration == i)] for s, i in picks.items()])
    return rows.groupby("scenario")[COLUMNS].mean()


def paired_ci(a: pd.DataFrame, b: pd.DataFrame, column: str, resamples=10_000) -> tuple[float, float, float]:
    """Mean of a - b over shared scenarios, with a 95% bootstrap CI over scenarios."""
    d = (a[column] - b.loc[a.index, column]).to_numpy()
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), (resamples, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_ref(spec: str) -> tuple[str, pd.DataFrame]:
    name, rest = spec.split("=", 1)
    folder, policy = rest.split(":", 1)  # the policy name may itself contain ':' (actor-sampled:<path>)
    e = pd.read_csv(Path(folder) / "episodes.csv", dtype={"scenario": str})
    e = e[e.policy == policy]
    if e.empty:
        raise ValueError(f"--ref {spec}: no episodes for policy {policy!r}")
    return name, e.groupby("scenario")[COLUMNS].mean()


def load_ref_final(spec: str) -> tuple[str, pd.DataFrame]:
    name, folder = spec.split("=", 1)
    e = load_snapshots(Path(folder))
    return name, per_scenario(e, {s: FINAL for s in sorted(e.seed.unique())})


def analyse(folder: Path, refs: dict) -> dict:
    """Main and secondary results for `folder`, each with paired CIs against every reference."""
    e = load_snapshots(folder)
    s = summary(e)
    best = s.loc[s.groupby("seed").completion_rate.idxmax()].set_index("seed").iteration.to_dict()
    out = {"summary": s}
    for label, picks in (("final", {sd: FINAL for sd in sorted(e.seed.unique())}), ("best", best)):
        chosen = s.set_index(["seed", "iteration"]).loc[list(picks.items())]
        m = per_scenario(e, picks)
        out[label] = {
            "picks": picks,
            "per_seed": {sd: per_scenario(e, {sd: it})[COLUMNS].mean().to_dict() for sd, it in picks.items()},
            "mean": m[COLUMNS].mean().to_dict(),
            "sd_over_seeds": {"score": float(chosen.score.std()), "completion_rate": float(chosen.completion_rate.std())},
            "vs": {name: {c: paired_ci(m, r, c) for c in ("score", "completion_rate")} for name, r in refs.items()},
        }
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--ref", nargs="*", default=[], help="NAME=FOLDER:POLICY")
    parser.add_argument("--ref-final", nargs="*", default=[], help="NAME=FOLDER (final snapshots)")
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)

    refs = dict(load_ref(r) for r in args.ref) | dict(load_ref_final(r) for r in args.ref_final)
    result = analyse(args.folder, refs)
    s = result["summary"]
    if args.write_summary:
        s.to_csv(args.folder / "summary_by_snapshot.csv", index=False)

    pct = lambda x: f"{x:.0%}"  # noqa: E731
    print("Arrived by iteration:")
    print(s.pivot(index="seed", columns="iteration", values="completion_rate").map(pct).to_string())
    print("Never departed by iteration:")
    print(s.pivot(index="seed", columns="iteration", values="cancelled_rate").map(pct).to_string())
    for label, title in (("final", f"Main: iteration {FINAL}"), ("best", "Secondary: best by validation arrivals (optimistic)")):
        r = result[label]
        print(f"\n{title}, picks {r['picks']}")
        for sd, v in r["per_seed"].items():
            print(f"  seed {sd}: score {v['score']:.3f}  arrived {v['completion_rate']:.0%}  halts {v['halts_while_moving']:.1f}")
        m, sd = r["mean"], r["sd_over_seeds"]
        print(f"  mean of seeds: score {m['score']:.3f} (sd {sd['score']:.3f})  arrived {m['completion_rate']:.0%} "
              f"(sd {sd['completion_rate']:.0%})  halts {m['halts_while_moving']:.1f}")
        for name, cis in r["vs"].items():
            score, arrived = cis["score"], cis["completion_rate"]
            print(f"    vs {name}: score {score[0]:+.2f} [{score[1]:+.2f}, {score[2]:+.2f}]  "
                  f"arrived {arrived[0]:+.0%} [{arrived[1]:+.0%}, {arrived[2]:+.0%}]")
    print()
    for name, r in refs.items():
        print(f"{name}: score {r.score.mean():.3f}  arrived {r.completion_rate.mean():.0%}  halts {r.halts_while_moving.mean():.1f}")


if __name__ == "__main__":
    main()
