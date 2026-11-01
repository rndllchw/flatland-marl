"""Exploratory (decided after seeing the curves): how much faster does MAPPO learn than IPPO on validation?

Reads the validation summaries of both arms over 400 iterations (E6/E7 for 10-200, E9 for 210-400; the runs are
bit-identical up to 200, so they are one run each) and the training logs, and writes results/e9_explore/:
    learning_speed_seeds.csv      per arm x training seed (+ the arm mean): first iteration whose smoothed arrivals
                                  reach 60 / 65 / 67.5%, and mean arrivals over iterations 10-200 and 210-400
    learning_speed_headstart.csv  per MAPPO level (its smoothed mean at 100 / 150 / 200): the first iteration at which
                                  IPPO's smoothed mean reaches it, and the difference in iterations, games, agent steps

Smoothed = trailing mean of 3 consecutive snapshots (snapshot i and the two before it), because single snapshots
move by about 3 points. "Not reached" is empty. 3 training seeds per arm, so no confidence intervals.

    python scripts/explore_learning_speed.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "e9_explore"
ARMS = {"IPPO": (("e6_val_sampled", "e9_ippo_val_sampled"), "e9_ippo_training"),
        "MAPPO": (("e7_val_sampled", "e9_mappo_val_sampled"), "e9_mappo_training")}
THRESHOLDS = (0.60, 0.65, 0.675)
LEVELS_AT = (100, 150, 200)
GAMES_PER_ITERATION = 32


def validation(folders) -> pd.DataFrame:
    """Arrivals per training seed x iteration, from one or more consecutive validation folders."""
    v = pd.concat([pd.read_csv(ROOT / "results" / f / "summary_by_snapshot.csv") for f in folders])
    assert not v.duplicated(["seed", "iteration"]).any(), folders
    return v.pivot(index="iteration", columns="seed", values="completion_rate").sort_index()


def smooth(curve: pd.Series) -> pd.Series:
    return curve.rolling(3).mean()


def first_reaching(curve: pd.Series, level: float):
    """First iteration whose smoothed value reaches `level`, or None."""
    hit = smooth(curve)[lambda c: c >= level - 1e-12]
    return int(hit.index[0]) if len(hit) else None


def agent_steps(training: str, iteration: int) -> float:
    """Train-steps (agents x steps) played up to `iteration`, mean over training seeds."""
    logs = [pd.read_csv(ROOT / "results" / training / f"seed_{s}" / "train_log.csv").set_index("iteration")
            for s in range(3)]
    return sum(log.loc[iteration, "agent_steps_total"] for log in logs) / len(logs)


def learning_speed(arms=ARMS, thresholds=THRESHOLDS, levels_at=LEVELS_AT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The two tables above, for `arms` = {name: (validation folders, training folder)}; the first arm is the one
    that has to catch up (IPPO), the second the one whose levels it chases (MAPPO)."""
    curves = {arm: validation(folders) for arm, (folders, _) in arms.items()}
    rows = []
    for arm, c in curves.items():
        for seed in [*c.columns, "mean"]:
            curve = c.mean(axis=1) if seed == "mean" else c[seed]
            row = {"arm": arm, "seed": seed,
                   **{f"reach_{t:.1%}": first_reaching(curve, t) for t in thresholds},
                   "mean_10_200": curve.loc[10:200].mean(), "mean_210_400": curve.loc[210:400].mean()}
            rows.append(row)
    seeds = pd.DataFrame(rows).astype({f"reach_{t:.1%}": "Int64" for t in thresholds})

    (slow, (_, slow_training)), (fast, (_, fast_training)) = arms.items()
    ippo, mappo = curves[slow].mean(axis=1), curves[fast].mean(axis=1)
    head = []
    for at in levels_at:
        level = smooth(mappo).loc[at]
        reached = first_reaching(ippo, level)
        row = {"mappo_iteration": at, "mappo_level": level, "ippo_iteration": reached}
        if reached is not None:
            row.update(extra_iterations=reached - at, extra_games=(reached - at) * GAMES_PER_ITERATION,
                       extra_agent_steps=agent_steps(slow_training, reached) - agent_steps(fast_training, at))
        head.append(row)
    head = pd.DataFrame(head)  # the extra_* columns are missing if IPPO never reaches any level
    head = head.astype({c: "Int64" for c in ("ippo_iteration", "extra_iterations", "extra_games") if c in head})
    return seeds, head


def main():
    seeds, head = learning_speed()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(OUT / "learning_speed_seeds.csv", index=False)
    head.to_csv(OUT / "learning_speed_headstart.csv", index=False)
    pd.set_option("display.width", 160)
    print("Validation arrivals (smoothed over 3 snapshots): first iteration reaching each level; mean arrivals")
    print(seeds.astype(object).fillna("not reached").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nIterations IPPO needs to reach MAPPO's smoothed level (arm means):")
    print(head.astype(object).fillna("not reached").to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"wrote {OUT / 'learning_speed_seeds.csv'} and {OUT / 'learning_speed_headstart.csv'}")


if __name__ == "__main__":
    main()
