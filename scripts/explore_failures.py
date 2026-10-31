"""Exploratory (E8 exploratory 2, not pre-registered): how do the trains that miss their destination fail?

Replays test set T1 (scene 1, 5-10 trains), no disruptions, with the E8 policies: the IPPO (E6) and MAPPO (E7)
final networks and the starter-kit checkpoint with sampled actions, evaluation seeds 0-2, and the heuristic with
seed 0 only (without breakdowns its three seeds are identical). Every step it records each train's cell, state and
chosen action. Every replayed train must match the stored E8 result (arrived, penalty) or the script stops.

Each train that does not arrive goes in one bucket, by where it is at the time limit:
    never_left     still at its start (never entered the map)
    stuck          same cell for the last >= STILL steps
    still_driving  moved within the last STILL steps but ran out of time
and is tagged left_late if it entered the map more than LATE steps after its earliest departure.
Main thresholds STILL = 50, LATE = 20; the summary repeats the counts for every pair in 25/50/100 x 10/20/40.
Stuck is not split into "blocked" and "chose to stop": in a spot check the same two trains sat in the same jam
for ~240 steps under the checkpoint (asking to move ~90% of steps) and MAPPO s1 (0%), so whether a stuck train
keeps asking to move says more about the policy than about the jam. wanted_move_share stays in the CSV as a detail.
Route numbers (shortest_route, driven, revisited_cells, remaining_at_end) tell lost trains (drove at least their
shortest route, so went the wrong way) from trains that ran out of time (drove less than their shortest route).

Writes to results/e8_explore/:
    failures_trains.csv   one row per train that did not arrive: raw numbers + buckets at the main thresholds
    failures_summary.csv  one row per policy (and arm = 3 seeds pooled) and threshold pair: shares of ALL trains
and prints the main summary and two video candidates.

    python scripts/explore_failures.py
"""
import sys
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import pandas as pd
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from flatmarl import ROOT
from flatmarl.envs import clear_flatland_caches, load_scenario
from flatmarl.metrics import episode_metrics
from flatmarl.policies import make_policy

TEST_SET = ROOT / "eval_sets/test/t1_scene1_a5-10_ll2"
STORED = ROOT / "results/e8_test/t1_scene1_a5-10_ll2/none"
CK = "third_party/ecml2026-starterkit/reinforcement_learning/checkpoint.pt"
# label -> (policy name as stored in E8, folder it is stored in, evaluation seeds)
POLICIES = {**{f"IPPO s{s}": (f"actor-sampled:results/e6_training/seed_{s}/actor_200.pt", "e6_sampled", (0, 1, 2))
               for s in range(3)},
            **{f"MAPPO s{s}": (f"actor-sampled:results/e7_training/seed_{s}/actor_200.pt", "e7_sampled", (0, 1, 2))
               for s in range(3)},
            "checkpoint": (f"actor-sampled:{CK}", "baselines", (0, 1, 2)),
            "heuristic": ("heuristic", "baselines", (0,))}
ARMS = {"IPPO": [f"IPPO s{s}" for s in range(3)], "MAPPO": [f"MAPPO s{s}" for s in range(3)]}
STILL, LATE = 50, 20
STILL_GRID, LATE_GRID = (25, 50, 100), (10, 20, 40)
BUCKETS = ("never_left", "stuck", "still_driving")
MOVES = {1, 2, 3}  # RailEnvActions MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT
OUT = ROOT / "results" / "e8_explore"


def runnable(name: str) -> str:
    """Stored names use repo-relative paths (E8 ran from the repo root); make them work from anywhere."""
    kind, sep, path = name.partition(":")
    return f"{kind}:{ROOT / path}" if sep else name


def train_facts(positions, states, actions, earliest_departure) -> dict:
    """Raw numbers for one train that did not arrive. positions[t] is its cell before step t (None = off the map)
    plus, last, its cell after the final step; states[t] and actions[t] are its state before step t and the action
    chosen at step t.
        departure_wait     steps between its earliest departure and entering the map (0 = on time)
        still_steps        final steps in which it did not change cell
        wanted_move_share  share of those steps in which it asked to move (a move action, or do-nothing while moving)
        driven             cells moved on the map
        revisited_cells    moves into a cell it had already been in (> 0: it went round a loop or turned back)
    """
    entered = next((t for t, p in enumerate(positions) if p is not None), None)
    if entered is None:
        return {"departure_wait": None, "still_steps": None, "wanted_move_share": None, "driven": None,
                "revisited_cells": None}
    assert positions[-1] is not None, "only trains that did not arrive (arrived trains leave the map)"
    still = 0
    while still < len(actions) and positions[-2 - still] == positions[-1]:
        still += 1
    tail = range(len(actions) - still, len(actions))
    wanted = [actions[t] in MOVES or (actions[t] == 0 and states[t] == TrainState.MOVING) for t in tail]
    on_map = positions[entered:]
    driven = sum(a != b for a, b in zip(on_map, on_map[1:]))
    return {"departure_wait": entered - 1 - earliest_departure,  # a train leaving on time is first seen 1 step later
            "still_steps": still, "wanted_move_share": sum(wanted) / still if still else None,
            "driven": driven, "revisited_cells": driven - (len(set(on_map)) - 1)}


def bucket(failed: pd.DataFrame, still: int) -> pd.Series:
    """Bucket of each failing train (rows with train_facts columns) at stillness threshold `still`."""
    b = pd.Series("still_driving", index=failed.index)
    b[failed.still_steps >= still] = "stuck"
    b[failed.departure_wait.isna()] = "never_left"
    return b


def replay(job) -> list[dict]:
    """One E8 episode again (built exactly as evaluate.run_episode builds it), recording every step -> one row per train."""
    label, name, scenario, seed = job
    policy, obs_builder = make_policy(runnable(name), seed=seed)
    env = load_scenario(scenario, disruption="none", seed=seed, obs_builder=obs_builder)
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    positions, states, actions = ({h: [] for h in handles} for _ in range(3))
    while True:
        for a in env.agents:
            positions[a.handle].append(a.position)
            states[a.handle].append(a.state)
        chosen = policy.act_many(handles, [obs[h] for h in handles])
        for h in handles:  # policies return ints or RailEnvActions; flatland treats a missing one as do-nothing
            actions[h].append(RailEnvActions(chosen.get(h, 0)).value)
        obs, _, dones, _ = env.step(chosen)
        if dones["__all__"]:
            break
    for a in env.agents:
        positions[a.handle].append(a.position)
    m = episode_metrics(env)
    distance = env.distance_map.get()  # [train, row, column, direction]: shortest route length to its destination
    rows = []
    for t in m.trains:
        agent = env.agents[t.train]
        row = {"label": label, "policy": name, "scenario": Path(scenario).stem, "seed": seed, "train": t.train,
               "arrived": t.arrived, "penalty": t.penalty, "halts_while_moving": t.halts_while_moving,
               "earliest_departure": agent.earliest_departure}
        if not t.arrived:
            row.update(train_facts(positions[t.train], states[t.train], actions[t.train], agent.earliest_departure))
            row["shortest_route"] = distance[t.train, *agent.initial_position, agent.initial_direction]
            row["remaining_at_end"] = (distance[t.train, *agent.position, agent.direction]
                                       if agent.position is not None else row["shortest_route"])
        rows.append(row)
    clear_flatland_caches()  # flatland's caches would otherwise keep every env alive (~45 MB per episode)
    return rows


def check_against_stored(trains: pd.DataFrame):
    """Stop unless the replay reproduces every stored E8 train exactly (and no stored train is missing)."""
    folders = dict.fromkeys(folder for _, folder, _ in POLICIES.values())
    stored = pd.concat(pd.read_csv(STORED / f / "trains.csv", dtype={"scenario": str}) for f in folders)
    wanted = {(name, seed) for name, _, seeds in POLICIES.values() for seed in seeds}
    stored = stored[[(p, s) in wanted for p, s in zip(stored.policy, stored.seed)]]
    keys = ["policy", "scenario", "seed", "train"]
    merged = trains.merge(stored[keys + ["arrived", "penalty"]], on=keys, how="outer", suffixes=("", "_stored"),
                          validate="one_to_one", indicator=True)
    differ = (merged._merge != "both") | (merged.arrived != merged.arrived_stored) | \
             (merged.penalty != merged.penalty_stored)
    if differ.any():
        print(merged[differ].head(10).to_string())
        raise SystemExit(f"replay differs from the stored E8 results for {differ.sum()} trains; nothing written")
    print(f"replay matches the stored E8 results for all {len(merged)} trains")


def summarise(trains: pd.DataFrame) -> pd.DataFrame:
    """Shares of ALL trains in each bucket, per policy and per arm (seeds pooled), for every threshold pair."""
    groups = {**{label: [label] for label in POLICIES}, **ARMS}
    failed = trains[~trains.arrived]
    rows = []
    for still, late in product(STILL_GRID, LATE_GRID):
        b = bucket(failed, still)
        for group, labels in groups.items():
            n = trains.label.isin(labels).sum()
            mine = failed.label.isin(labels)
            rows.append({"policy": group, "still_steps": still, "late_steps": late, "trains": n,
                         "not_arrived": mine.sum() / n, **{k: ((b == k) & mine).sum() / n for k in BUCKETS},
                         "not_arrived_left_late": (mine & (failed.departure_wait > late)).sum() / n})
    return pd.DataFrame(rows)


def video_candidates(trains: pd.DataFrame, failed: pd.DataFrame):
    episodes = trains.groupby(["label", "scenario", "seed"]).agg(trains=("train", "size"), arrived=("arrived", "sum"))
    mappo = episodes[episodes.index.get_level_values("label").isin(ARMS["MAPPO"])]
    mappo = mappo.assign(share=mappo.arrived / mappo.trains).sort_values(["share", "trains"], ascending=False)
    ippo = failed[failed.label.isin(ARMS["IPPO"]) & (failed.bucket == "stuck")]
    stuck = ippo.groupby(["label", "scenario", "seed"]).size().sort_values(ascending=False)
    print("\nvideo candidates (label, scenario, seed):")
    print(f"  MAPPO success: {mappo.index[0]}  {mappo.arrived.iloc[0]}/{mappo.trains.iloc[0]} trains arrived")
    if len(stuck):
        key = stuck.index[0]
        print(f"  IPPO stuck:    {key}  {stuck.iloc[0]} trains stuck, "
              f"{episodes.loc[key, 'arrived']}/{episodes.loc[key, 'trains']} arrived")


def main():
    scenarios = sorted(TEST_SET.glob("*.pkl"))
    jobs = [(label, name, str(sc), seed) for label, (name, _, seeds) in POLICIES.items()
            for sc in scenarios for seed in seeds]
    print(f"{len(jobs)} episodes, 10 workers")
    rows = []
    with get_context("spawn").Pool(10, maxtasksperchild=50) as pool:
        for i, episode in enumerate(pool.imap(replay, jobs), 1):
            rows += episode
            print(f"\r  {i}/{len(jobs)} done", end="", flush=True)
    print()
    trains = pd.DataFrame(rows)
    check_against_stored(trains)

    failed = trains[~trains.arrived].copy()
    failed["bucket"] = bucket(failed, STILL)
    failed["left_late"] = failed.departure_wait > LATE
    failed["departure_wait"] = failed.departure_wait.astype("Int64")
    failed["still_steps"] = failed.still_steps.astype("Int64")
    summary = summarise(trains)
    OUT.mkdir(parents=True, exist_ok=True)
    failed.drop(columns=["arrived", "penalty"]).to_csv(OUT / "failures_trains.csv", index=False)
    summary.to_csv(OUT / "failures_summary.csv", index=False)

    main_table = summary[(summary.still_steps == STILL) & (summary.late_steps == LATE)]
    print(f"\nshares of all trains, still >= {STILL} steps, left late > {LATE} steps (exploratory):")
    print(main_table.drop(columns=["still_steps", "late_steps"]).round(3).to_string(index=False))
    video_candidates(trains, failed)
    print(f"\nwrote {OUT / 'failures_trains.csv'} ({len(failed)} rows), {OUT / 'failures_summary.csv'} ({len(summary)} rows)")


if __name__ == "__main__":
    sys.exit(main())
