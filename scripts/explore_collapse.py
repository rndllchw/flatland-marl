"""Exploratory (E10, decided after seeing the results): what changes when a dense-trained policy collapses?

In 5 of the 6 E10 runs, validation arrivals rose to ~60% and then fell to ~40% within 20-40 iterations, with the
policy entropy in training halving (EXPERIMENTS.md, E10). This replays test set D1, no disruptions, evaluation
seed 0, with every E10 run's actor at iteration 200 and 400 (sampled, as stored in results/e10_test), and records
each step's masked action probabilities next to the drawn action (computing them does not touch the sampling
generator, so every replayed train must match the stored result, or the script stops).

Per train:
    p_depart          mean probability of a move action per step while ready to depart (off the map)
    departure_wait    steps between its earliest departure and entering the map (None = never entered)
    entropy_depart    mean policy entropy (nats) while ready to depart
    entropy_choice    mean entropy on the map at steps with >= 2 legal actions
    stop_share        share of on-map steps at which it chose STOP
    and, for trains that did not arrive, explore_failures' facts and bucket (never_left / stuck / still_driving),
    plus lost = drove at least its shortest route (it went the wrong way)

Writes results/e10_explore/collapse_trains.csv (one row per train) and collapse_summary.csv (per run x iteration),
and prints the summary.

    python scripts/explore_collapse.py
"""
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_failures import STILL, bucket, runnable, train_facts  # noqa: E402
from flatmarl import ROOT  # noqa: E402
from flatmarl.envs import clear_flatland_caches, load_scenario  # noqa: E402
from flatmarl.metrics import episode_metrics  # noqa: E402
from flatmarl.models import OBS_SIZE, masked_policy  # noqa: E402
from flatmarl.policies import make_policy  # noqa: E402

TEST_SET = ROOT / "eval_sets/test/d1_scene1_a15-20_ll2"
STORED = ROOT / "results/e10_test/d1_scene1_a15-20_ll2/none"
RUNS = {f"{arm} s{s}": (arm.lower(), s) for arm in ("IPPO", "MAPPO") for s in range(3)}
ITERATIONS = (200, 400)
SEED = 0
MOVES = (1, 2, 3)  # MOVE_LEFT, MOVE_FORWARD, MOVE_RIGHT
STOP = 4
OUT = ROOT / "results" / "e10_explore"


def name(arm: str, seed: int, iteration: int) -> str:
    return f"actor-sampled:results/e10_{arm}_training/seed_{seed}/actor_{iteration}.pt"


def replay(job) -> list[dict]:
    """One stored E10 episode again (as evaluate.run_episode builds it), with the policy's probabilities per step."""
    label, iteration, policy_name, scenario = job
    policy, obs_builder = make_policy(runnable(policy_name), seed=SEED)
    env = load_scenario(scenario, disruption="none", seed=SEED, obs_builder=obs_builder)
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    positions, states, actions = ({h: [] for h in handles} for _ in range(3))
    depart = {h: [] for h in handles}   # (p_move, entropy) per step while ready to depart
    choice = {h: [] for h in handles}   # entropy per on-map step with >= 2 legal actions
    stops = {h: [] for h in handles}    # chose STOP, per on-map step
    while True:
        for a in env.agents:
            positions[a.handle].append(a.position)
            states[a.handle].append(a.state)
        with torch.no_grad():
            x = torch.from_numpy(np.stack([obs[h] for h in handles]))
            logits, _ = policy(x[:, :OBS_SIZE])
            dist = masked_policy(logits, x[:, OBS_SIZE:])
            probs, entropy = dist.probs, dist.entropy()
            legal = (x[:, OBS_SIZE:] > 0.5).sum(dim=1)
        chosen = policy.act_many(handles, [obs[h] for h in handles])
        for i, h in enumerate(handles):
            act = RailEnvActions(chosen.get(h, 0)).value
            actions[h].append(act)
            state = env.agents[h].state
            if state == TrainState.READY_TO_DEPART:
                depart[h].append((probs[i, list(MOVES)].sum().item(), entropy[i].item()))
            elif state in (TrainState.MOVING, TrainState.STOPPED):
                stops[h].append(act == STOP)
                if legal[i] >= 2:
                    choice[h].append(entropy[i].item())
        obs, _, dones, _ = env.step(chosen)
        if dones["__all__"]:
            break
    for a in env.agents:
        positions[a.handle].append(a.position)
    m = episode_metrics(env)
    distance = env.distance_map.get()
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    rows = []
    for t in m.trains:
        agent, h = env.agents[t.train], t.train
        entered = next((i for i, p in enumerate(positions[h]) if p is not None), None)
        row = {"label": label, "iteration": iteration, "policy": policy_name, "scenario": Path(scenario).stem,
               "seed": SEED, "train": h, "arrived": t.arrived, "penalty": t.penalty, "lateness": t.lateness,
               "p_depart": mean([p for p, _ in depart[h]]), "entropy_depart": mean([e for _, e in depart[h]]),
               "entropy_choice": mean(choice[h]), "stop_share": mean(stops[h]),
               "entered": entered is not None or t.arrived,
               "departure_wait": (entered - 1 - agent.earliest_departure) if entered is not None else None}
        if not t.arrived:
            row.update(train_facts(positions[h], states[h], actions[h], agent.earliest_departure))
            row["shortest_route"] = distance[h, *agent.initial_position, agent.initial_direction]
        rows.append(row)
    clear_flatland_caches()  # flatland's caches would otherwise keep every env alive (~45 MB per episode)
    return rows


def check_against_stored(trains: pd.DataFrame):
    stored = pd.concat(pd.read_csv(STORED / f"{arm}_{it}_sampled" / "trains.csv", dtype={"scenario": str})
                       for arm in ("ippo", "mappo") for it in ITERATIONS)
    stored = stored[stored.seed == SEED]
    keys = ["policy", "scenario", "seed", "train"]
    merged = trains.merge(stored[keys + ["arrived", "penalty"]], on=keys, how="outer", suffixes=("", "_stored"),
                          validate="one_to_one", indicator=True)
    differ = (merged._merge != "both") | (merged.arrived != merged.arrived_stored) | \
             (merged.penalty != merged.penalty_stored)
    if differ.any():
        print(merged[differ].head(10).to_string())
        raise SystemExit(f"replay differs from the stored E10 results for {differ.sum()} trains; nothing written")
    print(f"replay matches the stored E10 results for all {len(merged)} trains")


def summarise(trains: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (label, it), g in trains.groupby(["label", "iteration"], sort=False):
        failed = g[~g.arrived]
        b = failed.bucket
        n = len(g)
        rows.append({"run": label, "iteration": it, "trains": n, "arrived": g.arrived.mean(),
                     "p_depart": g.p_depart.mean(), "entropy_depart": g.entropy_depart.mean(),
                     "departure_wait": g.departure_wait.mean(), "entropy_choice": g.entropy_choice.mean(),
                     "stop_share": g.stop_share.mean(), "lateness_arrived": g[g.arrived].lateness.mean(),
                     **{k: (b == k).sum() / n for k in ("never_left", "stuck", "still_driving")},
                     "lost": failed.lost.sum() / n})
    return pd.DataFrame(rows)


def main():
    scenarios = sorted(TEST_SET.glob("*.pkl"))
    jobs = [(label, it, name(arm, s, it), str(sc)) for label, (arm, s) in RUNS.items() for it in ITERATIONS
            for sc in scenarios]
    print(f"{len(jobs)} episodes, 10 workers")
    rows = []
    with get_context("spawn").Pool(10, maxtasksperchild=50) as pool:
        for i, episode in enumerate(pool.imap(replay, jobs), 1):
            rows += episode
            print(f"\r  {i}/{len(jobs)} done", end="", flush=True)
    print()
    trains = pd.DataFrame(rows)
    check_against_stored(trains)
    failed = ~trains.arrived
    trains["bucket"] = None
    trains.loc[failed, "bucket"] = bucket(trains[failed], STILL)
    trains["lost"] = failed & (trains.driven >= trains.shortest_route)
    summary = summarise(trains)
    OUT.mkdir(parents=True, exist_ok=True)
    trains.to_csv(OUT / "collapse_trains.csv", index=False)
    summary.to_csv(OUT / "collapse_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print("per run and iteration (D1 none, evaluation seed 0; buckets and lost are shares of all trains):")
    print(summary.round(3).to_string(index=False))
    print(f"wrote {OUT / 'collapse_trains.csv'} and {OUT / 'collapse_summary.csv'}")


if __name__ == "__main__":
    sys.exit(main())
