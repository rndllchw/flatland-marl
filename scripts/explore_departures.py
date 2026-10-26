"""Exploratory (E8 exploratory 1): why does MAPPO survive top-action (greedy) execution while IPPO collapses?

For each final network (checkpoint, E6 IPPO and E7 MAPPO seeds 0-2) on the first 20 validation scenarios (not
the test sets), no disruptions:
  1. sampled play (as in training): action probabilities at every recorded decision, split into
     ready-to-depart (only do-nothing and forward legal) and on the map;
  2. greedy play (top action): steps each train waits between becoming ready to depart and first moving.
Writes results/e8_explore/departures.csv (one row per network) and prints it.

    python scripts/explore_departures.py
"""
import functools
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from flatland.envs.step_utils.states import TrainState

from flatmarl import ROOT
from flatmarl.envs import clear_flatland_caches, load_scenario
from flatmarl.models import OBS_SIZE, make_actor, masked_policy
from flatmarl.policies import STARTER_KIT_CHECKPOINT, make_policy
from flatmarl.rollout import collect_episode

NETS = {"checkpoint": STARTER_KIT_CHECKPOINT,
        **{f"IPPO s{s}": ROOT / f"results/e6_training/seed_{s}/actor_200.pt" for s in range(3)},
        **{f"MAPPO s{s}": ROOT / f"results/e7_training/seed_{s}/actor_200.pt" for s in range(3)}}
SCENARIOS = sorted((ROOT / "pools/val_scene1_a5-10_ll2").glob("*.pkl"))[:20]
READY_MASK = np.array([1, 0, 1, 0, 0])  # do-nothing and forward only: a train waiting at its start
OUT = ROOT / "results" / "e8_explore"


def sampled_probabilities(args):
    name, path, scenario = args
    actor = make_actor(path)
    state = {k: v.clone() for k, v in actor.state_dict().items()}
    env_fn = functools.partial(load_scenario, scenario, disruption="none")
    result = collect_episode((env_fn, state, 0, {"scale": 0.01}, 0.99))
    obs = np.concatenate([s.obs for s in result.segments])
    x = torch.from_numpy(obs)
    with torch.no_grad():
        probs = masked_policy(actor(x[:, :OBS_SIZE])[0], x[:, OBS_SIZE:]).probs.numpy()
    ready = (obs[:, OBS_SIZE:] == READY_MASK).all(axis=1)
    return name, probs[ready], probs[~ready], result.metrics["completion_rate"]


def greedy_waits(args):
    name, path, scenario = args
    policy, obs_builder = make_policy(f"actor:{path}")
    env = load_scenario(scenario, disruption="none", obs_builder=obs_builder)
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    ready_at, left_at = {}, {}
    while True:
        for a in env.agents:
            if a.state == TrainState.READY_TO_DEPART and a.handle not in ready_at:
                ready_at[a.handle] = env._elapsed_steps
            if a.state == TrainState.MOVING and a.handle not in left_at:
                left_at[a.handle] = env._elapsed_steps
        obs, _, dones, _ = env.step(policy.act_many(handles, [obs[h] for h in handles]))
        if dones["__all__"]:
            break
    clear_flatland_caches()
    return name, [left_at[t] - ready_at[t] if t in left_at else None for t in ready_at]


def main():
    torch.set_num_threads(1)
    jobs = [(name, path, sc) for name, path in NETS.items() for sc in SCENARIOS]
    with get_context("spawn").Pool(10, maxtasksperchild=20) as pool:
        sampled, greedy = pool.map(sampled_probabilities, jobs), pool.map(greedy_waits, jobs)
    rows = []
    for name in NETS:
        ready = np.concatenate([o[1] for o in sampled if o[0] == name])
        on_map = np.concatenate([o[2] for o in sampled if o[0] == name])
        waits = [w for o in greedy if o[0] == name for w in o[1]]
        left = np.array([w for w in waits if w is not None])
        go, top = ready[:, 2], on_map.max(axis=1)
        rows.append({"network": name, "ready_decisions": len(ready), "mean_p_go": go.mean(),
                     "share_top_action_go": np.mean(go > 0.5), "share_p_go_35_65": np.mean((go > .35) & (go < .65)),
                     "on_map_decisions": len(on_map), "on_map_mean_top_prob": top.mean(),
                     "sampled_arrived": np.mean([o[3] for o in sampled if o[0] == name]),
                     "greedy_trains": len(waits), "greedy_never_left": sum(w is None for w in waits) / len(waits),
                     "greedy_median_wait": float(np.median(left)), "greedy_mean_wait": left.mean(),
                     "greedy_share_wait_over_20": np.mean(left > 20)})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "departures.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"wrote {OUT / 'departures.csv'}")


if __name__ == "__main__":
    sys.exit(main())
