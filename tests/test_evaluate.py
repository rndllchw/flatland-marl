"""Tests for scripts/evaluate.py (and flatmarl.policies).

Run from repo root:  pytest tests/
Uses fast policies on small scenarios; the slow heuristic only runs on the 6-train sample.
"""
import csv
import json
import sys

import numpy as np
import pytest
import torch

from flatmarl import EXAMPLE_CURRICULUM, ROOT, STARTER_KIT
from flatmarl.envs import load_scenario
from flatmarl.metrics import episode_metrics
from flatmarl.models import OBS_SIZE, make_actor, masked_policy, save_actor
from flatmarl.policies import STARTER_KIT_CHECKPOINT, make_policy

sys.path.insert(0, str(ROOT / "scripts"))
import evaluate  # noqa: E402

SAMPLE = STARTER_KIT / "reinforcement_learning" / "sampling" / "level_0_scenario_1.pkl"
SMALL = ["00_scene_1_ll-2_a-1", "03_scene_1_ll-2_a-10", "12_scene_1_ll-3_a-10"]  # 1, 10, 10 trains


@pytest.fixture(scope="module")
def small_set(tmp_path_factory):
    """A folder with three small example-curriculum scenarios."""
    folder = tmp_path_factory.mktemp("scenarios")
    for name in SMALL:
        (folder / f"{name}.pkl").write_bytes((EXAMPLE_CURRICULUM / f"{name}.pkl").read_bytes())
    return folder


def run(out, *args):
    evaluate.main([*map(str, args), "--out", str(out)])
    read = lambda name: list(csv.DictReader((out / name).open()))  # noqa: E731
    return read("episodes.csv"), read("trains.csv"), json.loads((out / "run.json").read_text())


def without_runtime(rows):
    return [{k: v for k, v in r.items() if k != "runtime_s"} for r in rows]


def test_matches_direct_episode_metrics(small_set, tmp_path):
    episodes, _, _ = run(tmp_path, "--scenarios", small_set, "--policies", "forward", "checkpoint")
    assert len(episodes) == 2 * len(SMALL)
    for row in episodes:
        policy, obs_builder = make_policy(row["policy"])
        env = load_scenario(small_set / f"{row['scenario']}.pkl", obs_builder=obs_builder)
        obs, _ = env.reset()
        handles = env.get_agent_handles()
        while True:
            obs, _, dones, _ = env.step(policy.act_many(handles, [obs[h] for h in handles]))
            if dones["__all__"]:
                break
        direct = episode_metrics(env).as_row()
        for key in ("score", "completion_rate", "halts_while_moving", "malfunctions", "penalty_collision"):
            assert float(row[key]) == pytest.approx(float(direct[key])), key


def test_reference_values_on_sample(tmp_path):
    episodes, trains, _ = run(tmp_path, "--scenarios", SAMPLE, "--policies", "checkpoint", "heuristic")
    scores = {r["policy"]: float(r["score"]) for r in episodes}
    assert scores == pytest.approx({"checkpoint": 0.730, "heuristic": 0.513}, abs=1e-3)
    assert {r["policy"]: int(r["halts_while_moving"]) for r in episodes} == {"checkpoint": 0, "heuristic": 4}


def test_parallel_and_repeated_runs_are_identical(small_set, tmp_path):
    args = ("--scenarios", small_set, "--policies", "random", "checkpoint", "--seeds", "0", "1",
            "--disruption", "breakdowns")
    first, first_trains, _ = run(tmp_path / "a", *args)
    again, again_trains, _ = run(tmp_path / "b", *args)
    parallel, parallel_trains, _ = run(tmp_path / "c", *args, "--workers", "2")
    assert without_runtime(first) == without_runtime(again) == without_runtime(parallel)
    assert first_trains == again_trains == parallel_trains


def test_files_consistent(small_set, tmp_path):
    episodes, trains, info = run(tmp_path, "--scenarios", small_set, "--policies", "do_nothing")
    assert len(trains) == sum(int(e["n_trains"]) for e in episodes)
    assert len(info["git"]["commit"]) == 40 and info["args"]["policies"] == ["do_nothing"]
    assert len(info["scenarios"]) == len(SMALL) and info["episodes"] == len(SMALL)
    assert all(float(e["cancelled_rate"]) == 1.0 for e in episodes)   # do_nothing never departs


def test_seeds_and_disruption_options(small_set, tmp_path):
    episodes, _, _ = run(tmp_path / "none", "--scenarios", small_set, "--policies", "forward", "--disruption", "none")
    assert all(int(e["malfunctions"]) + int(e["malfunctions_before_departure"]) == 0 for e in episodes)
    assert {e["seed"] for e in episodes} == {""}                      # no --seeds: file's randomness

    # Each seed effect tested on its own, so one can't hide the other being lost. Compare uncapped
    # penalties: random is so bad every train hits the cap, so its score is 0.0 whatever the seed.
    penalties = lambda e: tuple(v for k, v in e.items() if k.startswith("penalty_"))  # noqa: E731

    def per_seed(out, policy, disruption):
        rows, _, _ = run(out, "--scenarios", small_set, "--policies", policy, "--seeds", "0", "1",
                         "--disruption", disruption)
        assert len(rows) == 2 * len(SMALL) and {e["seed"] for e in rows} == {"0", "1"}
        return [[penalties(e) for e in rows if e["seed"] == s] for s in ("0", "1")]

    seed0, seed1 = per_seed(tmp_path / "breakdown_seeds", "forward", "breakdowns")  # deterministic policy
    assert seed0 != seed1, "seed must change the breakdown pattern"
    seed0, seed1 = per_seed(tmp_path / "policy_seeds", "random", "none")            # no breakdown randomness
    assert seed0 != seed1, "seed must reach the random policy"


def test_actor_prefix_loads_a_checkpoint_file(small_set, tmp_path):
    named, _, _ = run(tmp_path / "named", "--scenarios", small_set, "--policies", "checkpoint")
    by_path, _, _ = run(tmp_path / "path", "--scenarios", small_set, "--policies", f"actor:{STARTER_KIT_CHECKPOINT}")
    strip = lambda rows: [{k: v for k, v in r.items() if k not in ("policy", "runtime_s")} for r in rows]  # noqa: E731
    assert strip(named) == strip(by_path)


def test_errors(small_set, tmp_path):
    with pytest.raises(ValueError, match="Unknown policy"):
        run(tmp_path / "a", "--scenarios", small_set, "--policies", "chekpoint")
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "b", "--scenarios", small_set, "--policies", "actor:no/such/file.pt")
    with pytest.raises(FileNotFoundError):
        run(tmp_path / "b2", "--scenarios", small_set, "--policies", "actor-sampled:no/such/file.pt")
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no .pkl"):
        run(tmp_path / "c", "--scenarios", tmp_path / "empty", "--policies", "forward")
    run(tmp_path / "d", "--scenarios", small_set, "--policies", "do_nothing")
    with pytest.raises(SystemExit, match="overwrite"):                 # never silently replace results
        run(tmp_path / "d", "--scenarios", small_set, "--policies", "do_nothing")


# actor-sampled:<path> (26 Oct)

@pytest.fixture(scope="module")
def untrained_actor(tmp_path_factory):
    """A freshly initialised actor: its policy spreads probability over the legal actions."""
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("actor") / "actor.pt"
    save_actor(make_actor(), path)
    return path


def test_sampled_actor_draws_actions_with_the_policys_probabilities(untrained_actor):
    env = load_scenario(SAMPLE, obs_builder=make_policy("checkpoint")[1])
    obs, _ = env.reset()
    for _ in range(5):                                       # until some train has a real choice
        on_map = [o for o in obs.values() if o[OBS_SIZE:].sum() > 1.5]
        if on_map:
            break
        obs, _, _, _ = env.step({h: 2 for h in env.get_agent_handles()})
    x = torch.from_numpy(on_map[0])
    with torch.no_grad():
        probs = masked_policy(make_actor(untrained_actor)(x[:OBS_SIZE])[0], x[OBS_SIZE:]).probs.numpy()
    policy, _ = make_policy(f"actor-sampled:{untrained_actor}", seed=0)
    draws = np.array([policy.act(on_map[0]) for _ in range(4000)])
    freq = np.bincount(draws, minlength=5) / len(draws)
    assert (freq[on_map[0][OBS_SIZE:] < 0.5] == 0).all()     # never an illegal action
    assert freq == pytest.approx(probs, abs=0.03)
    again, _ = make_policy(f"actor-sampled:{untrained_actor}", seed=0)
    other, _ = make_policy(f"actor-sampled:{untrained_actor}", seed=1)
    first = draws[:50].tolist()
    assert [again.act(on_map[0]) for _ in range(50)] == first          # same seed, same draws
    assert [other.act(on_map[0]) for _ in range(50)] != first          # different seed, different draws


def test_sampled_evaluation_is_reproducible_and_seeded(small_set, untrained_actor, tmp_path):
    args = ("--scenarios", small_set, "--policies", f"actor-sampled:{untrained_actor}", "--seeds", "0", "1",
            "--disruption", "none")
    first, first_trains, _ = run(tmp_path / "a", *args)
    parallel, parallel_trains, _ = run(tmp_path / "b", *args, "--workers", "2")
    assert without_runtime(first) == without_runtime(parallel) and first_trains == parallel_trains
    penalties = lambda e: tuple(v for k, v in e.items() if k.startswith("penalty_"))  # noqa: E731
    assert [penalties(e) for e in first if e["seed"] == "0"] != [penalties(e) for e in first if e["seed"] == "1"]


def test_seeds_without_disruptions_change_nothing_for_a_top_action_policy(small_set, tmp_path):
    # So in sampled evaluation with --disruption none, --seeds varies only the action draws.
    rows, _, _ = run(tmp_path, "--scenarios", small_set, "--policies", "checkpoint", "--seeds", "0", "1",
                     "--disruption", "none")
    strip = lambda e: {k: v for k, v in e.items() if k not in ("seed", "runtime_s")}  # noqa: E731
    assert [strip(e) for e in rows if e["seed"] == "0"] == [strip(e) for e in rows if e["seed"] == "1"]
