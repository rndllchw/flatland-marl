"""Tests for scenario pools: scripts/make_scenario_pool.py and flatmarl.envs.ScenarioPoolEnv."""
import functools
import multiprocessing
import sys

import pytest

from flatmarl import ROOT
from flatmarl.envs import load_scenario, make_env, make_pool_env
from flatmarl.metrics import episode_metrics
from test_envs import play, routes

sys.path.insert(0, str(ROOT / "scripts"))
import make_scenario_pool  # noqa: E402

SPEC = ["--scene", "scene_1", "--n-agents", "3", "--line-length", "2", "--seed", "100"]


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    out = tmp_path_factory.mktemp("pool")
    make_scenario_pool.main(["--out", str(out), "--n", "3", *SPEC])
    return out


def test_pool_files_are_the_make_env_scenarios(pool):
    files = sorted(pool.glob("*.pkl"))
    assert [f.name for f in files] == ["0.pkl", "1.pkl", "2.pkl"] and (pool / "pool.json").exists()
    for i, f in enumerate(files):
        original = make_env(scene="scene_1", n_agents=3, line_length=2, seed=100 + i)
        original.reset()
        saved = load_scenario(f)
        saved.reset()
        assert routes(saved) == routes(original)


def test_rerun_resumes_and_rejects_a_different_spec(pool):
    before = {f: f.stat().st_mtime_ns for f in pool.glob("*.pkl")}
    make_scenario_pool.main(["--out", str(pool), "--n", "3", *SPEC])          # nothing to build
    assert {f: f.stat().st_mtime_ns for f in pool.glob("*.pkl")} == before
    with pytest.raises(SystemExit, match="was built with"):
        make_scenario_pool.main(["--out", str(pool), "--n", "3", "--scene", "scene_4", "--n-agents", "3",
                                 "--line-length", "2", "--seed", "100"])


def sequence(env, n=6):
    """Scenario names and episode results over n resets."""
    out = []
    for _ in range(n):
        result = play(env)                      # play() resets, so env.scenario is this episode's file
        out.append((env.scenario.name, result["score"], result["breakdowns"]))
    return out


def test_pool_env_is_reproducible_and_seed_dependent(pool):
    a = sequence(make_pool_env(pool, seed=1))
    assert a == sequence(make_pool_env(pool, seed=1))
    assert a != sequence(make_pool_env(pool, seed=2))
    assert {name for name, _, _ in a} <= {"0.pkl", "1.pkl", "2.pkl"}


def test_pool_env_replays_the_chosen_file(pool):
    env = make_pool_env(pool, seed=0)
    env.reset()
    reference = load_scenario(env.scenario)
    reference.reset()
    assert routes(env) == routes(reference)


def test_pool_env_disruption_setting(pool):
    assert all(b == 0 for _, _, b in sequence(make_pool_env(pool, disruption="none", seed=0)))
    assert sum(b for _, _, b in sequence(make_pool_env(pool, disruption="breakdowns", seed=0), n=8)) > 0
    with pytest.raises(ValueError):
        make_pool_env(pool, disruption="earthquake")


def test_pool_env_works_with_metrics_and_errors_clearly(pool):
    env = make_pool_env(pool, seed=0)
    with pytest.raises(AttributeError, match="before the first reset"):
        env.agents
    play(env)
    assert episode_metrics(env).n_trains == 3
    with pytest.raises(ValueError, match="no .pkl"):
        make_pool_env(pool / "missing")


def _play_in_worker(recipe):
    return sequence(recipe(), n=2)


def test_pool_env_recipe_works_in_another_process(pool):
    recipe = functools.partial(make_pool_env, pool=pool, disruption="breakdowns", seed=4)
    with multiprocessing.get_context("spawn").Pool(1) as workers:
        assert workers.apply(_play_in_worker, (recipe,)) == sequence(recipe(), n=2)
