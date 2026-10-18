"""Tests for flatmarl.envs. Each test pins down one behaviour the project relies on.

Run from repo root:  pytest tests/
Episodes use an always-FORWARD policy: we test the environment, not a policy.
"""
import functools
import multiprocessing
import pickle

import numpy as np
import pytest
from flatland.envs.rewards import ECML2026Rewards

from flatmarl import STARTER_KIT
from flatmarl.envs import load_scenario, make_env

SAMPLE = STARTER_KIT / "reinforcement_learning" / "sampling" / "level_0_scenario_1.pkl"
FORWARD = 2


def routes(env):
    return tuple((a.initial_position, a.initial_direction, a.target, a.earliest_departure, a.latest_arrival)
                 for a in env.agents)


def play(env):
    """Reset and run one episode; return everything needed to compare two episodes."""
    env.reset()
    handles = env.get_agent_handles()
    totals = {h: 0.0 for h in handles}
    prev = {h: env.agents[h].state for h in handles}
    breakdowns = 0
    while True:
        _, rewards, dones, _ = env.step({h: FORWARD for h in handles})
        for h in handles:
            totals[h] += rewards[h]
            state = env.agents[h].state
            breakdowns += state.is_malfunction_state() and not prev[h].is_malfunction_state()
            prev[h] = state
        if dones["__all__"]:
            break
    score = env.rewards.normalize(*totals.values(), num_agents=len(handles), max_episode_steps=env._max_episode_steps)
    return {"routes": routes(env), "score": round(float(score), 6), "breakdowns": breakdowns}


def test_none_means_no_disruptions():
    # The competition map file carries its own breakdowns and stop delays; "none" must remove them.
    assert play(make_env(n_agents=10, disruption="none", seed=0))["breakdowns"] == 0


def test_breakdown_preset_causes_breakdowns():
    # ~10 trains x ~600 steps x 1/540 => roughly 10 expected; seeded, so deterministic.
    assert play(make_env(n_agents=10, disruption="breakdowns", seed=0))["breakdowns"] > 0


def test_same_seed_is_reproducible():
    a = play(make_env(n_agents=10, disruption="breakdowns", seed=3))
    b = play(make_env(n_agents=10, disruption="breakdowns", seed=3))
    assert a == b


def test_different_seeds_differ():
    a = make_env(n_agents=10, seed=1); a.reset()
    b = make_env(n_agents=10, seed=2); b.reset()
    assert routes(a) != routes(b)


def test_each_reset_samples_new_scenario():
    env = make_env(n_agents=10, seed=0)
    env.reset(); first = routes(env)
    env.reset(); second = routes(env)
    assert first != second


@pytest.mark.parametrize("disruption", [None, "none", "breakdowns"])
def test_replay_is_exact(disruption):
    # Replaying a frozen scenario on the SAME env object must be identical every time,
    # breakdowns included (flatland's generators otherwise carry state across episodes).
    env = load_scenario(SAMPLE, disruption=disruption, seed=5)
    first, second = play(env), play(env)
    assert first == second
    if disruption == "breakdowns":
        assert first["breakdowns"] > 0


def test_n_agents_range():
    counts = []
    env = make_env(n_agents=(2, 6), seed=0)
    for _ in range(10):
        env.reset()
        counts.append(env.get_num_agents())
    assert all(2 <= n <= 6 for n in counts)
    assert len(set(counts)) > 1

    again = make_env(n_agents=(2, 6), seed=0)
    counts_again = []
    for _ in range(10):
        again.reset()
        counts_again.append(again.get_num_agents())
    assert counts == counts_again


def test_official_rewards_always():
    assert isinstance(make_env().rewards, ECML2026Rewards)
    assert isinstance(load_scenario(SAMPLE).rewards, ECML2026Rewards)


@pytest.mark.parametrize("kwargs", [
    {"scene": "scene_9"},
    {"line_length": 1},
    {"disruption": "earthquake"},
    {"n_agents": (5, 2)},
])
def test_invalid_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        make_env(**kwargs)


def test_no_step_recording():
    # flatland's loader turns on per-step recording, and its action log is never cleared:
    # in long training runs it would grow without limit.
    for env in (make_env(n_agents=3, seed=0), load_scenario(SAMPLE)):
        play(env)
        play(env)
        assert len(env.list_actions) == 0


def test_numpy_integer_n_agents():
    # e.g. `for n in np.arange(5, 30, 5): make_env(n_agents=n)`
    env = make_env(n_agents=np.int64(4), seed=0)
    env.reset()
    assert env.get_num_agents() == 4


def play_heuristic(env, stop_at_stations):
    """Deadlock-avoidance heuristic; optionally forced to STOP once on each required station cell."""
    from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    policy = DeadLockAvoidancePolicy()
    served = {h: set() for h in handles}
    totals = {h: 0.0 for h in handles}
    while True:
        actions = policy.act_many(handles, [obs[h] for h in handles])
        if stop_at_stations:
            for h in handles:
                a = env.agents[h]
                if a.position in {wp.position for wps in a.waypoints[1:-1] for wp in wps} - served[h]:
                    actions[h] = 4  # STOP
                    served[h].add(a.position)
        obs, rewards, dones, _ = env.step(actions)
        for h in handles:
            totals[h] += rewards[h]
        if dones["__all__"]:
            return totals


def test_reward_memory_does_not_leak_between_episodes():
    # Flatland's rewards object remembers station visits forever. Without a fresh one per reset,
    # a never-stopping policy got credit for stops served in the PREVIOUS episode (0.620 vs 0.513).
    from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
    fresh = play_heuristic(load_scenario(SAMPLE, disruption="none", obs_builder=FullEnvObservation()), False)
    env = load_scenario(SAMPLE, disruption="none", obs_builder=FullEnvObservation())
    play_heuristic(env, stop_at_stations=True)
    assert play_heuristic(env, stop_at_stations=False) == fresh


def test_make_env_fresh_rewards_each_reset():
    env = make_env(n_agents=2, seed=0)
    env.reset()
    first = env.rewards
    env.reset()
    assert env.rewards is not first and isinstance(env.rewards, ECML2026Rewards)


def test_load_scenario_reset_rejects_arguments():
    # A replay env ignoring reset(random_seed=...) silently would be misleading; it must fail loudly.
    env = load_scenario(SAMPLE)
    with pytest.raises(TypeError, match="seed"):
        env.reset(random_seed=99)


def _build_and_play(recipe):
    """Runs in a separate process: build the env from its recipe and play one episode."""
    return play(recipe())


def test_env_recipe_works_in_another_process():
    # Envs themselves can't be pickled (the starter kit's route generator and our reset wrappers are
    # inner functions), so parallel workers must be sent a RECIPE and build their own env.
    # "spawn" is the macOS default and the strictest: everything sent must be picklable.
    recipe = functools.partial(make_env, scene="scene_1", n_agents=5, disruption="breakdowns", seed=3)
    pickle.dumps(recipe)
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        in_worker = pool.apply(_build_and_play, (recipe,))
    assert in_worker == play(recipe())
