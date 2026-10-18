"""Tests for flatmarl.envs. Each test pins down one behaviour the project relies on.

Run from repo root:  pytest tests/
Episodes use an always-FORWARD policy: we test the environment, not a policy.
"""
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


def test_load_scenario_reset_rejects_arguments():
    # A replay env ignoring reset(random_seed=...) silently would be misleading; it must fail loudly.
    env = load_scenario(SAMPLE)
    with pytest.raises(TypeError, match="seed"):
        env.reset(random_seed=99)
