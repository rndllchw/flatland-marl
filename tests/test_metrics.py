"""Tests for flatmarl.metrics. Reference numbers are the hand analysis of the starter-kit sample
scenario (6 trains, file's own disruptions), recorded on 2026-10-17.

Run from repo root:  pytest tests/
"""
import math
import pickle

import pytest
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards
from flatland.envs.step_utils.states import TrainState

from flatmarl import STARTER_KIT
from flatmarl.envs import _default_obs_builder, load_scenario, make_env
from flatmarl.metrics import PENALTY_TYPES, episode_metrics, gini
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from reinforcement_learning.my_policy import ActorCritic
from test_envs import play_heuristic

SAMPLE = STARTER_KIT / "reinforcement_learning" / "sampling" / "level_0_scenario_1.pkl"
CHECKPOINT = STARTER_KIT / "reinforcement_learning" / "checkpoint.pt"
POLICIES = {
    "checkpoint": (lambda: ActorCritic(checkpoint_path=str(CHECKPOINT)), lambda: None),
    "heuristic": (DeadLockAvoidancePolicy, FullEnvObservation),
}


def run(env, policy):
    """Play one episode; return the per-train reward totals the env actually handed out."""
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    totals = {h: 0.0 for h in handles}
    while True:
        obs, rewards, dones, _ = env.step(policy.act_many(handles, [obs[h] for h in handles]))
        for h in handles:
            totals[h] += rewards[h]
        if dones["__all__"]:
            return totals


@pytest.fixture(scope="module", params=list(POLICIES))
def finished(request):
    """(policy name, finished env, env-returned totals) for each baseline on the sample scenario."""
    make_policy, make_obs = POLICIES[request.param]
    env = load_scenario(SAMPLE, disruption=None, obs_builder=make_obs())
    totals = run(env, make_policy())
    return request.param, env, totals


# ---- 1. Totals unchanged ----

def test_penalties_match_rewards_env_returned(finished):
    _, env, totals = finished
    m = episode_metrics(env)
    for t in m.trains:
        assert t.penalty == totals[t.train]
        assert sum(t.penalty_by_type.values()) == t.penalty
        assert set(t.penalty_by_type) == set(PENALTY_TYPES)


@pytest.mark.parametrize("name", list(POLICIES))
def test_recording_rewards_identical_to_raw_flatland(name):
    # Same scenario through plain flatland + plain ECML2026Rewards (how the reference was first recorded).
    make_policy, make_obs = POLICIES[name]
    raw_obs = make_obs() or _default_obs_builder()   # plain flatland would otherwise use a dummy builder
    raw = RailEnvPersister.load_new(str(SAMPLE), obs_builder=raw_obs, rewards=ECML2026Rewards())[0]
    raw.reset = lambda reset=raw.reset: reset(regenerate_rail=False, regenerate_schedule=False)
    ours = load_scenario(SAMPLE, disruption=None, obs_builder=make_obs())
    assert run(ours, make_policy()) == run(raw, make_policy())


class ConstantPolicy:
    def __init__(self, action):
        self.action = action

    def act_many(self, handles, observations):
        return {h: self.action for h in handles}


def test_unfinished_journeys_recorded():
    # Penalties for trains that never arrive are only handed out at episode end; the baselines
    # get every train home on the sample scenario, so they never exercise this path.
    env = load_scenario(SAMPLE, disruption="none")
    totals = run(env, ConstantPolicy(0))                 # do-nothing: no train ever departs
    m = episode_metrics(env)
    assert m.cancelled_rate == 1.0 and m.completion_rate == 0.0
    assert all(t.lateness is None for t in m.trains)
    assert [t.penalty for t in m.trains] == [totals[t.train] for t in m.trains]
    assert m.penalty_by_type["CANCELLATION"] < 0

    env = make_env(n_agents=10, seed=0)
    totals = run(env, ConstantPolicy(2))                 # always forward: some trains get stuck
    m = episode_metrics(env)
    stuck = [t for t in m.trains if not t.arrived and not t.cancelled]
    assert stuck, "scenario no longer produces stuck trains; pick another seed"
    assert [t.penalty for t in m.trains] == [totals[t.train] for t in m.trains]
    assert all(t.penalty_by_type["TARGET_NOT_REACHED"] < 0 for t in stuck)


# ---- 2. Score unchanged ----

def test_score_is_official_and_matches_reference(finished):
    name, env, totals = finished
    m = episode_metrics(env)
    official = env.rewards.normalize(*totals.values(), num_agents=len(totals), max_episode_steps=env._max_episode_steps)
    assert m.score == pytest.approx(official)
    assert m.score == pytest.approx({"checkpoint": 0.730, "heuristic": 0.513}[name], abs=1e-3)


# ---- 3. Reproduces the hand analysis ----

def test_checkpoint_breakdown(finished):
    name, env, _ = finished
    if name != "checkpoint":
        pytest.skip()
    m = episode_metrics(env)
    assert [t.missed_stops for t in m.trains] == [3, 3, 1, 2, 3, 4]
    assert [t.required_stops for t in m.trains] == [3, 3, 1, 2, 3, 4]
    assert m.halts_while_moving == 0 and m.completion_rate == 1.0 and m.stop_service_rate == 0.0
    assert all(t.lateness == 0 for t in m.trains)          # all arrived early
    assert m.penalty_by_type["INTERMEDIATE_NOT_SERVED"] == -800


def test_heuristic_breakdown(finished):
    name, env, _ = finished
    if name != "heuristic":
        pytest.skip()
    m = episode_metrics(env)
    assert [t.halts_while_moving for t in m.trains] == [0, 0, 0, 3, 0, 1]
    t3 = m.trains[3]
    assert (t3.penalty, t3.penalty_capped, t3.lateness, t3.missed_stops) == (-883, -494, 33, 2)
    assert t3.penalty_by_type["COLLISION"] == -750


def test_stops_served_are_counted():
    # The baselines never stop, so also check a policy that does: served stops must show up.
    env = load_scenario(SAMPLE, disruption="none", obs_builder=FullEnvObservation())
    play_heuristic(env, stop_at_stations=True)
    m = episode_metrics(env)
    assert 0 < m.stop_service_rate <= 1
    for t in m.trains:
        assert t.missed_stops * -50 == t.penalty_by_type["INTERMEDIATE_NOT_SERVED"]


# ---- 4. Malfunctions (breakdowns + departure delays) ----

def count_malfunctions_independently(env):
    """Reset, drive forward, and count transitions into a malfunction state from env.agents."""
    env.reset()
    handles = env.get_agent_handles()
    was = {h: False for h in handles}
    on_map = off_map = 0
    while True:
        _, _, dones, _ = env.step({h: 2 for h in handles})
        for h in handles:
            s = env.agents[h].state
            if s.is_malfunction_state() and not was[h]:
                off_map += s == TrainState.MALFUNCTION_OFF_MAP
                on_map += s == TrainState.MALFUNCTION
            was[h] = s.is_malfunction_state()
        if dones["__all__"]:
            return on_map, off_map


@pytest.mark.parametrize("disruption", ["none", "breakdowns"])
def test_malfunctions_match_independent_count(disruption):
    env = make_env(n_agents=10, disruption=disruption, seed=0)
    on_map, off_map = count_malfunctions_independently(env)
    m = episode_metrics(env)
    assert (m.malfunctions, m.malfunctions_before_departure) == (on_map, off_map)
    if disruption == "none":
        assert on_map + off_map == 0
    else:
        assert on_map + off_map > 0


# ---- 5. Fairness ----

def test_gini():
    assert gini([5, 5, 5]) == 0
    assert gini([0, 0]) == 0 and gini([]) == 0
    assert gini([0, 0, 10]) == pytest.approx(2 / 3)       # one of three takes everything: (n-1)/n


# ---- 6. Errors and output ----

def test_error_if_episode_not_finished():
    env = make_env(n_agents=2, seed=0)
    env.reset()
    env.step({h: 2 for h in env.get_agent_handles()})
    with pytest.raises(ValueError, match="not finished"):
        episode_metrics(env)


def test_error_if_env_not_from_flatmarl():
    raw = RailEnvPersister.load_new(str(SAMPLE), rewards=ECML2026Rewards())[0]
    with pytest.raises(ValueError, match="flatmarl.envs"):
        episode_metrics(raw)


def test_as_row_is_flat(finished):
    _, env, _ = finished
    row = episode_metrics(env).as_row()
    assert "trains" not in row and "penalty_by_type" not in row
    assert {"score", "completion_rate", "gini_penalty", "penalty_collision"} <= set(row)
    assert all(isinstance(v, (int, float, bool)) and not isinstance(v, dict) for v in row.values())
    assert not math.isnan(row["score"])


def test_recorded_breakdown_can_be_pickled(finished):
    # Our recorded data must be picklable (it once used a lambda). The rewards object as a whole is
    # not, after an episode: flatland's own station-visit memory holds a lambda (defaultdict_list).
    # That is one reason parallel workers get a recipe, not an env (test_env_recipe_works_in_another_process).
    _, env, _ = finished
    restored = pickle.loads(pickle.dumps(env.rewards.by_type))
    assert dict(restored) == dict(env.rewards.by_type)
