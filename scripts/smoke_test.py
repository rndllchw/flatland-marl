"""Smoke test: run the two reference policies on the starter-kit sample scenario.

Checks that the environment is installed correctly and that results match across machines.
Goes through our own flatmarl.envs.load_scenario, so it also checks that our env layer
reproduces the raw flatland results exactly.
Exits non-zero if a score differs from the reference values recorded on 2026-10-17.

Usage (from repo root, venv active):
    python scripts/smoke_test.py
"""
import sys
import time

from flatmarl import STARTER_KIT
from flatmarl.envs import load_scenario
from flatland.envs.step_utils.states import TrainState
from reinforcement_learning.my_policy import ActorCritic
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation

SCENARIO = STARTER_KIT / "reinforcement_learning" / "sampling" / "level_0_scenario_1.pkl"
CHECKPOINT = STARTER_KIT / "reinforcement_learning" / "checkpoint.pt"

# name -> (expected normalised score, expected arrivals); 6 agents, 494 max steps
REFERENCE = {
    "ppo_checkpoint": (0.730, 6),
    "deadlock_heuristic": (0.513, 6),
}


def run_episode(policy, obs_builder):
    """Run one full episode; return (normalised score, trains arrived, n agents, env steps/s)."""
    # disruption=None keeps the file's own disruptions (breakdowns + stop delays), as recorded.
    env = load_scenario(SCENARIO, disruption=None, obs_builder=obs_builder)
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    totals = {h: 0.0 for h in handles}

    start, steps = time.time(), 0
    while True:
        actions = policy.act_many(handles, [obs[h] for h in handles])
        obs, step_rewards, dones, _ = env.step(actions)
        steps += 1
        for h in handles:
            totals[h] += step_rewards[h]
        if dones["__all__"]:
            break
    steps_per_s = steps / (time.time() - start)

    # Official ECML 2026 normalisation (per-agent penalty capped at max_episode_steps).
    score = env.rewards.normalize(*[totals[h] for h in handles],
                              num_agents=len(handles), max_episode_steps=env._max_episode_steps)
    arrived = sum(a.state == TrainState.DONE for a in env.agents)
    return score, arrived, len(handles), steps_per_s


def main():
    policies = {
        # None = load_scenario's default obs builder (starter-kit FastTreeObs + action mask)
        "ppo_checkpoint": (ActorCritic(checkpoint_path=str(CHECKPOINT)), None),
        "deadlock_heuristic": (DeadLockAvoidancePolicy(), FullEnvObservation()),
    }

    ok = True
    for name, (policy, obs_builder) in policies.items():
        score, arrived, n, steps_per_s = run_episode(policy, obs_builder)
        exp_score, exp_arrived = REFERENCE[name]
        match = abs(score - exp_score) < 1e-3 and arrived == exp_arrived
        ok &= match
        print(f"{name:20s} score {score:.3f} (expected {exp_score:.3f})  "
              f"arrived {arrived}/{n} (expected {exp_arrived})  "
              f"{steps_per_s:6.0f} steps/s  {'OK' if match else 'MISMATCH'}")

    print("smoke test PASSED" if ok else "smoke test FAILED: results differ from reference")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
