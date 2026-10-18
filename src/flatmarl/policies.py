"""Named policies, each paired with the observation builder it needs.

    policy, obs_builder = make_policy("heuristic")
    env = load_scenario(path, obs_builder=obs_builder)

Names: do_nothing, forward, random, heuristic, checkpoint, or actor:<path> for any network with the
starter-kit ActorCritic shape saved as {"model": state_dict} (how IPPO/MAPPO actors plug in).
Call make_policy once per episode: it returns fresh objects, so no policy memory carries over.
"""
from pathlib import Path

from flatland.core.env_observation_builder import DummyObservationBuilder

from flatmarl import STARTER_KIT
from flatmarl.envs import _default_obs_builder
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from flatland_baselines.do_nothing_heuristic.do_nothing_policy import DoNothingPolicy
from flatland_baselines.forward_only_heuristic.forward_only_policy import ForwardOnlyPolicy
from reinforcement_learning.my_policy import ActorCritic
from submission.random_policy import RandomPolicy

STARTER_KIT_CHECKPOINT = STARTER_KIT / "reinforcement_learning" / "checkpoint.pt"
POLICY_NAMES = ("do_nothing", "forward", "random", "heuristic", "checkpoint")  # plus "actor:<path>"


def _actor(path: Path):
    # ActorCritic silently starts from random weights if the file is missing, so check first.
    if not path.is_file():
        raise FileNotFoundError(f"actor checkpoint not found: {path}")
    return ActorCritic(checkpoint_path=str(path)), _default_obs_builder()


def make_policy(name: str, seed: int = 0):
    """Return (policy, obs_builder) for `name`; `seed` only affects the random policy."""
    if name == "do_nothing":
        return DoNothingPolicy(), DummyObservationBuilder()
    if name == "forward":
        return ForwardOnlyPolicy(), DummyObservationBuilder()
    if name == "random":
        return RandomPolicy(seed=seed), DummyObservationBuilder()
    if name == "heuristic":
        return DeadLockAvoidancePolicy(), FullEnvObservation()
    if name == "checkpoint":
        return _actor(STARTER_KIT_CHECKPOINT)
    if name.startswith("actor:"):
        return _actor(Path(name.removeprefix("actor:")))
    raise ValueError(f"Unknown policy {name!r}; choose from {list(POLICY_NAMES)} or 'actor:<path>'")
