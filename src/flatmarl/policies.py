"""Named policies, each paired with the observation builder it needs.

    policy, obs_builder = make_policy("heuristic")
    env = load_scenario(path, obs_builder=obs_builder)

Names: do_nothing, forward, random, heuristic, checkpoint, or actor:<path> for any network with the
starter-kit ActorCritic shape saved as {"model": state_dict} (how IPPO/MAPPO actors plug in).
actor-sampled:<path> plays the same network but draws each action from its policy (as in training) instead
of taking the top action; `seed` makes the draws reproducible. Added 26 Oct: trained actors learned to
depart with ~50% probability per step (random staggering), which taking the top action turns into "all
trains depart at once" or "none ever departs" (EXPERIMENTS.md, E5).
Call make_policy once per episode: it returns fresh objects, so no policy memory carries over.
"""
from pathlib import Path

import torch
from flatland.core.env_observation_builder import DummyObservationBuilder

from flatmarl import STARTER_KIT
from flatmarl.envs import _default_obs_builder
from flatmarl.models import OBS_SIZE, masked_policy
from flatland_baselines.deadlock_avoidance_heuristic.observation.full_env_observation import FullEnvObservation
from flatland_baselines.deadlock_avoidance_heuristic.policy.deadlock_avoidance_policy import DeadLockAvoidancePolicy
from flatland_baselines.do_nothing_heuristic.do_nothing_policy import DoNothingPolicy
from flatland_baselines.forward_only_heuristic.forward_only_policy import ForwardOnlyPolicy
from reinforcement_learning.my_policy import ActorCritic
from submission.random_policy import RandomPolicy

STARTER_KIT_CHECKPOINT = STARTER_KIT / "reinforcement_learning" / "checkpoint.pt"
POLICY_NAMES = ("do_nothing", "forward", "random", "heuristic", "checkpoint")  # plus actor:<path>, actor-sampled:<path>


class SampledActor(ActorCritic):
    """The starter-kit actor, but each action is drawn from its masked policy instead of being the top one."""

    def __init__(self, checkpoint_path: str, seed: int):
        super().__init__(checkpoint_path=checkpoint_path)
        self.generator = torch.Generator().manual_seed(seed)

    def act(self, observation, **kwargs) -> int:
        x = torch.from_numpy(observation)
        with torch.no_grad():
            logits, _ = self(x[:OBS_SIZE])
            probs = masked_policy(logits, x[OBS_SIZE:]).probs
        return int(torch.multinomial(probs, 1, generator=self.generator))


def _actor(path: Path, sampled_seed: int | None = None):
    # ActorCritic silently starts from random weights if the file is missing, so check first.
    if not path.is_file():
        raise FileNotFoundError(f"actor checkpoint not found: {path}")
    if sampled_seed is not None:
        return SampledActor(str(path), sampled_seed), _default_obs_builder()
    return ActorCritic(checkpoint_path=str(path)), _default_obs_builder()


def make_policy(name: str, seed: int = 0):
    """Return (policy, obs_builder) for `name`; `seed` only affects random and actor-sampled policies."""
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
    if name.startswith("actor-sampled:"):
        return _actor(Path(name.removeprefix("actor-sampled:")), sampled_seed=seed)
    raise ValueError(f"Unknown policy {name!r}; choose from {list(POLICY_NAMES)}, 'actor:<path>' "
                     f"or 'actor-sampled:<path>'")
