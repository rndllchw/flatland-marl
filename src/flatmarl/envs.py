"""Environment factory. Every Flatland env in this project is built here.

Two entry points, both used as `obs, info = env.reset()` then `env.step(actions)`:
- make_env: random scenarios on the competition map; each reset samples new routes/timetables
  (training, and building frozen eval sets).
- load_scenario: a saved scenario; each reset replays it exactly, breakdowns included
  (frozen eval sets, smoke test).

Invariants:
- Rewards are always the official ECML2026Rewards. Training-time shaping lives in reward.py.
- Disruptions are always set explicitly after loading. A scenario file otherwise restores its
  own disruption settings and silently overrides anything passed to RailEnvPersister.load_new.
"""
import warnings
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
from flatland.core.effects_generator import EffectsGenerator, make_multi_effects_generator
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.envs.malfunction_effects_generators import (
    IntermediateStopMalfunctionEffectsGenerator,
    MalfunctionEffectsGenerator,
)
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env import RailEnv
from flatland.envs.rewards import ECML2026Rewards

from flatmarl import STARTER_KIT
from reinforcement_learning.my_observation_builder import FastTreeObsBuilder
from reinforcement_learning.sampling.sampling_env_generator import sampling_env_generator

# Full 120x150 competition grid; make_env re-samples routes on it, so this only supplies the rail.
COMPETITION_MAP = STARTER_KIT / "reinforcement_learning" / "sampling" / "level_0_scenario_1.pkl"
SCENES = ("scene_1", "scene_2", "scene_3", "scene_4", "scene_5")

# line_length is a maximum: in small scenes the starter-kit line generator truncates routes it
# cannot connect and warns. Expected behaviour, not an error.
warnings.filterwarnings("ignore", message="Could not find path connecting all", category=UserWarning)


@dataclass(frozen=True)
class Disruption:
    """Disruption settings. Rates are Poisson rates per train per step (P = 1 - exp(-rate))."""
    breakdown_rate: float = 0.0                 # any train, any time
    breakdown_duration: tuple[int, int] = (20, 50)
    stop_delay_rate: float = 0.0                # only while stopped at a station (departure delay)
    stop_delay_duration: tuple[int, int] = (2, 5)


# Values are the organisers' own, from the starter-kit sample scenario (level_0_scenario_1.pkl).
DISRUPTIONS = {
    "none": Disruption(),
    "breakdowns": Disruption(breakdown_rate=1 / 540),
    "breakdowns+delays": Disruption(breakdown_rate=1 / 540, stop_delay_rate=0.1),
}


def _resolve(disruption: str | Disruption) -> Disruption:
    if isinstance(disruption, Disruption):
        return disruption
    if disruption not in DISRUPTIONS:
        raise ValueError(f"Unknown disruption {disruption!r}; choose from {list(DISRUPTIONS)} or pass a Disruption")
    return DISRUPTIONS[disruption]


def _effects_generator(d: Disruption) -> EffectsGenerator:
    """Build a fresh effects generator (fresh = no cached random numbers or counters)."""
    gens = []
    if d.stop_delay_rate > 0:
        gens.append(IntermediateStopMalfunctionEffectsGenerator(
            malfunction_rate=d.stop_delay_rate, min_duration=d.stop_delay_duration[0], max_duration=d.stop_delay_duration[1]))
    if d.breakdown_rate > 0:
        gens.append(MalfunctionEffectsGenerator(ParamMalfunctionGen(
            MalfunctionParameters(d.breakdown_rate, d.breakdown_duration[0], d.breakdown_duration[1]))))
    if not gens:
        return EffectsGenerator()  # base class: no-op
    return gens[0] if len(gens) == 1 else make_multi_effects_generator(*gens)


def _default_obs_builder() -> ObservationBuilder:
    # 36 features + 5-dim action mask, identical in training and inference (CLAUDE.md §6).
    return FastTreeObsBuilder(max_depth=3, with_action_mask=True)


def make_env(
    scene: str = "scene_1",
    n_agents: int | tuple[int, int] = 10,
    line_length: int = 2,
    disruption: str | Disruption = "none",
    seed: int = 0,
    obs_builder: ObservationBuilder | None = None,
) -> RailEnv:
    """Random scenarios on the competition map. Every env.reset() samples new routes/timetables.

    scene:       "scene_1".."scene_5" (scene_5 = full map); restricts which stations are used.
    n_agents:    fixed count, or (lo, hi) inclusive, re-drawn at every reset.
    line_length: max stations per route incl. start and end (2 = no intermediate stops).
    disruption:  preset name from DISRUPTIONS, or a Disruption.
    seed:        the whole sequence of episodes from this env is reproducible from it.
                 Use a different seed per training worker, or all workers see identical scenarios.
    """
    if scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {scene!r}")
    if line_length < 2:
        raise ValueError("line_length must be >= 2 (start and destination)")

    env = RailEnvPersister.load_new(
        str(COMPETITION_MAP), obs_builder=obs_builder or _default_obs_builder(), rewards=ECML2026Rewards())[0]
    env.record_steps = False  # load_new turns on per-step recording, whose action log is never cleared
    env.effects_generator = _effects_generator(_resolve(disruption))  # after load: replaces the file's own
    env = sampling_env_generator(env, line_length=line_length, scene=scene)
    env._seed(seed)  # load_new restores the file's RNG state; without this every env is identical

    if isinstance(n_agents, Integral):  # Integral, not int: also accepts numpy integers
        env.number_of_agents = n_agents
    else:
        lo, hi = n_agents
        if not 1 <= lo <= hi:
            raise ValueError(f"Invalid n_agents range {n_agents}; need 1 <= lo <= hi")
        # Own seeded RNG (the starter kit's curriculum helper uses numpy's global RNG).
        rng = np.random.default_rng(seed)
        original_reset = env.reset

        def reset_with_random_n(*args, **kwargs):
            env.number_of_agents = int(rng.integers(lo, hi + 1))
            return original_reset(*args, **kwargs)

        env.reset = reset_with_random_n

    return env


def load_scenario(
    path: str | Path,
    disruption: str | Disruption | None = None,
    seed: int | None = None,
    obs_builder: ObservationBuilder | None = None,
) -> RailEnv:
    """A saved scenario. Every env.reset() replays it exactly: same routes, timetable and breakdowns.

    disruption: preset name or Disruption; None keeps the disruption settings stored in the file.
    seed:       seeds the breakdown randomness; None reuses the file's stored RNG state.
    env.reset() takes no arguments; for a different breakdown pattern, pass a different seed here.
    """
    env, env_dict = RailEnvPersister.load_new(
        str(path), obs_builder=obs_builder or _default_obs_builder(), rewards=ECML2026Rewards())
    env.record_steps = False  # load_new turns on per-step recording, whose action log is never cleared
    initial_rng_state = env.np_random.get_state()

    def fresh_effects() -> EffectsGenerator:
        if disruption is None:
            return RailEnvPersister._apply_malfunction(env_dict)  # same rebuild load_new itself does
        return _effects_generator(_resolve(disruption))

    original_reset = env.reset

    def replay_reset(*args, **kwargs):
        if args or kwargs:
            raise TypeError("load_scenario envs always replay the same scenario; reset() takes no arguments. "
                            "To get a different breakdown pattern, use load_scenario(path, seed=...) instead.")
        # Effects generators cache random numbers and counters across episodes, so rebuild them,
        # and rewind the RNG, so that every reset is identical.
        env.effects_generator = fresh_effects()
        if seed is None:
            env.np_random.set_state(initial_rng_state)
        return original_reset(regenerate_rail=False, regenerate_schedule=False, random_seed=seed)

    env.reset = replay_reset
    return env
