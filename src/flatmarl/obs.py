"""Global state for MAPPO's central critic (training only; the actor never sees it).

global_state(env, obs) summarises all trains in GLOBAL_SIZE = 40 numbers, the same for every train:
    0-35   mean of the 36 observation features over active trains (zeros if none is active)
    36     share of trains arrived (DONE)
    37     share of trains on the map (MOVING, STOPPED, MALFUNCTION)
    38     share of trains broken down (MALFUNCTION, MALFUNCTION_OFF_MAP)
    39     elapsed steps / max steps
Active = past its earliest departure and not arrived: any state except WAITING and DONE.
Fixed size whatever the number of trains (1 to 500+), as CLAUDE.md §6 requires. Each train's own
observation reaches the critic through the actor's trunk features (models.CentralCritic).
"""
import numpy as np
from flatland.envs.step_utils.states import TrainState

from flatmarl.models import OBS_SIZE

GLOBAL_SIZE = OBS_SIZE + 4
INACTIVE = (TrainState.WAITING, TrainState.DONE)
ON_MAP = (TrainState.MOVING, TrainState.STOPPED, TrainState.MALFUNCTION)
BROKEN = (TrainState.MALFUNCTION, TrainState.MALFUNCTION_OFF_MAP)


def global_state(env, obs: dict) -> np.ndarray:
    """(GLOBAL_SIZE,) float32 summary of all trains; obs = the env's current observations by handle."""
    states = [a.state for a in env.agents]
    active = [a.handle for a in env.agents if a.state not in INACTIVE]
    pooled = (np.mean([obs[h][:OBS_SIZE] for h in active], axis=0) if active
              else np.zeros(OBS_SIZE, dtype=np.float32))
    n = len(states)
    scalars = [sum(s == TrainState.DONE for s in states) / n, sum(s in ON_MAP for s in states) / n,
               sum(s in BROKEN for s in states) / n, env._elapsed_steps / env._max_episode_steps]
    return np.concatenate([pooled, scalars]).astype(np.float32)
