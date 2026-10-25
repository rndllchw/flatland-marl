"""Training reward. Reported results always use the official reward (metrics.py); this only changes
what the trainer learns from.

Default: official reward x `scale`. The official penalties are large (-50 per missed stop, -250 per
halt while moving), so scaling keeps value targets in a range the critic can learn.

Shaping terms (each a config key under `reward:`, default off, so IPPO and MAPPO runs can share
exactly the same reward):

progress   Potential-based progress towards the destination (Ng et al., 1999): each step adds
           progress * (gamma * phi(after) - phi(before)), with phi = -(cells to the destination).
           Moving one cell closer earns about +progress. Potential-based shaping does not change
           which policy is optimal; it gives the feedback immediately instead of at arrival.
           Uses the destination only (not intermediate stops).

exempt_chosen_stops
           Team decision, 25 Oct. Flatland 4.2.6 charges COLLISION (250) on every MOVING->STOPPED,
           including a STOP the train chose, so the official reward punishes giving way and serving
           stations. With this on, the COLLISION charged in a step where the train's action was STOP
           is removed from its training reward; halts forced by being blocked keep the charge. This
           changes which policy is best (unlike scale and progress), so it is a stated design choice.
           Flatland charges it in the same step as the STOP (checked 25 Oct, 639/639 STOPs). The charge
           is read from RecordingRewards (env built by flatmarl.envs), not recomputed.
"""
import numpy as np
from flatland.envs.rail_env import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from flatmarl.metrics import COLLISION, RecordingRewards

STOP = RailEnvActions.STOP_MOVING.value  # 4; RailEnvActions is a plain Enum, so 4 != STOP_MOVING itself


class RewardShaper:
    def __init__(self, scale: float = 0.01, progress: float = 0.0, exempt_chosen_stops: bool = False,
                 gamma: float = 0.99):
        self.scale = scale
        self.progress = progress
        self.exempt_chosen_stops = exempt_chosen_stops
        self.gamma = gamma
        self.exempted_halts = 0  # chosen STOPs whose COLLISION charge was removed this episode
        self._phi = {}
        self._collision = {}

    def start(self, env):
        """Call once after env.reset(), before the first step."""
        self._phi = {a.handle: self._potential(env, a, 0.0) for a in env.agents} if self.progress else {}
        self.exempted_halts = 0
        if self.exempt_chosen_stops:
            if not isinstance(env.rewards, RecordingRewards):
                raise ValueError("exempt_chosen_stops needs an env built by flatmarl.envs (RecordingRewards)")
            self._collision = {a.handle: env.rewards.by_type[a.handle][COLLISION] for a in env.agents}

    def __call__(self, env, handle: int, official: float, action: int | None = None) -> float:
        """Training reward for train `handle` this step (call after env.step), given its official reward
        and the action it was given this step."""
        if self.exempt_chosen_stops:
            total = env.rewards.by_type[handle][COLLISION]
            charged = total - self._collision[handle]  # this step's COLLISION penalty: 0 or negative
            self._collision[handle] = total
            if charged and action == STOP:
                official -= charged
                self.exempted_halts += 1
        reward = self.scale * official
        if self.progress:
            before = self._phi[handle]
            after = self._potential(env, env.agents[handle], before)
            self._phi[handle] = after
            reward += self.progress * (self.gamma * after - before)
        return reward

    @staticmethod
    def _potential(env, agent, fallback: float) -> float:
        """-(cells from the train to its destination); 0 once arrived. Trains not yet on the map use
        their starting cell. If the distance is undefined (facing a way that cannot reach the
        destination), keep the previous potential so the step adds no shaping."""
        if agent.state == TrainState.DONE:
            return 0.0
        position = agent.position if agent.position is not None else agent.initial_position
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        distance = env.distance_map.get()[agent.handle, position[0], position[1], direction]
        return -float(distance) if np.isfinite(distance) else fallback
