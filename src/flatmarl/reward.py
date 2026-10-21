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
"""
import numpy as np
from flatland.envs.step_utils.states import TrainState


class RewardShaper:
    def __init__(self, scale: float = 0.01, progress: float = 0.0, gamma: float = 0.99):
        self.scale = scale
        self.progress = progress
        self.gamma = gamma
        self._phi = {}

    def start(self, env):
        """Call once after env.reset(), before the first step."""
        self._phi = {a.handle: self._potential(env, a, 0.0) for a in env.agents} if self.progress else {}

    def __call__(self, env, handle: int, official: float) -> float:
        """Training reward for train `handle` this step (call after env.step), given its official reward."""
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
