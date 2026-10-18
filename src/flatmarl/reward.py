"""Training reward. Reported results always use the official reward (metrics.py); this only changes
what the trainer learns from.

Default: official reward x `scale`. The official penalties are large (-50 per missed stop, -250 per
halt while moving), so scaling keeps value targets in a range the critic can learn.

Shaping (Member 4) goes in RewardShaper.__call__, e.g. a bonus the moment a train stops at a required
station, or potential-based progress towards the next stop. Every new term needs a config key that
defaults to off, so IPPO and MAPPO runs can share exactly the same reward.
"""


class RewardShaper:
    def __init__(self, scale: float = 0.01):
        self.scale = scale

    def __call__(self, env, handle: int, official: float) -> float:
        """Training reward for train `handle` this step, given its official reward."""
        return self.scale * official
