"""JUDGE + ADJUST: advantages (GAE) and the PPO update.

IPPO: one shared actor for all trains; its value head is the critic and sees only the train's own
observation. MAPPO: the same, plus a CentralCritic branch (models.py) that also sees a summary of all
trains (obs.global_state); its weights are trained together with the actor's. Nothing else differs.

Value-target normalisation (ppo.value_norm, default off; on for both arms from 25 Oct): the critic
learns returns standardised by a running mean/std (ValueNormalizer), as recommended for PPO/MAPPO by
Yu et al. (2022). Its outputs are converted back to reward units before GAE, so advantages are unchanged
in meaning. The stats are not saved with the actor (evaluation never uses the value head); train.py
logs them per iteration.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from flatmarl.models import OBS_SIZE, masked_policy, policy_and_values


class ValueNormalizer:
    """Running mean and standard deviation of every value target seen so far. Before the first update
    it is the identity (mean 0, std 1)."""

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.var = 1.0

    @property
    def std(self) -> float:
        return math.sqrt(self.var) + 1e-8

    def update(self, x: np.ndarray):
        """Merge a batch into the running stats (Chan et al.'s parallel update; exact, not a moving average)."""
        n = len(x)
        if n == 0:
            return
        batch_mean, batch_var = float(np.mean(x)), float(np.var(x))
        if self.count == 0:
            self.count, self.mean, self.var = n, batch_mean, batch_var
            return
        total = self.count + n
        delta = batch_mean - self.mean
        self.mean += delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta ** 2 * self.count * n / total) / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype(np.float32)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return (x * self.std + self.mean).astype(np.float32)


def gae(rewards: np.ndarray, values: np.ndarray, discounts: np.ndarray, lam: float):
    """Advantages and returns for one train's decisions in one episode (the last one is terminal).

    discounts[k] = gamma ** (steps from decision k to decision k+1), so gaps between decisions
    (e.g. while broken down) are discounted correctly.
    """
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    next_advantage = 0.0
    for k in reversed(range(n)):
        terminal = k == n - 1
        next_value = 0.0 if terminal else values[k + 1]
        delta = rewards[k] + discounts[k] * next_value - values[k]
        next_advantage = delta + (0.0 if terminal else discounts[k] * lam * next_advantage)
        advantages[k] = next_advantage
    return advantages, advantages + values


@dataclass
class Batch:
    obs: torch.Tensor          # (N, 41)
    actions: torch.Tensor      # (N,)
    logprobs: torch.Tensor     # (N,) under the collecting policy
    advantages: torch.Tensor   # (N,)
    returns: torch.Tensor      # (N,) value targets
    global_state: torch.Tensor | None = None  # (N, 40), MAPPO only


def make_batch(results, lam: float, normalizer: ValueNormalizer | None = None) -> Batch:
    """All trains' decisions from a list of EpisodeResult, with advantages computed per train.

    With a normalizer: the recorded values are the critic's normalised outputs, so they are converted
    to reward units for GAE; then the normalizer is updated with this batch's returns and the value
    targets are returned normalised."""
    parts = {"obs": [], "actions": [], "logprobs": [], "advantages": [], "returns": []}
    for result in results:
        for s in result.segments:
            values = s.values if normalizer is None else normalizer.denormalize(s.values)
            advantages, returns = gae(s.rewards, values, s.discounts, lam)
            for key, value in (("obs", s.obs), ("actions", s.actions), ("logprobs", s.logprobs),
                               ("advantages", advantages), ("returns", returns)):
                parts[key].append(value)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    if normalizer is not None:
        normalizer.update(arrays["returns"])
        arrays["returns"] = normalizer.normalize(arrays["returns"])
    globals_ = [s.global_state for r in results for s in r.segments]
    if any(g is not None for g in globals_):
        arrays["global_state"] = np.concatenate(globals_)
    return Batch(**{k: torch.from_numpy(v) for k, v in arrays.items()})


def ppo_update(actor, optimizer, batch: Batch, cfg: dict, generator: torch.Generator, critic=None) -> dict:
    """Clipped PPO on `batch` for cfg["epochs"] passes in minibatches. Returns mean training stats.
    critic (MAPPO): the CentralCritic; the optimizer must hold its parameters too."""
    n = len(batch.actions)
    params = list(actor.parameters()) + (list(critic.parameters()) if critic is not None else [])
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    updates = 0
    for _ in range(cfg["epochs"]):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg["minibatch_size"]):
            idx = order[start:start + cfg["minibatch_size"]]
            g = batch.global_state[idx] if critic is not None else None
            logits, values = policy_and_values(actor, batch.obs[idx, :OBS_SIZE], critic, g)
            dist = masked_policy(logits, batch.obs[idx, OBS_SIZE:])
            log_ratio = dist.log_prob(batch.actions[idx]) - batch.logprobs[idx]
            ratio = log_ratio.exp()

            advantages = batch.advantages[idx]
            if len(idx) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            clipped = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            # batch.returns is already normalised when ppo.value_norm is on (make_batch)
            value_loss = 0.5 * (values - batch.returns[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                totals["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                totals["clip_fraction"] += ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            updates += 1
    return {k: v / max(updates, 1) for k, v in totals.items()}
