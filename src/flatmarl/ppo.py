"""JUDGE + ADJUST: advantages (GAE) and the PPO update.

IPPO: one shared actor for all trains; its value head is the critic and sees only the train's own
observation. MAPPO (Member 3) would replace `values` below with a central critic's estimates from a
summary of all trains (obs.py), and train that critic instead of the actor's value head.
Extension points for Member 2 are marked "Member 2:".
"""
from dataclasses import dataclass

import numpy as np
import torch

from flatmarl.models import OBS_SIZE, masked_policy


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


def make_batch(results, lam: float) -> Batch:
    """All trains' decisions from a list of EpisodeResult, with advantages computed per train."""
    parts = {"obs": [], "actions": [], "logprobs": [], "advantages": [], "returns": []}
    for result in results:
        for s in result.segments:
            advantages, returns = gae(s.rewards, s.values, s.discounts, lam)
            for key, value in (("obs", s.obs), ("actions", s.actions), ("logprobs", s.logprobs),
                               ("advantages", advantages), ("returns", returns)):
                parts[key].append(value)
    return Batch(**{k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()})


def ppo_update(actor, optimizer, batch: Batch, cfg: dict, generator: torch.Generator) -> dict:
    """Clipped PPO on `batch` for cfg["epochs"] passes in minibatches. Returns mean training stats."""
    n = len(batch.actions)
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    updates = 0
    for _ in range(cfg["epochs"]):
        order = torch.randperm(n, generator=generator)
        for start in range(0, n, cfg["minibatch_size"]):
            idx = order[start:start + cfg["minibatch_size"]]
            logits, values = actor(batch.obs[idx, :OBS_SIZE])
            dist = masked_policy(logits, batch.obs[idx, OBS_SIZE:])
            log_ratio = dist.log_prob(batch.actions[idx]) - batch.logprobs[idx]
            ratio = log_ratio.exp()

            advantages = batch.advantages[idx]
            if len(idx) > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            clipped = ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"])
            policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
            # Member 2: value-target normalisation (running mean/std of returns) goes here.
            value_loss = 0.5 * (values - batch.returns[idx]).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg["value_coef"] * value_loss - cfg["entropy_coef"] * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg["max_grad_norm"])
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                totals["approx_kl"] += ((ratio - 1) - log_ratio).mean().item()
                totals["clip_fraction"] += ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            updates += 1
    return {k: v / max(updates, 1) for k, v in totals.items()}
