"""Networks for training.

The actor is always the starter kit's ActorCritic (README "For teammates", rule 1), so a trained actor
saved as {"model": state_dict} runs in evaluate.py via actor:<path> and is a valid submission checkpoint.
IPPO uses the actor's own value head as its (local) critic. MAPPO (critic: central, option A decided 26 Oct)
adds CentralCritic: a branch that sees the actor's trunk features plus a summary of all trains
(obs.global_state), whose output is added to the value head. Its last layer starts at zero, so at the start
MAPPO's critic equals IPPO's exactly; the only difference between the arms is what the critic can see.
The critic is used in training only: the saved actor is unchanged, so execution stays decentralised.
"""
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from reinforcement_learning.my_policy import ActorCritic

OBS_SIZE = 36    # FastTreeObsBuilder features; the 5-number action mask follows them
N_ACTIONS = 5    # 0 do-nothing, 1 left, 2 forward, 3 right, 4 stop
CRITICS = ("local", "central")


def make_actor(checkpoint: str | Path | None = None) -> ActorCritic:
    """Fresh weights (None) or a warm start from a {"model": state_dict} file. Never ActorCritic()'s
    default, which silently depends on the working directory."""
    if checkpoint is not None and not Path(checkpoint).is_file():
        raise FileNotFoundError(f"actor checkpoint not found: {checkpoint}")
    actor = ActorCritic(checkpoint_path=str(checkpoint) if checkpoint is not None else None)
    actor.train()
    return actor


def save_actor(actor: ActorCritic, path: str | Path):
    torch.save({"model": actor.state_dict()}, path)


def masked_policy(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Action distribution with illegal actions (mask < 0.5) given probability 0.
    A large negative number rather than -inf, so entropy stays finite."""
    return Categorical(logits=logits.masked_fill(mask < 0.5, -1e9))


def check_critic(kind: str):
    if kind not in CRITICS:
        raise ValueError(f"critic must be one of {CRITICS}, got {kind!r}")


class CentralCritic(nn.Module):
    """MAPPO's extra value branch: (actor trunk features, global state) -> 128 Tanh -> 1, added to the
    actor's value head. The last layer is zero-initialised, so it adds exactly 0 until trained."""

    def __init__(self, features: int = 128, global_size: int = OBS_SIZE + 4, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features + global_size, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([features, global_state], dim=-1)).squeeze(-1)


def make_critic(kind: str) -> CentralCritic | None:
    """None for IPPO (the actor's value head is the critic); a fresh CentralCritic for MAPPO."""
    check_critic(kind)
    return CentralCritic() if kind == "central" else None


def policy_and_values(actor: ActorCritic, obs: torch.Tensor, critic: CentralCritic | None = None,
                      global_state: torch.Tensor | None = None):
    """Logits and critic values for (N, 36) observations: the actor's own heads (same as actor(obs)),
    plus the central branch when a critic is given."""
    features = actor.trunk(obs)
    logits, values = actor.policy_head(features), actor.value_head(features).squeeze(-1)
    if critic is not None:
        values = values + critic(features, global_state)
    return logits, values
