"""Networks for training.

The actor is always the starter kit's ActorCritic (README "For teammates", rule 1), so a trained actor
saved as {"model": state_dict} runs in evaluate.py via actor:<path> and is a valid submission checkpoint.
IPPO uses the actor's own value head as its (local) critic. MAPPO adds a separate central critic.
"""
from pathlib import Path

import torch
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
    if kind == "central":
        # MAPPO (Member 3): a separate critic network whose input is a fixed-size summary of all trains
        # (obs.py), trained in ppo.py in place of the actor's value head. The actor stays unchanged.
        raise NotImplementedError("critic: central (MAPPO) is not implemented yet; see models.check_critic")
