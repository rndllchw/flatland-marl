"""Tests for the IPPO framework: models, rollout, ppo, scripts/train.py.

These check the training logic is correct, not that it learns well (see configs/ippo_debug.yaml).
"""
import functools
import sys

import numpy as np
import pytest
import torch
import yaml

from flatmarl import ROOT, STARTER_KIT
from flatmarl.envs import load_scenario
from flatmarl.metrics import PENALTY_TYPES
from flatmarl.models import OBS_SIZE, make_actor, masked_policy
from flatmarl.policies import make_policy
from flatmarl.ppo import gae, make_batch, ppo_update
from flatmarl.rollout import collect_episode

sys.path.insert(0, str(ROOT / "scripts"))
import train  # noqa: E402

SAMPLE = STARTER_KIT / "reinforcement_learning" / "sampling" / "level_0_scenario_1.pkl"
PPO = train.DEFAULTS["ppo"]


def state_of(actor):
    return {k: v.detach().clone() for k, v in actor.state_dict().items()}


def test_masked_policy_never_picks_illegal_actions():
    logits = torch.zeros(4, 5)
    mask = torch.tensor([[1, 0, 1, 0, 1], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 0, 1, 0, 0]], dtype=torch.float32)
    dist = masked_policy(logits, mask)
    samples = torch.multinomial(dist.probs, 2000, replacement=True)
    assert all(mask[i, samples[i]].min() == 1 for i in range(4))
    assert torch.isfinite(dist.entropy()).all()


def test_gae_matches_hand_calculation():
    rewards, values, discounts = np.array([1.0, 2.0]), np.array([0.5, 1.0]), np.array([0.9, 0.9])
    # last decision terminal: delta1 = 2 - 1 = 1.  delta0 = 1 + 0.9*1.0 - 0.5 = 1.4
    advantages, returns = gae(rewards, values, discounts, lam=0.95)
    assert advantages == pytest.approx([1.4 + 0.9 * 0.95 * 1.0, 1.0])
    assert returns == pytest.approx(advantages + values)
    advantages, _ = gae(rewards, values, discounts, lam=1.0)       # lam=1: discounted return - value
    assert advantages[0] == pytest.approx((1 + 0.9 * 2) - 0.5)


@pytest.fixture(scope="module")
def episode():
    """Sample scenario (6 trains, stops) with breakdowns, untrained actor, gamma=1 and reward scale 1,
    so stored rewards must add up exactly to the official penalties."""
    env_fn = functools.partial(load_scenario, SAMPLE, disruption="breakdowns")
    return collect_episode((env_fn, state_of(make_actor()), 3, {"scale": 1.0}, 1.0))


def test_every_official_penalty_reaches_a_decision(episode):
    stored = sum(s.rewards.sum() for s in episode.segments)
    official = sum(episode.metrics[f"penalty_{t.lower()}"] for t in PENALTY_TYPES)
    assert official < 0 and stored == pytest.approx(official, rel=1e-5)
    assert episode.unattributed == 0


def test_only_real_decisions_are_recorded(episode):
    for s in episode.segments:
        masks = s.obs[:, OBS_SIZE:]
        assert (masks.sum(axis=1) >= 2).all()                         # more than one legal action
        assert (masks[np.arange(len(s.actions)), s.actions] == 1).all()  # chosen action was legal
        assert np.isfinite(s.logprobs).all() and (s.discounts > 0).all()


def test_ppo_update_changes_the_actor_and_reports_stats(episode):
    actor = make_actor()
    before = state_of(actor)
    stats = ppo_update(actor, torch.optim.Adam(actor.parameters(), lr=1e-3), make_batch([episode], 0.95),
                       PPO, torch.Generator().manual_seed(0))
    assert any(not torch.equal(before[k], v) for k, v in actor.state_dict().items())
    assert all(np.isfinite(v) for v in stats.values())


def write_config(tmp_path, workers=1, **overrides):
    cfg = {"name": "tiny", "env": {"source": "make_env", "scene": "scene_1", "n_agents": 2, "line_length": 2},
           "train": {"iterations": 2, "episodes_per_iteration": 2, "workers": workers, "save_every": 1}}
    for section, values in overrides.items():
        cfg.setdefault(section, {}).update(values)
    path = tmp_path / f"tiny_w{workers}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_train_end_to_end_produces_an_evaluable_actor(tmp_path):
    out = tmp_path / "run"
    train.main(["--config", str(write_config(tmp_path)), "--seed", "0", "--out", str(out)])
    for name in ("config.yaml", "run.json", "train_log.csv", "actor.pt", "actor_1.pt", "actor_2.pt"):
        assert (out / name).exists(), name
    assert len((out / "train_log.csv").read_text().strip().splitlines()) == 1 + 2
    policy, _ = make_policy(f"actor:{out / 'actor.pt'}")              # loads as an evaluation policy
    assert policy is not None
    with pytest.raises(SystemExit, match="overwrite"):
        train.main(["--config", str(write_config(tmp_path)), "--seed", "0", "--out", str(out)])


def test_train_is_reproducible_and_worker_count_independent(tmp_path):
    def weights(name, workers):
        out = tmp_path / name
        train.main(["--config", str(write_config(tmp_path, workers)), "--seed", "7", "--out", str(out)])
        return torch.load(out / "actor.pt")["model"]
    a, b, parallel = weights("a", 1), weights("b", 1), weights("c", 2)
    assert all(torch.equal(a[k], b[k]) and torch.equal(a[k], parallel[k]) for k in a)


def test_config_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        train.load_config(write_config(tmp_path, ppo={"learning_rate": 0.1}))   # typo for lr
    with pytest.raises(NotImplementedError, match="MAPPO"):
        path = tmp_path / "central.yaml"
        path.write_text(yaml.safe_dump({"name": "x", "critic": "central"}))
        train.load_config(path)
    with pytest.raises(ValueError, match="env.pool"):
        train.env_recipe({**train.DEFAULTS["env"], "source": "pool"})
