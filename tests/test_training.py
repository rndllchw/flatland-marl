"""Tests for the IPPO framework: models, rollout, ppo, scripts/train.py.

These check the training logic is correct, not that it learns well (see configs/ippo_debug.yaml).
"""
import copy
import dataclasses
import functools
import math
import sys

import numpy as np
import pytest
import torch
import yaml
from flatland.envs.step_utils.states import TrainState

from flatmarl import ROOT, STARTER_KIT
from flatmarl.envs import load_scenario, make_env
from flatmarl.metrics import PENALTY_TYPES, episode_metrics
from flatmarl.models import OBS_SIZE, make_actor, masked_policy
from flatmarl.policies import STARTER_KIT_CHECKPOINT, make_policy
from flatmarl.ppo import ValueNormalizer, gae, make_batch, ppo_update
from flatmarl.reward import RewardShaper
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
           "train": {"iterations": 2, "episodes_per_iteration": 2, "workers": workers, "save_every": 1,
                     "eval_episodes": 2}}
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


ONE_TRAIN = functools.partial(make_env, scene="scene_1", n_agents=1, line_length=2)


def checkpoint_state():
    return state_of(make_actor(STARTER_KIT_CHECKPOINT))


def test_greedy_is_deterministic_and_the_checkpoint_gets_home():
    job = (ONE_TRAIN, checkpoint_state(), 5000, {"scale": 0.01}, 0.99, True)
    first, second = collect_episode(job), collect_episode(job)
    assert first.metrics == second.metrics
    assert first.metrics["completion_rate"] == 1.0


def test_progress_shaping_telescopes_to_the_starting_distance():
    # gamma = 1 and no official reward: the shaping terms sum to phi(end) - phi(start) = start distance
    # for a train that arrives. Checks every step is shaped and credited exactly once.
    seed = 5001
    env = ONE_TRAIN(seed=seed)
    env.reset()
    a = env.agents[0]
    start = env.distance_map.get()[0, a.initial_position[0], a.initial_position[1], a.initial_direction]
    result = collect_episode((ONE_TRAIN, checkpoint_state(), seed, {"scale": 0.0, "progress": 1.0}, 1.0, True))
    assert result.metrics["completion_rate"] == 1.0
    assert sum(s.rewards.sum() for s in result.segments) == pytest.approx(start, rel=1e-6)


def test_train_writes_greedy_checks_and_overwrite_starts_fresh_logs(tmp_path):
    out = tmp_path / "run"
    config = write_config(tmp_path)
    train.main(["--config", str(config), "--seed", "0", "--out", str(out)])
    train.main(["--config", str(config), "--seed", "0", "--out", str(out), "--overwrite"])
    eval_rows = (out / "eval_log.csv").read_text().strip().splitlines()
    assert [r.split(",")[0] for r in eval_rows[1:]] == ["0", "1", "2"]       # before training + each save
    assert len((out / "train_log.csv").read_text().strip().splitlines()) == 1 + 2   # not appended to the old run


# reward.exempt_chosen_stops (team decision, 25 Oct)

STOP, FORWARD = 4, 2


def test_chosen_stop_is_free_in_training_but_still_charged_officially():
    env = ONE_TRAIN(seed=5000)
    env.reset()
    exempt, official_only = RewardShaper(scale=1.0, exempt_chosen_stops=True), RewardShaper(scale=1.0)
    exempt.start(env)
    official_only.start(env)
    for _ in range(20):                                   # drive until moving
        if env.agents[0].state == TrainState.MOVING:
            break
        _, rewards, _, _ = env.step({0: FORWARD})
        exempt(env, 0, rewards[0], FORWARD)
        official_only(env, 0, rewards[0], FORWARD)
    assert env.agents[0].state == TrainState.MOVING
    _, rewards, _, _ = env.step({0: STOP})
    assert rewards[0] == -250                              # flatland charges the chosen STOP
    assert official_only(env, 0, rewards[0], STOP) == -250
    assert exempt(env, 0, rewards[0], STOP) == 0
    assert exempt.exempted_halts == 1


def test_forced_halts_keep_their_charge():
    # Scene 1, 4 trains, seed 0, everyone driving forward: one train is halted by being blocked
    # (found 25 Oct). No STOP is ever pressed, so the exemption must change nothing.
    env = make_env(scene="scene_1", n_agents=4, line_length=2, seed=0)
    env.reset()
    shaper = RewardShaper(scale=1.0, exempt_chosen_stops=True)
    shaper.start(env)
    official = training = 0.0
    while True:
        actions = {h: FORWARD for h in env.get_agent_handles()}
        _, rewards, dones, _ = env.step(actions)
        for h in actions:
            official += rewards[h]
            training += shaper(env, h, rewards[h], actions[h])
        if dones["__all__"]:
            break
    assert episode_metrics(env).halts_while_moving >= 1
    assert shaper.exempted_halts == 0 and training == official


def test_exemption_changes_the_training_reward_only():
    # Same actor and seed with and without the exemption: identical actions, so identical official
    # metrics; the stored training rewards differ by exactly 250 per exempted halt (gamma 1, scale 1).
    env_fn = functools.partial(load_scenario, SAMPLE, disruption="breakdowns")
    state = state_of(make_actor())
    plain = collect_episode((env_fn, state, 3, {"scale": 1.0}, 1.0))
    exempt = collect_episode((env_fn, state, 3, {"scale": 1.0, "exempt_chosen_stops": True}, 1.0))
    same = lambda a, b: a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))  # noqa: E731
    assert plain.metrics.keys() == exempt.metrics.keys()
    assert all(same(plain.metrics[k], exempt.metrics[k]) for k in plain.metrics)
    assert plain.exempted_halts == 0 and 0 < exempt.exempted_halts <= exempt.metrics["halts_while_moving"]
    stored = lambda r: sum(s.rewards.sum() for s in r.segments)  # noqa: E731
    assert stored(exempt) - stored(plain) == pytest.approx(250 * exempt.exempted_halts, rel=1e-5)


# ppo.value_norm (25 Oct)

def test_value_normalizer_matches_numpy_over_batches():
    rng = np.random.default_rng(0)
    norm = ValueNormalizer()
    assert norm.denormalize(np.array([1.5])) == pytest.approx([1.5])   # identity before the first update
    batches = [rng.normal(-3, 5, n) for n in (10, 1, 250)]
    for b in batches:
        norm.update(b)
    everything = np.concatenate(batches)
    assert norm.mean == pytest.approx(everything.mean()) and norm.std == pytest.approx(everything.std())
    x = rng.normal(size=5)
    assert norm.denormalize(norm.normalize(x)) == pytest.approx(x, abs=1e-5)


def test_make_batch_normalises_targets_and_denormalises_values(episode):
    plain = make_batch([episode], 0.95)
    norm = ValueNormalizer()
    first = make_batch([episode], 0.95, norm)
    assert torch.allclose(first.advantages, plain.advantages)          # identity stats: same GAE
    assert norm.mean == pytest.approx(plain.returns.mean().item(), rel=1e-4)
    assert first.returns.mean().item() == pytest.approx(0, abs=1e-4)
    assert first.returns.std(unbiased=False).item() == pytest.approx(1, abs=1e-4)
    # Later batches: the recorded values are the critic's normalised outputs, so GAE must use
    # value * std + mean.
    before = copy.deepcopy(norm)
    second = make_batch([episode], 0.95, norm)
    in_reward_units = dataclasses.replace(episode, segments=[dataclasses.replace(s, values=before.denormalize(s.values))
                                                             for s in episode.segments])
    assert torch.allclose(second.advantages, make_batch([in_reward_units], 0.95).advantages, atol=1e-4)


def test_train_with_exemption_and_value_norm(tmp_path):
    out = tmp_path / "run"
    config = write_config(tmp_path, reward={"exempt_chosen_stops": True}, ppo={"value_norm": True})
    train.main(["--config", str(config), "--seed", "0", "--out", str(out)])
    header = (out / "train_log.csv").read_text().splitlines()[0].split(",")
    assert {"exempted_halts", "return_mean", "return_std"} <= set(header)
    assert set(torch.load(out / "actor.pt")) == {"model"}                # still a starter-kit checkpoint
    policy, _ = make_policy(f"actor:{out / 'actor.pt'}")
    assert policy is not None
