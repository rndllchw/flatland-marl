"""PLAY: run one episode with the current actor and record what PPO needs.

Which steps are recorded ("decisions"): a train's step is recorded only if its choice matters, i.e.
its action mask allows more than one action and it is not broken down. This skips trains WAITING to
depart (only do-nothing is legal), DONE trains, and malfunctioning trains (their actions are ignored).
On the map STOP and DO_NOTHING are always legal, so every on-map step is a decision. That is
deliberate: stopping at stations matters everywhere, not only at switches.

Where rewards go: every reward a train receives is added to its most recent decision, discounted by
the steps since that decision. So the large penalties that arrive at arrival or at episode end reach
the decisions that caused them. `unattributed` sums any reward that arrived before a train's first
decision; it should always be 0 (tests check this).

collect_episode takes one picklable job, so it runs the same in-process or in a worker process.
"""
from dataclasses import dataclass

import numpy as np
import torch

from flatmarl.metrics import episode_metrics
from flatmarl.models import OBS_SIZE, make_actor, masked_policy
from flatmarl.reward import RewardShaper


@dataclass
class Segment:
    """One train's decisions in one episode, in order. The last one is terminal (the episode ends)."""
    obs: np.ndarray        # (n, 41) float32: 36 features + action mask
    actions: np.ndarray    # (n,) int64
    logprobs: np.ndarray   # (n,) float32, under the policy that collected them
    values: np.ndarray     # (n,) float32, the critic's estimate at each decision
    rewards: np.ndarray    # (n,) float32: discounted training reward until the next decision
    discounts: np.ndarray  # (n,) float32: gamma ** (steps until the next decision)


@dataclass
class EpisodeResult:
    segments: list
    metrics: dict          # episode_metrics(env).as_row(): official numbers, for logging
    unattributed: float    # |official reward| that reached no decision (should be 0)


def collect_episode(job) -> EpisodeResult:
    """job = (env_fn, actor_state_dict, seed, reward_config, gamma). env_fn(seed=...) builds the env,
    e.g. functools.partial(make_env, ...) or functools.partial(make_pool_env, pool=...)."""
    env_fn, actor_state, seed, reward_config, gamma = job
    actor = make_actor()
    actor.load_state_dict(actor_state)
    generator = torch.Generator().manual_seed(seed)
    shaper = RewardShaper(**reward_config)

    env = env_fn(seed=seed)
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    rec = {h: {"obs": [], "actions": [], "logprobs": [], "values": [], "rewards": [], "steps": []} for h in handles}
    unattributed = 0.0

    while True:
        deciding = [h for h in handles
                    if obs[h][OBS_SIZE:].sum() > 1.5 and not env.agents[h].state.is_malfunction_state()]
        actions = {h: 0 for h in handles}  # do-nothing: the only legal action, or ignored while broken
        if deciding:
            x = torch.from_numpy(np.stack([obs[h] for h in deciding]))
            with torch.no_grad():
                logits, values = actor(x[:, :OBS_SIZE])
                dist = masked_policy(logits, x[:, OBS_SIZE:])
                chosen = torch.multinomial(dist.probs, 1, generator=generator).squeeze(1)
                logprobs = dist.log_prob(chosen)
            for i, h in enumerate(deciding):
                r = rec[h]
                r["obs"].append(x[i].numpy())
                r["actions"].append(int(chosen[i]))
                r["logprobs"].append(float(logprobs[i]))
                r["values"].append(float(values[i]))
                r["rewards"].append(0.0)
                r["steps"].append(0)
                actions[h] = int(chosen[i])

        obs, rewards, dones, _ = env.step(actions)
        for h in handles:
            r = rec[h]
            if r["steps"]:
                r["rewards"][-1] += gamma ** r["steps"][-1] * shaper(env, h, rewards[h])
                r["steps"][-1] += 1
            else:
                unattributed += abs(rewards[h])
        if dones["__all__"]:
            break

    segments = [Segment(obs=np.stack(r["obs"]).astype(np.float32),
                        actions=np.array(r["actions"], dtype=np.int64),
                        logprobs=np.array(r["logprobs"], dtype=np.float32),
                        values=np.array(r["values"], dtype=np.float32),
                        rewards=np.array(r["rewards"], dtype=np.float32),
                        discounts=(gamma ** np.array(r["steps"], dtype=np.float64)).astype(np.float32))
                for r in rec.values() if r["actions"]]
    return EpisodeResult(segments=segments, metrics=episode_metrics(env).as_row(), unattributed=unattributed)
