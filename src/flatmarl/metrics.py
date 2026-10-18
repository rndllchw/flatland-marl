"""Episode metrics. Usage, for any env built by flatmarl.envs:

    obs, info = env.reset()
    ...step until dones["__all__"]...
    m = episode_metrics(env)          # m.score, m.completion_rate, m.trains[3].halts_while_moving, m.as_row()

How it works: envs.py gives every episode a RecordingRewards, a drop-in ECML2026Rewards that
returns exactly the same rewards but also writes down, per train, flatland's own per-type
penalties plus halts while moving and malfunctions as they happen. Nothing is recomputed here.
"""
import math
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.rewards import DefaultPenalties, ECML2026Rewards
from flatland.envs.step_utils.states import TrainState

PENALTY_TYPES = tuple(p.value for p in DefaultPenalties)  # flatland's 8 names, e.g. "COLLISION"
COLLISION = DefaultPenalties.COLLISION.value
NOT_SERVED = DefaultPenalties.INTERMEDIATE_NOT_SERVED.value


def _zero_penalties() -> dict:
    # A named function, not a lambda, so RecordingRewards can be pickled.
    return dict.fromkeys(PENALTY_TYPES, 0.0)


class RecordingRewards(ECML2026Rewards):
    """ECML2026Rewards that also records what each penalty was for.

    Returns exactly what ECML2026Rewards returns: DefaultRewards sums the same per-type dict
    from self._proxy; we record that dict before summing it.
    """

    def __init__(self):
        super().__init__()
        self.by_type = defaultdict(_zero_penalties)  # train -> type -> total
        self.halts_while_moving = defaultdict(int)
        self.malfunctions = defaultdict(int)
        self.malfunctions_before_departure = defaultdict(int)
        self._was_broken = defaultdict(bool)
        self.episode_ended = False

    def _record(self, handle, penalties: dict) -> float:
        for name, value in penalties.items():
            self.by_type[handle][name] += value
        return sum(penalties.values())  # identical to DefaultRewards

    def step_reward(self, agent, agent_transition_data, distance_map, elapsed_steps) -> float:
        penalties = self._proxy.step_reward(agent, agent_transition_data, distance_map, elapsed_steps)
        if penalties[COLLISION] != 0:
            self.halts_while_moving[agent.handle] += 1
        broken = agent.state.is_malfunction_state()
        if broken and not self._was_broken[agent.handle]:
            if agent.state == TrainState.MALFUNCTION_OFF_MAP:
                self.malfunctions_before_departure[agent.handle] += 1
            else:
                self.malfunctions[agent.handle] += 1
        self._was_broken[agent.handle] = broken
        return self._record(agent.handle, penalties)

    def end_of_episode_reward(self, agent, distance_map, elapsed_steps) -> float:
        self.episode_ended = True
        return self._record(agent.handle, self._proxy.end_of_episode_reward(agent, distance_map, elapsed_steps))


@dataclass
class TrainRecord:
    train: int
    required_stops: int
    missed_stops: int
    arrived: bool
    cancelled: bool                      # never departed
    arrival_step: int | None
    deadline: int | None                 # latest allowed arrival at destination
    lateness: int | None                 # steps late (0 if on time or early); None if not arrived
    halts_while_moving: int              # times a MOVING train came to a halt, by choosing STOP or by being
                                         # blocked: flatland charges COLLISION (250) for both
    malfunctions: int                    # on the map: breakdowns + departure delays (one state in flatland)
    malfunctions_before_departure: int   # breakdowns before the train entered the map
    penalty: float                       # total, uncapped
    penalty_capped: float                # max(penalty, -max_steps): what the score counts
    penalty_by_type: dict                # flatland's 8 types; sums to `penalty`


@dataclass
class EpisodeMetrics:
    score: float                         # official normalised score (env.rewards.normalize)
    n_trains: int
    max_steps: int
    steps: int
    completion_rate: float
    cancelled_rate: float
    on_time_rate: float                  # among arrived trains; nan if none arrived
    mean_lateness: float                 # among arrived trains; nan if none arrived
    stop_service_rate: float             # served / required over all trains; nan if no stops
    halts_while_moving: int
    malfunctions: int                    # breakdowns + departure delays, on the map
    malfunctions_before_departure: int
    p90_lateness: float                  # fairness, among arrived trains; nan if none arrived
    max_lateness: float
    gini_penalty: float                  # fairness: 0 = penalties shared equally, ->1 = a few trains take all
    penalty_by_type: dict
    trains: list

    def as_row(self) -> dict:
        """Everything except the per-train list, flattened: one CSV row."""
        row = {k: v for k, v in asdict(self).items() if k not in ("trains", "penalty_by_type")}
        row.update({f"penalty_{name.lower()}": v for name, v in self.penalty_by_type.items()})
        return row


def gini(values) -> float:
    """Gini coefficient of non-negative values; 0 when all are equal (including all zero)."""
    x = np.sort(np.asarray(values, dtype=float))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * total))


def _mean_or_nan(values) -> float:
    return float(np.mean(values)) if len(values) else math.nan


def episode_metrics(env: RailEnv) -> EpisodeMetrics:
    """Metrics for the episode that just finished on `env` (built by flatmarl.envs)."""
    rec = env.rewards
    if not isinstance(rec, RecordingRewards):
        raise ValueError("env.rewards is not a RecordingRewards; build the env with flatmarl.envs "
                         "(make_env / load_scenario)")
    if not rec.episode_ended:
        raise ValueError("episode not finished; call episode_metrics after dones['__all__'] is True")

    max_steps = env._max_episode_steps
    trains = []
    for agent in env.agents:
        h = agent.handle
        by_type = dict(rec.by_type[h])
        penalty = sum(by_type.values())
        arrived = agent.state == TrainState.DONE
        trains.append(TrainRecord(
            train=h,
            required_stops=max(len(agent.waypoints) - 2, 0),
            missed_stops=round(-by_type[NOT_SERVED] / rec.intermediate_not_served_penalty),
            arrived=arrived,
            cancelled=not arrived and agent.state.is_off_map_state(),
            arrival_step=agent.arrival_time if arrived else None,
            deadline=agent.latest_arrival,
            lateness=max(agent.arrival_time - agent.latest_arrival, 0) if arrived else None,
            halts_while_moving=rec.halts_while_moving[h],
            malfunctions=rec.malfunctions[h],
            malfunctions_before_departure=rec.malfunctions_before_departure[h],
            penalty=penalty,
            penalty_capped=max(penalty, -max_steps),
            penalty_by_type=by_type,
        ))

    n = len(trains)
    arrived = [t for t in trains if t.arrived]
    lateness = [t.lateness for t in arrived]
    required = sum(t.required_stops for t in trains)
    return EpisodeMetrics(
        score=float(rec.normalize(*[t.penalty for t in trains], num_agents=n, max_episode_steps=max_steps)),
        n_trains=n,
        max_steps=max_steps,
        steps=env._elapsed_steps,
        completion_rate=len(arrived) / n,
        cancelled_rate=sum(t.cancelled for t in trains) / n,
        on_time_rate=_mean_or_nan([lat == 0 for lat in lateness]),
        mean_lateness=_mean_or_nan(lateness),
        stop_service_rate=(required - sum(t.missed_stops for t in trains)) / required if required else math.nan,
        halts_while_moving=sum(t.halts_while_moving for t in trains),
        malfunctions=sum(t.malfunctions for t in trains),
        malfunctions_before_departure=sum(t.malfunctions_before_departure for t in trains),
        p90_lateness=float(np.percentile(lateness, 90)) if lateness else math.nan,
        max_lateness=float(max(lateness)) if lateness else math.nan,
        gini_penalty=gini([max(-t.penalty, 0.0) for t in trains]),
        penalty_by_type={name: sum(t.penalty_by_type[name] for t in trains) for name in PENALTY_TYPES},
        trains=trains,
    )
