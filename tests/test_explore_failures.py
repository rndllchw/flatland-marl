"""Tests for scripts/explore_failures.py: the per-train numbers and the bucket rules, on hand-made train histories.
(The replay itself is checked by the script: it stops unless every train matches the stored E8 results.)"""
import sys

import pandas as pd
from flatland.envs.step_utils.states import TrainState as S

from flatmarl import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import explore_failures as ef  # noqa: E402

A, B, C = (0, 0), (0, 1), (0, 2)  # cells
FORWARD, STOP = 2, 4


def facts(positions, actions, states=None, earliest_departure=0):
    return ef.train_facts(positions, states or [S.STOPPED] * len(actions), actions, earliest_departure)


def test_never_left():
    assert facts([None] * 4, [0] * 3) == {"departure_wait": None, "still_steps": None, "wanted_move_share": None,
                                          "driven": None, "revisited_cells": None}


def test_departure_wait_counts_from_earliest_departure():
    # earliest departure 2: first seen on the map before step 3 = left on time; before step 5 = waited 2 steps
    assert facts([None, None, None, A, B, C], [0, 0, FORWARD, FORWARD, FORWARD], earliest_departure=2)["departure_wait"] == 0
    assert facts([None] * 5 + [A, B], [0, 0, 0, 0, FORWARD, FORWARD], earliest_departure=2)["departure_wait"] == 2


def test_still_steps_and_share_asking_to_move():
    # enters on A, moves to B, then stands on B for the last 3 steps: asks to move twice, chooses stop once
    f = facts([None, A, B, B, B, B], [FORWARD, FORWARD, FORWARD, FORWARD, STOP])
    assert f["still_steps"] == 3 and f["wanted_move_share"] == 2 / 3
    # do-nothing counts as asking to move only while the train is moving
    f = facts([A, B, B, B], [0, 0, 0], states=[S.MOVING, S.MOVING, S.STOPPED])
    assert f["still_steps"] == 2 and f["wanted_move_share"] == 1 / 2
    # moved on the last step: not still at all
    f = facts([A, B, C], [FORWARD, FORWARD])
    assert f["still_steps"] == 0 and f["wanted_move_share"] is None


def test_driven_and_revisited_cells():
    # enters on A, drives A-B-C, turns back C-B-A, then stands on A: 4 moves, 2 of them into cells seen before
    f = facts([None, A, B, C, B, A, A], [FORWARD] * 6)
    assert f["driven"] == 4 and f["revisited_cells"] == 2
    f = facts([None, A, B, C, C], [FORWARD] * 4)
    assert f["driven"] == 2 and f["revisited_cells"] == 0


def test_buckets_and_threshold():
    failed = pd.DataFrame({"departure_wait": [None, 0, 0, 0], "still_steps": [None, 60, 50, 49],
                           "wanted_move_share": [None, 0.9, 0.0, 1.0]})
    assert list(ef.bucket(failed, 50)) == ["never_left", "stuck", "stuck", "still_driving"]
    assert ef.bucket(failed, 25).iloc[3] == "stuck"
    assert ef.bucket(failed, 100).iloc[1] == "still_driving"
