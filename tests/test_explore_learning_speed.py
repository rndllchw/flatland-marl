"""Tests for scripts/explore_learning_speed.py: smoothing and the first-iteration rule, on hand-made curves."""
import sys

import pandas as pd

from flatmarl import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import explore_learning_speed as ls  # noqa: E402


def curve(values):
    return pd.Series(values, index=range(10, 10 * len(values) + 1, 10))


def test_single_spike_is_not_enough():
    # one snapshot at 0.70 between 0.55s: smoothed peak 0.60, so 0.65 is never reached
    assert ls.first_reaching(curve([0.55, 0.55, 0.70, 0.55, 0.55]), 0.65) is None
    assert ls.first_reaching(curve([0.55, 0.55, 0.70, 0.55, 0.55]), 0.60) == 30


def test_first_iteration_uses_trailing_window():
    c = curve([0.50, 0.60, 0.64, 0.66, 0.68, 0.70])  # smoothed: -, -, 0.58, 0.633, 0.66, 0.68
    assert ls.first_reaching(c, 0.65) == 50
    assert ls.first_reaching(c, 0.66) == 50  # exactly equal counts as reached
    assert ls.first_reaching(c, 0.69) is None
