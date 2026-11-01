"""Tests for scripts/e9_analysis.py without touching the E9 test results: with E8's iteration-200 arms as both the
"200" and the "400" input, every 400 - 200 difference and the change in the gap must be exactly 0, and the primary
must equal E8's (results/e8_test/summary_arrivals.csv)."""
import sys

import pandas as pd
import pytest

from flatmarl import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import e9_analysis as e9  # noqa: E402

E8_ARRIVALS = ROOT / "results" / "e8_test" / "summary_arrivals.csv"


@pytest.fixture(scope="module")
def against_itself():
    return e9.analyse(new_root=e9.E8, new_folders=e9.E8_FOLDERS, new_iteration=200)


def test_rows(against_itself):
    arrivals, metrics = against_itself
    assert len(arrivals) == 7  # 6 settings sampled + T1 none greedy
    assert set(arrivals[arrivals["mode"] == "greedy"].set) == {"t1_scene1_a5-10_ll2"}
    assert len(metrics) == 6 * len(e9.METRICS)


def test_no_change_against_itself(against_itself):
    arrivals, metrics = against_itself
    for c in ("ippo_400_minus_200", "mappo_400_minus_200", "gap_change"):
        assert (arrivals[[c, f"{c}_lo", f"{c}_hi"]] == 0).all().all(), c
    for c in ("ippo_400_minus_200", "mappo_400_minus_200"):
        assert (metrics[[c, f"{c}_lo", f"{c}_hi"]].dropna() == 0).all().all(), c
    assert (arrivals.mappo_minus_ippo_400 == arrivals.mappo_minus_ippo_200).all()


def test_matches_e8(against_itself):
    arrivals, _ = against_itself
    e8 = pd.read_csv(E8_ARRIVALS)
    merged = arrivals.merge(e8, on=["set", "disruption", "mode"])
    assert len(merged) == 7
    for mine, theirs in [("mappo_minus_ippo_400", "mappo_minus_ippo"),
                         ("mappo_minus_ippo_400_lo", "mappo_minus_ippo_lo"),
                         ("mappo_minus_ippo_400_hi", "mappo_minus_ippo_hi"),
                         ("ippo_400", "ippo_arrived"), ("mappo_400", "mappo_arrived"),
                         ("heuristic", "heuristic_arrived"),
                         ("mappo_400_minus_heuristic", "mappo_minus_heuristic")]:
        assert merged[mine].to_numpy() == pytest.approx(merged[theirs].to_numpy(), abs=1e-12), mine
    primary = merged[(merged.set == "t1_scene1_a5-10_ll2") & (merged.disruption == "none")
                     & (merged["mode"] == "sampled")]
    assert primary.mappo_minus_ippo_400.item() == pytest.approx(0.054, abs=0.0005)


@pytest.mark.parametrize("stand_in, unchanged, moved_by_gap", [
    ("e6", "ippo_400_minus_200", {"mappo_400_minus_200": -1, "gap_change": -1}),
    ("e7", "mappo_400_minus_200", {"ippo_400_minus_200": +1, "gap_change": -1})])
def test_signs(stand_in, unchanged, moved_by_gap):
    """One iteration-200 arm standing in for both arms at "400": no gap at 400, so the gap change is minus E8's
    MAPPO - IPPO; the stand-in arm's own 400 - 200 is 0 and the other arm's moves by the E8 gap."""
    arrivals, _ = e9.analyse(new_root=e9.E8, new_folders={"ippo": stand_in, "mappo": stand_in}, new_iteration=200)
    gap = arrivals.mappo_minus_ippo_200
    assert (arrivals.mappo_minus_ippo_400 == 0).all() and (arrivals[unchanged] == 0).all()
    for c, sign in moved_by_gap.items():
        assert arrivals[c].to_numpy() == pytest.approx(sign * gap.to_numpy(), abs=1e-12), c
        bound = arrivals.mappo_minus_ippo_200_hi if sign < 0 else arrivals.mappo_minus_ippo_200_lo
        assert arrivals[f"{c}_lo"].to_numpy() == pytest.approx(sign * bound.to_numpy(), abs=1e-12), c


def test_wrong_iteration_is_refused():
    with pytest.raises(ValueError, match="expected iteration 400"):
        e9.analyse(new_root=e9.E8, new_folders=e9.E8_FOLDERS, new_iteration=400)


def test_validation_windows():
    v = e9.validation_windows()
    assert len(v) == 8  # 2 arms x (3 seeds + mean)
    mean = v[v.seed == "mean"].set_index("arm")
    assert mean.loc["ippo", "iterations_151_200"] == pytest.approx(0.646, abs=0.0005)
    assert mean.loc["mappo", "iterations_151_200"] == pytest.approx(0.683, abs=0.0005)
