"""Tests for scripts/e10_analysis.py without any E10 result: E8's iteration-200 arms on T1 stand in for both the
"200" and the "400" D1 arms (so every 400 - 200 is exactly 0 and MAPPO - IPPO is E8's), E9's T1 arms stand in for
the dense-trained ones in the transfer table, and the validation reader runs on hand-made files."""
import sys

import pandas as pd
import pytest

from flatmarl import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import e10_analysis as e10  # noqa: E402

E8_ARRIVALS = ROOT / "results" / "e8_test" / "summary_arrivals.csv"
E9_ARRIVALS = ROOT / "results" / "e9_test" / "summary_arrivals.csv"
E8_FOLDERS = {"ippo": "e6", "mappo": "e7"}


def e8_name(name, iteration, mode):
    return f"{E8_FOLDERS[name]}_{mode}", 200


@pytest.fixture(scope="module")
def stand_in():
    return e10.analyse(root=e10.E8, test_set=e10.T1, name=e8_name)


def test_rows(stand_in):
    arrivals, metrics = stand_in
    assert list(zip(arrivals.disruption, arrivals["mode"])) == [("none", "sampled"), ("none", "greedy"),
                                                                ("breakdowns", "sampled")]
    assert len(metrics) == 2 * len(e10.METRICS)


def test_no_change_against_itself(stand_in):
    arrivals, _ = stand_in
    sampled = arrivals[arrivals["mode"] == "sampled"]
    for c in ("ippo_400_minus_200", "mappo_400_minus_200"):
        assert (sampled[[c, f"{c}_lo", f"{c}_hi"]] == 0).all().all(), c
    assert (sampled.mappo_minus_ippo_400 == sampled.mappo_minus_ippo_200).all()


def test_matches_e8(stand_in):
    arrivals, metrics = stand_in
    e8 = pd.read_csv(E8_ARRIVALS)
    merged = arrivals.merge(e8, on=["set", "disruption", "mode"])
    assert len(merged) == 3
    for mine, theirs in [("mappo_minus_ippo_400", "mappo_minus_ippo"),
                         ("mappo_minus_ippo_400_lo", "mappo_minus_ippo_lo"),
                         ("mappo_minus_ippo_400_hi", "mappo_minus_ippo_hi"), ("ippo_400", "ippo_arrived"),
                         ("mappo_400", "mappo_arrived"), ("heuristic", "heuristic_arrived"),
                         ("ippo_400_minus_heuristic", "ippo_minus_heuristic"),
                         ("mappo_400_minus_heuristic", "mappo_minus_heuristic")]:
        assert merged[mine].to_numpy() == pytest.approx(merged[theirs].to_numpy(), abs=1e-12), mine
    sampled = merged[merged["mode"] == "sampled"]  # E8 compared both modes with the sampled checkpoint
    for mine, theirs in [("mappo_400_minus_checkpoint", "mappo_minus_checkpoint"),
                         ("ippo_400_minus_checkpoint", "ippo_minus_checkpoint"),
                         ("checkpoint", "checkpoint_sampled_arrived")]:
        assert sampled[mine].to_numpy() == pytest.approx(sampled[theirs].to_numpy(), abs=1e-12), mine
    greedy = merged[merged["mode"] == "greedy"]
    assert greedy.checkpoint.to_numpy() == pytest.approx(greedy.checkpoint_greedy_arrived.to_numpy(), abs=1e-12)
    primary = arrivals[(arrivals.disruption == "none") & (arrivals["mode"] == "sampled")]
    assert primary.mappo_minus_ippo_400.item() == pytest.approx(0.054, abs=0.0005)
    e8m = pd.read_csv(ROOT / "results" / "e8_test" / "summary_metrics.csv")
    m = metrics.merge(e8m, on=["set", "disruption", "metric"], suffixes=("", "_e8"))
    assert len(m) == len(metrics)
    for mine, theirs in [("mappo_minus_ippo_400", "mappo_minus_ippo"), ("lo", "lo_e8"), ("hi", "hi_e8")]:
        assert m[mine].to_numpy() == pytest.approx(m[theirs].to_numpy(), abs=1e-12, nan_ok=True), mine


@pytest.mark.parametrize("stand_in, unchanged, moved_by_gap", [
    ("e6", "ippo_400_minus_200", {"mappo_400_minus_200": -1}),
    ("e7", "mappo_400_minus_200", {"ippo_400_minus_200": +1})])
def test_signs(stand_in, unchanged, moved_by_gap):
    """One E8 arm standing in for both arms at "400": no gap at 400 (arrivals and every metric); the stand-in arm's
    own 400 - 200 is 0 and the other arm's moves by E8's MAPPO - IPPO."""
    def name(a, iteration, mode):
        return f"{stand_in if iteration == 400 else E8_FOLDERS[a]}_{mode}", 200
    arrivals, metrics = e10.analyse(root=e10.E8, test_set=e10.T1, name=name)
    sampled = arrivals[arrivals["mode"] == "sampled"]
    assert (sampled.mappo_minus_ippo_400 == 0).all() and (sampled[unchanged] == 0).all()
    assert (metrics.mappo_minus_ippo_400.dropna() == 0).all()
    gap = sampled.mappo_minus_ippo_200
    for c, sign in moved_by_gap.items():
        assert sampled[c].to_numpy() == pytest.approx(sign * gap.to_numpy(), abs=1e-12), c
        bound = sampled.mappo_minus_ippo_200_hi if sign < 0 else sampled.mappo_minus_ippo_200_lo
        assert sampled[f"{c}_lo"].to_numpy() == pytest.approx(sign * bound.to_numpy(), abs=1e-12), c


def test_wrong_iteration_is_refused():
    with pytest.raises(ValueError, match="expected iteration 400"):
        e10.analyse(root=e10.E8, test_set=e10.T1, name=lambda a, it, mode: (f"{E8_FOLDERS[a]}_{mode}", it))


def test_transfer_against_itself():
    t = e10.transfer(new_root=e10.E9, new_name="{arm}_sampled").iloc[0]
    for c in ("ippo_new_minus_old", "mappo_new_minus_old"):
        assert t[c] == t[f"{c}_lo"] == t[f"{c}_hi"] == 0, c
    e9 = pd.read_csv(E9_ARRIVALS)
    e9 = e9[(e9.set == e10.T1) & (e9.disruption == "none") & (e9["mode"] == "sampled")].iloc[0]
    for suffix in ("", "_lo", "_hi"):
        assert t[f"mappo_minus_ippo_new{suffix}"] == pytest.approx(e9[f"mappo_minus_ippo_400{suffix}"], abs=1e-12)
        assert t[f"mappo_minus_ippo_old{suffix}"] == pytest.approx(e9[f"mappo_minus_ippo_400{suffix}"], abs=1e-12)


def test_transfer_signs():
    """E9's IPPO standing in for both dense-trained arms: no dense gap, IPPO unchanged, MAPPO down by E9's gap."""
    t = e10.transfer(new_root=e10.E9, new_name="ippo_sampled").iloc[0]
    assert t.mappo_minus_ippo_new == 0 and t.ippo_new_minus_old == 0
    assert t.mappo_new_minus_old == pytest.approx(-t.mappo_minus_ippo_old, abs=1e-12)
    assert t.mappo_new_minus_old_lo == pytest.approx(-t.mappo_minus_ippo_old_hi, abs=1e-12)


def write_validation(tmp_path, ippo_200, mappo_200, checkpoint, missing=False):
    for a, level in (("ippo", ippo_200), ("mappo", mappo_200)):
        rows = [{"seed": s, "iteration": it, "completion_rate": level + 0.0625 * (s - 1)}  # exact in binary
                for s in range(3) for it in e10.VAL_ITERATIONS if not (missing and a == "mappo" and it == 400)]
        (tmp_path / e10.VALIDATION[a]).mkdir()
        pd.DataFrame(rows).to_csv(tmp_path / e10.VALIDATION[a] / "summary_by_snapshot.csv", index=False)
    (tmp_path / "baselines").mkdir()
    pd.DataFrame([{"policy": e10.CHECKPOINT_SAMPLED, "scenario": "00", "completion_rate": checkpoint},
                  {"policy": "heuristic", "scenario": "00", "completion_rate": 0.6}]
                 ).to_csv(tmp_path / "baselines" / "episodes.csv", index=False)
    return e10.validation(root=tmp_path, baselines=tmp_path / "baselines")


@pytest.mark.parametrize("ippo, mappo, stop", [(0.125, 0.1875, True), (0.125, 0.375, False), (0.375, 0.125, False),
                                               (0.25, 0.125, False), (0.125, 0.25, False)])  # equal is not below
def test_early_stop(tmp_path, ippo, mappo, stop):
    v = write_validation(tmp_path, ippo, mappo, checkpoint=0.25)
    assert v.set_index("iteration").loc[200, "ippo"] == ippo  # seeds -0.0625 / 0 / +0.0625, mean exact
    assert e10.early_stop(v) is stop


def test_incomplete_validation_is_refused(tmp_path):
    with pytest.raises(ValueError, match="expected seeds 0-2"):
        write_validation(tmp_path, 0.25, 0.25, 0.25, missing=True)


@pytest.mark.parametrize("lo, hi, start", [(0.01, 0.05, "the central critic helps"), (-0.02, 0.03, "no detectable"),
                                           (-0.05, -0.01, "MAPPO is behind"), (0.0, 0.04, "no detectable")])
def test_reading(lo, hi, start):
    assert e10.reading(lo, hi).startswith(start)
