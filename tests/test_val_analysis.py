"""Tests for scripts/val_analysis.py: it must reproduce E3's published validation numbers (EXPERIMENTS.md)
from the committed results, and average sampled evaluations over evaluation seeds before pairing."""
import sys

import pandas as pd
import pytest

from flatmarl import ROOT

sys.path.insert(0, str(ROOT / "scripts"))
import val_analysis  # noqa: E402

E3_VAL = ROOT / "results" / "e3_val_seeds"


def test_reproduces_e3_published_numbers():
    _, checkpoint = val_analysis.load_ref(f"checkpoint={ROOT / 'results/e3_val_baselines'}:checkpoint")
    result = val_analysis.analyse(E3_VAL, {"checkpoint": checkpoint})
    final, best = result["final"], result["best"]
    assert final["mean"]["score"] == pytest.approx(0.780, abs=5e-4)
    assert final["mean"]["completion_rate"] == pytest.approx(0.38, abs=5e-3)
    mean, lo, hi = final["vs"]["checkpoint"]["completion_rate"]        # published: -7% [-14, -1]
    assert (round(mean, 2), round(lo, 2), round(hi, 2)) == (-0.07, -0.14, -0.01)
    assert best["picks"] == {0: 20, 1: 90, 2: 20} and best["mean"]["completion_rate"] == pytest.approx(0.68, abs=5e-3)
    committed = pd.read_csv(E3_VAL / "summary_by_snapshot.csv")
    pd.testing.assert_frame_equal(result["summary"], committed, check_dtype=False)


def test_evaluation_seeds_are_averaged_before_pairing():
    rows = [{"policy": "actor-sampled:r/seed_0/actor_100.pt", "scenario": sc, "seed": es,
             "score": v, "completion_rate": v, "halts_while_moving": 0.0, "cancelled_rate": 0.0}
            for sc, values in (("a", (0.2, 0.4)), ("b", (1.0, 1.0))) for es, v in zip((0, 1), values)]
    e = pd.DataFrame(rows)
    e["iteration"], e["seed"] = 100, 0
    m = val_analysis.per_scenario(e, {0: 100})
    assert m.loc["a", "completion_rate"] == pytest.approx(0.3) and m.loc["b", "completion_rate"] == 1.0
    reference = pd.DataFrame({c: [0.3, 0.5] for c in val_analysis.COLUMNS}, index=["a", "b"])
    mean, lo, hi = val_analysis.paired_ci(m, reference, "completion_rate")
    assert mean == pytest.approx(0.25) and lo <= mean <= hi             # differences 0.0 and 0.5
