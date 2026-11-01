"""E9 stop rule: the 400-iteration runs must reproduce E6 (IPPO) and E7 (MAPPO) exactly up to iteration 200.

For every seed whose E9 run has reached it, compares each saved network at iterations 10-200 (actor_<i>.pt, and
critic_<i>.pt for MAPPO) with the E6/E7 file tensor by tensor (torch.equal: bit-identical values). Exits 1 if any
differs. Seeds still below iteration 200 are reported as pending, not as failures.

    python scripts/check_e9_identity.py
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ARMS = {"ippo": ("e6_training", ["actor"]), "mappo": ("e7_training", ["actor", "critic"])}


def identical(a: Path, b: Path) -> bool:
    sa, sb = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu")
    sa, sb = next(iter(sa.values())), next(iter(sb.values()))  # {"model": ...} or {"critic": ...}
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def main():
    failed = False
    for arm, (reference, kinds) in ARMS.items():
        for seed in range(3):
            run = ROOT / "results" / f"e9_{arm}_training" / f"seed_{seed}"
            if not (run / "actor_200.pt").exists():
                print(f"{arm} seed {seed}: pending (not at iteration 200 yet)")
                continue
            bad = [f"{kind}_{i}" for i in range(10, 201, 10) for kind in kinds
                   if not identical(run / f"{kind}_{i}.pt", ROOT / "results" / reference / f"seed_{seed}" / f"{kind}_{i}.pt")]
            n = 20 * len(kinds)
            print(f"{arm} seed {seed}: {n - len(bad)}/{n} networks identical to {reference}" +
                  (f"; DIFFERENT: {', '.join(bad)}" if bad else ""))
            failed |= bool(bad)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
