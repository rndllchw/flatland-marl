"""flatmarl: our code for the Flatland ECML 2026 IPPO vs MAPPO project."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "third_party"
STARTER_KIT = THIRD_PARTY / "ecml2026-starterkit"
BASELINES = THIRD_PARTY / "flatland-baselines"
# 27 example scenarios, unzipped by scripts/setup_third_party.sh
EXAMPLE_CURRICULUM = STARTER_KIT / "reinforcement_learning" / "curriculum" / "example_curriculum"

# Make the pinned third-party repos importable (cloned by scripts/setup_third_party.sh):
#   reinforcement_learning.*  (starter kit)   flatland_baselines.*  (heuristic)
for _path in (STARTER_KIT, BASELINES):
    if not _path.is_dir():
        raise ImportError(f"{_path} missing; run scripts/setup_third_party.sh")
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
