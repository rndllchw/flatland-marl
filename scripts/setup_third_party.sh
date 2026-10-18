#!/usr/bin/env bash
# Clone the external repos we depend on into third_party/, pinned to tested commits.
# Safe to re-run: an existing clone is just re-checked-out at the pinned commit.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p third_party

clone_pinned() {
  local name=$1 url=$2 sha=$3
  local dir="third_party/$name"
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  git -C "$dir" fetch -q --depth 1 origin "$sha"
  git -C "$dir" checkout -q "$sha"
  echo "$name @ $(git -C "$dir" rev-parse --short HEAD)"
}

# Observation builder, PPO checkpoint, scenario sampler (MIT).
clone_pinned ecml2026-starterkit \
  https://github.com/flatland-association/ecml2026-starterkit.git \
  54edacd63951bb7fe493aba24e399adc7a32bf53

# Deadlock-avoidance heuristic (MIT). Last commit pinned to flatland-rl 4.2.6; main needs 4.3.0.
clone_pinned flatland-baselines \
  https://github.com/flatland-association/flatland-baselines.git \
  1e9d4a604fb2b2ddaf24a3f5b480914d132acb9e

# The starter kit's 27 example scenarios (placeholder evaluation set until eval_sets/ is frozen).
curriculum=third_party/ecml2026-starterkit/reinforcement_learning/curriculum
unzip -q -o "$curriculum/example_curriculum.zip" -d "$curriculum"
echo "example scenarios: $(ls "$curriculum/example_curriculum"/*.pkl | wc -l | tr -d ' ') in $curriculum/example_curriculum"
