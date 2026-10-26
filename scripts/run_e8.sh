#!/bin/zsh
# E8: evaluate the pre-registered policies on the frozen test sets. Run from the repo root.
set -e -o pipefail
CK=third_party/ecml2026-starterkit/reinforcement_learning/checkpoint.pt
for set in t1_scene1_a5-10_ll2 t2_scene1_a15-20_ll2 t3_scene4_a5-10_ll2; do
  for dis in none breakdowns; do
    out=results/e8_test/$set/$dis
    run() { echo "== $set $dis $1 $(date +%H:%M:%S)"; local name=$1; shift
            .venv/bin/python scripts/evaluate.py --scenarios eval_sets/test/$set --disruption $dis --out $out/$name --workers 10 "$@" 2>&1 | grep -v -i warning | grep "wrote"; }
    for e in e6 e7; do
      run ${e}_sampled --seeds 0 1 2 --policies actor-sampled:results/${e}_training/seed_{0,1,2}/actor_200.pt
      if [ $dis = none ]; then g=(0); else g=(0 1 2); fi
      run ${e}_greedy --seeds $g --policies actor:results/${e}_training/seed_{0,1,2}/actor_200.pt
    done
    run baselines --seeds 0 1 2 --policies actor-sampled:$CK checkpoint heuristic
  done
done
echo "== all done $(date +%H:%M:%S)"
