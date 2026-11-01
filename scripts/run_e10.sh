#!/bin/zsh
# E10: evaluate the dense-trained actors (iterations 200 and 400) on the new test set D1, and the iteration-400
# actors on T1 none (exploratory). Run from the repo root, after the validation step of scripts/e10_analysis.py.
set -e -o pipefail
CK=third_party/ecml2026-starterkit/reinforcement_learning/checkpoint.pt
run() { echo "== $set $dis $1 $(date +%H:%M:%S)"; local name=$1; shift
        .venv/bin/python scripts/evaluate.py --scenarios eval_sets/test/$set --disruption $dis --out results/e10_test/$set/$dis/$name --workers 10 "$@" 2>&1 | grep -v -i warning | grep "wrote"; }
set=d1_scene1_a15-20_ll2
for dis in none breakdowns; do
  for arm in ippo mappo; do
    for it in 200 400; do
      run ${arm}_${it}_sampled --seeds 0 1 2 --policies actor-sampled:results/e10_${arm}_training/seed_{0,1,2}/actor_$it.pt
    done
    if [ $dis = none ]; then
      run ${arm}_400_greedy --seeds 0 --policies actor:results/e10_${arm}_training/seed_{0,1,2}/actor_400.pt
    fi
  done
  run baselines --seeds 0 1 2 --policies actor-sampled:$CK checkpoint heuristic
done
set=t1_scene1_a5-10_ll2 dis=none
for arm in ippo mappo; do
  run ${arm}_400_sampled --seeds 0 1 2 --policies actor-sampled:results/e10_${arm}_training/seed_{0,1,2}/actor_400.pt
done
echo "== all done $(date +%H:%M:%S)"
