#!/bin/zsh
# E9: evaluate the iteration-400 actors on the frozen test sets, as E8 did at 200. Run from the repo root.
# The checkpoint, heuristic and iteration-200 results are reused from results/e8_test (same scenarios and seeds).
set -e -o pipefail
for set in t1_scene1_a5-10_ll2 t2_scene1_a15-20_ll2 t3_scene4_a5-10_ll2; do
  for dis in none breakdowns; do
    out=results/e9_test/$set/$dis
    run() { echo "== $set $dis $1 $(date +%H:%M:%S)"; local name=$1; shift
            .venv/bin/python scripts/evaluate.py --scenarios eval_sets/test/$set --disruption $dis --out $out/$name --workers 10 "$@" 2>&1 | grep -v -i warning | grep "wrote"; }
    for arm in ippo mappo; do
      run ${arm}_sampled --seeds 0 1 2 --policies actor-sampled:results/e9_${arm}_training/seed_{0,1,2}/actor_400.pt
      if [ $set = t1_scene1_a5-10_ll2 ] && [ $dis = none ]; then
        run ${arm}_greedy --seeds 0 --policies actor:results/e9_${arm}_training/seed_{0,1,2}/actor_400.pt
      fi
    done
  done
done
echo "== all done $(date +%H:%M:%S)"
