"""Evaluate policies on saved scenarios.

Writes to --out:
    episodes.csv  one row per episode (policy, scenario, disruption, seed + all episode metrics)
    trains.csv    one row per train per episode (for fairness and failure analysis)
    run.json      exact command, git commit, versions, scenario list (how this table was made)

Examples (from repo root, venv active):
    python scripts/evaluate.py --policies random do_nothing forward checkpoint heuristic \\
        --out results/baselines_example_curriculum --workers 8
    python scripts/evaluate.py --scenarios eval_sets/test --disruption breakdowns --seeds 0 1 2 \\
        --policies heuristic actor:runs/ippo_s0/actor.pt --out results/test_breakdowns
    python scripts/evaluate.py --scenarios pools/val_scene1_a5-10_ll2 --disruption none --seeds 0 1 2 \
        --policies actor-sampled:runs/ippo_s0/actor.pt --out results/val_sampled
        (actor-sampled draws each action from the policy; --seeds then sets the draws, and with
        --disruption none nothing else)
"""
import argparse
import csv
import importlib.metadata
import json
import math
import multiprocessing
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from flatmarl import EXAMPLE_CURRICULUM, ROOT
from flatmarl.envs import DISRUPTIONS, load_scenario
from flatmarl.metrics import episode_metrics
from flatmarl.policies import make_policy

DISRUPTION_CHOICES = ("file", *DISRUPTIONS)  # "file" = each scenario's own disruptions


def run_episode(job):
    """One episode from a picklable recipe (scenario, policy, disruption, seed) -> (episode row, train rows).
    Builds its own env and a fresh policy, so it can run in a worker process."""
    scenario, policy_name, disruption, seed = job
    policy, obs_builder = make_policy(policy_name, seed=seed if seed is not None else 0)
    env = load_scenario(scenario, disruption=None if disruption == "file" else disruption,
                        seed=seed, obs_builder=obs_builder)
    start = time.time()
    obs, _ = env.reset()
    handles = env.get_agent_handles()
    while True:
        obs, _, dones, _ = env.step(policy.act_many(handles, [obs[h] for h in handles]))
        if dones["__all__"]:
            break
    m = episode_metrics(env)

    ids = {"policy": policy_name, "scenario": Path(scenario).stem, "disruption": disruption,
           "seed": "" if seed is None else seed}
    episode = {**ids, **m.as_row(), "runtime_s": round(time.time() - start, 2)}
    trains = []
    for t in m.trains:
        record = asdict(t)
        by_type = record.pop("penalty_by_type")
        trains.append({**ids, **record, **{f"penalty_{k.lower()}": v for k, v in by_type.items()}})
    return episode, trains


def find_scenarios(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        hint = " (run scripts/setup_third_party.sh)" if path == EXAMPLE_CURRICULUM else ""
        raise SystemExit(f"--scenarios: {path} does not exist{hint}")
    scenarios = sorted(path.glob("*.pkl"))
    if not scenarios:
        raise SystemExit(f"--scenarios: no .pkl files in {path}")
    return scenarios


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))}


def print_summary(episodes: list[dict]):
    def mean(values):
        values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
        return sum(values) / len(values) if values else math.nan

    groups = {}
    for e in episodes:
        groups.setdefault((e["policy"], e["disruption"]), []).append(e)
    print(f"\n{'policy':38s} {'disruption':18s} {'episodes':>8s} {'score':>6s} {'arrived':>8s} "
          f"{'stops':>6s} {'halts':>6s}")
    for (policy, disruption), rows in groups.items():
        print(f"{policy:38s} {disruption:18s} {len(rows):8d} {mean([r['score'] for r in rows]):6.3f} "
              f"{mean([r['completion_rate'] for r in rows]):8.0%} {mean([r['stop_service_rate'] for r in rows]):6.0%} "
              f"{mean([r['halts_while_moving'] for r in rows]):6.1f}")
    print("(score/arrived/stops/halts are means over episodes; stops excludes episodes without intermediate stops)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenarios", type=Path, default=EXAMPLE_CURRICULUM,
                        help="folder of .pkl scenarios, or one .pkl (default: starter kit's 27 examples)")
    parser.add_argument("--policies", nargs="+", required=True,
                        help="do_nothing forward random heuristic checkpoint, actor:<path> (top action), "
                             "or actor-sampled:<path> (actions drawn from the policy; seeded by --seeds)")
    parser.add_argument("--disruption", choices=DISRUPTION_CHOICES, default="file",
                        help="'file' = each scenario's own disruptions (default)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="one episode per seed: sets breakdown patterns and the random / actor-sampled "
                             "policies' draws (default: the file's stored randomness, policy seed 0)")
    parser.add_argument("--workers", type=int, default=1, help="scenarios run in parallel")
    parser.add_argument("--out", type=Path, required=True, help="output folder")
    parser.add_argument("--overwrite", action="store_true", help="replace results already in --out")
    args = parser.parse_args(argv)

    scenarios = find_scenarios(args.scenarios)
    for name in args.policies:  # fail fast on a bad name or missing actor file
        make_policy(name)
    if (args.out / "episodes.csv").exists() and not args.overwrite:
        raise SystemExit(f"{args.out} already has results; pass --overwrite to replace them")
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds is not None else [None]
    jobs = [(str(s), p, args.disruption, seed) for p in args.policies for s in scenarios for seed in seeds]
    print(f"{len(jobs)} episodes: {len(args.policies)} policies x {len(scenarios)} scenarios x {len(seeds)} seed(s), "
          f"{args.workers} worker(s)")

    start = time.time()
    if args.workers > 1:
        # Workers receive only recipes and build their own env + policy (envs can't be pickled).
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(run_episode, jobs), 1):  # imap keeps job order
                results.append(result)
                print(f"\r  {i}/{len(jobs)} done", end="", flush=True)
    else:
        results = []
        for i, job in enumerate(jobs, 1):
            results.append(run_episode(job))
            print(f"\r  {i}/{len(jobs)} done", end="", flush=True)
    print(f"  ({time.time() - start:.0f} s)")

    episodes = [e for e, _ in results]
    trains = [t for _, ts in results for t in ts]
    write_csv(args.out / "episodes.csv", episodes)
    write_csv(args.out / "trains.csv", trains)
    run_info = {
        "command": " ".join(["python", *sys.argv]) if argv is None else f"main({argv})",
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "git": git_state(),
        "versions": {pkg: importlib.metadata.version(pkg) for pkg in ("flatland-rl", "torch")},
        "python": sys.version.split()[0],
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenarios": [str(s.relative_to(ROOT)) if s.is_relative_to(ROOT) else str(s) for s in scenarios],
        "episodes": len(episodes),
    }
    (args.out / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    print_summary(episodes)
    print(f"\nwrote {args.out}/episodes.csv ({len(episodes)} rows), trains.csv ({len(trains)} rows), run.json")


if __name__ == "__main__":
    main()
