"""Pre-generate a pool of scenarios, so training can replay them instead of inventing routes per reset.

Inventing routes with intermediate stops takes 7-21 s per reset; replaying a saved scenario takes
0.2-0.9 s (flatmarl.envs.ScenarioPoolEnv). Scenario i is make_env(..., seed=seed + i), reset once and
saved as <out>/<i>.pkl with no disruptions; the breakdown setting is chosen when the pool is used.
Re-running skips files that already exist, so a long run can be resumed.

Keep seed ranges disjoint between training pools and evaluation sets (e.g. train from 0, eval from 10**6).

Example (from repo root, venv active):
    python scripts/make_scenario_pool.py --out pools/scene1_a5-10_ll3 --n 2000 \\
        --scene scene_1 --n-agents 5 10 --line-length 3 --seed 0 --workers 10
"""
import argparse
import importlib.metadata
import json
import multiprocessing
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from flatland.envs.persistence import RailEnvPersister

from flatmarl import ROOT
from flatmarl.envs import SCENES, make_env


def build_one(job):
    """(path, scene, n_agents, line_length, seed) -> number of trains. Picklable recipe for workers."""
    path, scene, n_agents, line_length, seed = job
    env = make_env(scene=scene, n_agents=n_agents, line_length=line_length, disruption="none", seed=seed)
    env.reset()
    # Flatland picks the save format from the name's ending ("pkl"); ".tmp_pkl" satisfies it but is
    # not matched by the pool's "*.pkl", so a half-written file is never mistaken for a scenario.
    tmp = Path(path).with_suffix(".tmp_pkl")
    RailEnvPersister.save(env, str(tmp))
    tmp.rename(path)  # only complete files get the .pkl name, so an interrupted run resumes cleanly
    return env.get_num_agents()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, required=True, help="number of scenarios")
    parser.add_argument("--scene", choices=SCENES, default="scene_1")
    parser.add_argument("--n-agents", type=int, nargs="+", default=[10], help="fixed count, or LO HI (inclusive)")
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0, help="scenario i uses seed + i")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if len(args.n_agents) not in (1, 2):
        raise SystemExit("--n-agents takes one number or a LO HI range")
    n_agents = args.n_agents[0] if len(args.n_agents) == 1 else tuple(args.n_agents)

    args.out.mkdir(parents=True, exist_ok=True)
    spec_path = args.out / "pool.json"
    spec = {"scene": args.scene, "n_agents": args.n_agents, "line_length": args.line_length, "seed": args.seed}
    if spec_path.exists():
        previous = json.loads(spec_path.read_text())["spec"]
        if previous != spec:
            raise SystemExit(f"{args.out} was built with {previous}, not {spec}; use another --out")

    width = len(str(args.n - 1))
    jobs = [(str(args.out / f"{i:0{width}d}.pkl"), args.scene, n_agents, args.line_length, args.seed + i)
            for i in range(args.n)]
    todo = [j for j in jobs if not Path(j[0]).exists()]
    print(f"{args.out}: {len(jobs) - len(todo)} of {len(jobs)} scenarios already exist; building {len(todo)} "
          f"with {args.workers} worker(s)")

    start = time.time()
    if args.workers > 1:
        with multiprocessing.get_context("spawn").Pool(args.workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(build_one, todo), 1):
                print(f"\r  {i}/{len(todo)} built", end="", flush=True)
    else:
        for i, job in enumerate(todo, 1):
            build_one(job)
            print(f"\r  {i}/{len(todo)} built", end="", flush=True)
    print(f"  ({time.time() - start:.0f} s)")

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    spec_path.write_text(json.dumps({
        "spec": spec,
        "n": args.n,
        "command": " ".join(["python", *sys.argv]) if argv is None else f"main({argv})",
        "git_commit": commit,
        "flatland_rl": importlib.metadata.version("flatland-rl"),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
