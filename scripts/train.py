"""Train a shared policy for all trains with PPO.

critic: local  = IPPO (each train's critic sees only its own observation). Implemented.
critic: central = MAPPO (critic sees a summary of all trains). Member 3; see flatmarl.models.check_critic.

    python scripts/train.py --config configs/ippo_debug.yaml --seed 0

Writes runs/<name>_s<seed>/:
    config.yaml     the full config used (defaults filled in)
    run.json        command, git commit, versions
    train_log.csv   one row per iteration: official metrics of the (sampled) training games + PPO stats
                    (+ exempted_halts per game; return_mean/return_std when ppo.value_norm is on)
    eval_log.csv    quick check on the same fixed games each time, at iteration 0 (before training) and
                    at every save: train.eval_mode greedy (top action, the default) or sampled (actions
                    drawn from the policy, as evaluate.py actor-sampled:; the main evaluation since 26 Oct)
    actor.pt        latest actor, as {"model": state_dict}; also actor_<iteration>.pt snapshots
Evaluate a run:  python scripts/evaluate.py --policies actor:runs/<name>_s<seed>/actor.pt --out ...

Same config + seed gives identical results, whatever --workers is set to in the config.

Training budget (decided 26 Oct): train.iterations x train.episodes_per_iteration training games, the same
for IPPO and MAPPO. train_log.csv also records agent_steps (env steps x trains, summed over the iteration's
games) and decisions, with running totals, so the report can state both.
"""
import argparse
import copy
import csv
import functools
import importlib.metadata
import json
import math
import multiprocessing
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from flatmarl import ROOT
from flatmarl.envs import make_env, make_pool_env
from flatmarl.models import check_critic, make_actor, save_actor
from flatmarl.ppo import ValueNormalizer, make_batch, ppo_update
from flatmarl.reward import RewardShaper
from flatmarl.rollout import collect_episode

DEFAULTS = {
    "name": None,
    "critic": "local",
    "env": {"source": "make_env", "scene": "scene_1", "n_agents": 10, "line_length": 2,
            "disruption": "none", "pool": None},
    "reward": {"scale": 0.01, "progress": 0.0, "exempt_chosen_stops": False, "chosen_stop_cost": 0.0,
               "waiting_cost": 0.0},
    "ppo": {"gamma": 0.99, "gae_lambda": 0.95, "clip": 0.2, "epochs": 4, "minibatch_size": 512,
            "lr": 3e-4, "entropy_coef": 0.01, "value_coef": 0.5, "max_grad_norm": 0.5, "value_norm": False},
    "train": {"iterations": 50, "episodes_per_iteration": 16, "workers": 1, "init_checkpoint": None,
              "save_every": 10, "eval_episodes": 8, "eval_mode": "greedy"},
}
EVAL_SEED = 10**6  # checks play the games seeded EVAL_SEED, EVAL_SEED + 1, ... in every run
EVAL_MODES = ("greedy", "sampled")


def merge(defaults: dict, given: dict, where="config") -> dict:
    """Defaults overridden by `given`; unknown keys are an error, so a typo can't silently use a default."""
    unknown = set(given) - set(defaults)
    if unknown:
        raise ValueError(f"unknown {where} key(s): {sorted(unknown)}; allowed: {sorted(defaults)}")
    out = copy.deepcopy(defaults)
    for key, value in given.items():
        out[key] = merge(defaults[key], value, f"{where}.{key}") if isinstance(defaults[key], dict) else value
    return out


def load_config(path: Path) -> dict:
    cfg = merge(DEFAULTS, yaml.safe_load(path.read_text()) or {})
    if not cfg["name"]:
        raise ValueError("config needs a name")
    check_critic(cfg["critic"])
    if cfg["train"]["eval_mode"] not in EVAL_MODES:
        raise ValueError(f"train.eval_mode must be one of {EVAL_MODES}, got {cfg['train']['eval_mode']!r}")
    RewardShaper(**cfg["reward"])  # reject bad combinations now, not inside a worker
    return cfg


def env_recipe(env_cfg: dict):
    """Picklable env_fn(seed=...) for collect_episode."""
    if env_cfg["source"] == "make_env":
        n_agents = env_cfg["n_agents"]
        return functools.partial(make_env, scene=env_cfg["scene"], line_length=env_cfg["line_length"],
                                 disruption=env_cfg["disruption"],
                                 n_agents=tuple(n_agents) if isinstance(n_agents, list) else n_agents)
    if env_cfg["source"] == "pool":
        if not env_cfg["pool"]:
            raise ValueError("env.source: pool needs env.pool (a folder built by scripts/make_scenario_pool.py)")
        pool = Path(env_cfg["pool"])
        return functools.partial(make_pool_env, pool=pool if pool.is_absolute() else ROOT / pool,
                                 disruption=env_cfg["disruption"])
    raise ValueError(f"env.source must be make_env or pool, got {env_cfg['source']!r}")


def append_csv(path: Path, row: dict):
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def mean(values) -> float:
    values = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(values)) if values else math.nan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None, help="run folder (default runs/<name>_s<seed>)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    run_dir = args.out or ROOT / "runs" / f"{cfg['name']}_s{args.seed}"
    if (run_dir / "train_log.csv").exists() and not args.overwrite:
        raise SystemExit(f"{run_dir} already has a run; pass --overwrite to replace it")
    run_dir.mkdir(parents=True, exist_ok=True)
    for old in ("train_log.csv", "eval_log.csv"):  # --overwrite: start the logs fresh, don't append
        (run_dir / old).unlink(missing_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    git = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    (run_dir / "run.json").write_text(json.dumps({
        "command": " ".join(["python", *sys.argv]) if argv is None else f"main({argv})",
        "seed": args.seed,
        "git": {"commit": git("rev-parse", "HEAD"), "uncommitted_changes": bool(git("status", "--porcelain"))},
        "versions": {p: importlib.metadata.version(p) for p in ("flatland-rl", "torch")},
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")

    # One thread everywhere: small networks gain nothing from more, and it keeps results identical
    # between in-process and worker-process collection.
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    init = cfg["train"]["init_checkpoint"]  # like env.pool, a relative path is relative to the repo root
    actor = make_actor(None if init is None else (Path(init) if Path(init).is_absolute() else ROOT / init))
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg["ppo"]["lr"])
    recipe = env_recipe(cfg["env"])
    train, ppo = cfg["train"], cfg["ppo"]
    normalizer = ValueNormalizer() if ppo["value_norm"] else None

    workers = multiprocessing.get_context("spawn").Pool(train["workers"]) if train["workers"] > 1 else None
    log_path, decisions_total, agent_steps_total = run_dir / "train_log.csv", 0, 0
    greedy_eval = train["eval_mode"] == "greedy"

    def run_jobs(jobs):
        return workers.map(collect_episode, jobs) if workers else [collect_episode(j) for j in jobs]

    def quick_check(iteration):
        """The actor on the same fixed games every time, taking its top action (eval_mode greedy) or
        sampling (eval_mode sampled; the draws are seeded by the game seed, so reproducible)."""
        if not train["eval_episodes"]:
            return
        state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        jobs = [(recipe, state, EVAL_SEED + i, cfg["reward"], ppo["gamma"], greedy_eval)
                for i in range(train["eval_episodes"])]
        episodes = [r.metrics for r in run_jobs(jobs)]
        row = {"iteration": iteration,
               **{k: mean([e[k] for e in episodes]) for k in
                  ("score", "completion_rate", "cancelled_rate", "stop_service_rate", "halts_while_moving")}}
        append_csv(run_dir / "eval_log.csv", row)
        print(f"  {train['eval_mode']} check after {iteration} iterations: score {row['score']:.3f}  "
              f"arrived {row['completion_rate']:4.0%}  never departed {row['cancelled_rate']:4.0%}  "
              f"halts {row['halts_while_moving']:5.1f}")

    print(f"{cfg['name']} seed {args.seed}: {train['iterations']} iterations x {train['episodes_per_iteration']} "
          f"episodes, critic={cfg['critic']}, {train['workers']} worker(s) -> {run_dir}")
    try:
        quick_check(0)
        for iteration in range(1, train["iterations"] + 1):
            start = time.time()
            seeds = np.random.SeedSequence([args.seed, iteration]).generate_state(train["episodes_per_iteration"])
            state = {k: v.detach().clone() for k, v in actor.state_dict().items()}
            jobs = [(recipe, state, int(s), cfg["reward"], ppo["gamma"]) for s in seeds]
            results = run_jobs(jobs)

            batch = make_batch(results, ppo["gae_lambda"], normalizer)
            stats = ppo_update(actor, optimizer, batch, ppo, generator)
            decisions_total += len(batch.actions)
            agent_steps = sum(r.agent_steps for r in results)
            agent_steps_total += agent_steps

            episodes = [r.metrics for r in results]
            row = {
                "iteration": iteration,
                "decisions": len(batch.actions),
                "decisions_total": decisions_total,
                "agent_steps": agent_steps,
                "agent_steps_total": agent_steps_total,
                "score": mean([e["score"] for e in episodes]),
                "completion_rate": mean([e["completion_rate"] for e in episodes]),
                "stop_service_rate": mean([e["stop_service_rate"] for e in episodes]),
                "halts_while_moving": mean([e["halts_while_moving"] for e in episodes]),
                "exempted_halts": mean([r.exempted_halts for r in results]),
                "unattributed_reward": sum(r.unattributed for r in results),
                **stats,
                **({"return_mean": normalizer.mean, "return_std": normalizer.std} if normalizer else {}),
                "seconds": round(time.time() - start, 2),
            }
            append_csv(log_path, row)
            print(f"  it {iteration:4d}  score {row['score']:.3f}  arrived {row['completion_rate']:4.0%}  "
                  f"halts {row['halts_while_moving']:5.1f}  entropy {row['entropy']:.2f}  "
                  f"decisions {row['decisions']:6d}  {row['seconds']:5.1f}s")

            if iteration % train["save_every"] == 0 or iteration == train["iterations"]:
                save_actor(actor, run_dir / f"actor_{iteration}.pt")
                save_actor(actor, run_dir / "actor.pt")
                quick_check(iteration)
    finally:
        if workers:
            workers.close()
            workers.join()


if __name__ == "__main__":
    main()
