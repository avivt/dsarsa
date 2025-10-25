#!/usr/bin/env python3
import argparse, math
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RET_TAG = "charts/episodic_return"

def parse_run_name(name: str):
    # Expect: {env_id}__{exp_name}__{seed}__{timestamp}
    parts = name.split("__")
    if len(parts) < 4:
        return None, None, None
    env_id = parts[0]
    exp_name = parts[1]
    try:
        seed = int(parts[2])
    except Exception:
        return None, None, None
    return env_id, exp_name, seed

def algo_from_exp(exp: str) -> str:
    e = (exp or "").lower()
    if "sarsa" in e: return "sarsa"
    if "dqn" in e:   return "dqn"
    return exp or "unknown"

def game_from_env(env_id: str) -> str:
    # MinAtar/Breakout-v0 -> Breakout
    if "MinAtar/" in env_id:
        rest = env_id.split("MinAtar/", 1)[1]
        return rest.split("-v")[0]
    return env_id

def load_returns(evdir: Path, tag=RET_TAG) -> List[Tuple[int,float]]:
    ea = EventAccumulator(str(evdir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return []
    if tag not in ea.Tags().get("scalars", []):
        return []
    xs = [(ev.step, float(ev.value)) for ev in ea.Scalars(tag)]
    xs.sort(key=lambda t: t[0])
    return xs

def last100_stats(series: List[Tuple[int,float]]):
    vals = [v for _, v in series]
    n = len(vals)
    if n == 0:
        return None, None, None, 0
    tail = vals[-100:] if n >= 100 else vals
    m = sum(tail) / len(tail)
    med = sorted(tail)[len(tail)//2]
    best = max(tail)
    return m, med, best, n

def mean_ci(values: List[float]):
    if not values:
        return (float("nan"), float("nan"))
    import statistics, math
    m = statistics.mean(values)
    if len(values) == 1:
        return (m, float("nan"))
    stdev = statistics.pstdev(values) if len(values) > 1 else 0.0
    se = stdev / math.sqrt(len(values))
    return (m, 1.96 * se)

def collect_runs(runs_root: Path):
    rows = []  # per-run rows
    for d in runs_root.rglob("*"):
        if not d.is_dir(): continue
        if not any("tfevents" in f.name for f in d.iterdir() if f.is_file()): continue

        # parse from leaf or parent
        env, exp, seed = parse_run_name(d.name)
        if env is None:
            env, exp, seed = parse_run_name(d.parent.name)
            if env is None: 
                continue

        series = load_returns(d)
        if not series:
            series = load_returns(d.parent)
            if not series: 
                continue

        algo = algo_from_exp(exp)
        game = game_from_env(env)
        m, med, best, num_eps = last100_stats(series)
        rows.append({
            "algo": algo, "env_id": env, "game": game, "seed": seed,
            "last100_mean": m, "last100_median": med, "last100_best": best, "num_episodes": num_eps,
            "log_dir": str(d)
        })
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs")
    ap.add_argument("--outdir", type=str, default="results_last100")
    args = ap.parse_args()

    runs_root = Path(args.runs_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows = collect_runs(runs_root)
    if not rows:
        print(f"No runs found under {runs_root}")
        return

    # write per-run CSV
    per_run_csv = outdir / "last100_per_run.csv"
    with per_run_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","game","seed","last100_mean","last100_median","last100_best","num_episodes","log_dir"])
        for r in rows:
            w.writerow([r["algo"], r["game"], r["seed"], r["last100_mean"], r["last100_median"], r["last100_best"], r["num_episodes"], r["log_dir"]])

    # aggregate by (algo, game)
    by_ag: Dict[Tuple[str,str], List[float]] = {}
    for r in rows:
        if r["last100_mean"] is not None:
            by_ag.setdefault((r["algo"], r["game"]), []).append(r["last100_mean"])

    summary_csv = outdir / "last100_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","game","n_seeds","mean_last100","ci95_halfwidth"])
        for (algo, game), vals in sorted(by_ag.items()):
            mean, hw = mean_ci(vals)
            w.writerow([algo, game, len(vals), mean, hw])

    print(f"Done.\nPer-run: {per_run_csv}\nSummary: {summary_csv}")

if __name__ == "__main__":
    main()
