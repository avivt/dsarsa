#!/usr/bin/env python3
"""
Plot MinAtar results from TensorBoard logs: one figure per game,
comparing DQN vs SARSA with mean ± 95% CI over seeds.

Assumptions:
- Run name pattern (from CleanRL): {env_id}__{exp_name}__{seed}__{timestamp}
  e.g., "MinAtar/Breakout-v0__dqn__0__1729876543"
- Episodic returns logged under tag "charts/episodic_return"
- We detect algos by substring: exp_name contains "dqn" -> DQN, "sarsa" -> SARSA

Outputs:
- <outdir>/<Game>.png
- <outdir>/<Game>_agg.csv  (grid of steps with mean/ci per algo)
"""

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RET_TAG = "charts/episodic_return"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs", help="Root with TensorBoard run subdirs")
    ap.add_argument("--outdir", type=str, default="plots_minatar", help="Where to save figures & CSVs")
    ap.add_argument("--grid-step", type=int, default=10_000, help="Step spacing for the common x-grid")
    ap.add_argument("--minatar-prefix", type=str, default="MinAtar/", help="Prefix for MinAtar env ids")
    ap.add_argument("--tag", type=str, default=RET_TAG, help="TB scalar tag to read (episodic returns)")
    ap.add_argument("--dpi", type=int, default=180)
    return ap.parse_args()

def parse_run_name(run_name: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Return (env_id, exp_name, seed) from "{env_id}__{exp_name}__{seed}__{ts}".
    Be tolerant to extra underscores inside env_id.
    """
    parts = run_name.split("__")
    if len(parts) < 4:
        return None, None, None
    env_id = "__".join(parts[:-3+1]) if len(parts) > 4 else parts[0]  # tolerant if env_id had "__"
    if len(parts) >= 4:
        env_id = parts[0]
        exp_name = parts[1]
        try:
            seed = int(parts[2])
        except Exception:
            return None, None, None
        return env_id, exp_name, seed
    return None, None, None

def algo_from_exp(exp_name: str) -> Optional[str]:
    e = exp_name.lower()
    if "sarsa" in e:
        return "sarsa"
    if "dqn" in e:
        return "dqn"
    return None

def game_from_env(env_id: str, prefix="MinAtar/") -> Optional[str]:
    if prefix in env_id:
        rest = env_id.split(prefix, 1)[1]
        return rest.split("-v")[0]
    return None

def load_returns(evdir: Path, tag: str) -> List[Tuple[int, float]]:
    """Load (step, value) list for the given tag from a TB event directory."""
    ea = EventAccumulator(str(evdir), size_guidance={"scalars": 0})
    try:
        ea.Reload()
    except Exception:
        return []
    if tag not in ea.Tags().get("scalars", []):
        return []
    vals = [(e.step, float(e.value)) for e in ea.Scalars(tag)]
    vals.sort(key=lambda x: x[0])
    return vals

def collect_runs(runs_root: Path, tag: str) -> List[Dict]:
    """
    Walk runs_root and collect runs that have the required tag.
    Returns a list of dicts with keys: env_id, game, exp_name, algo, seed, series [(step,ret)], log_dir.
    """
    results = []
    # Find any directory containing a tfevents file
    for d in runs_root.rglob("*"):
        if not d.is_dir():
            continue
        if not any("tfevents" in f.name for f in d.iterdir() if f.is_file()):
            continue

        # Try to parse name from leaf dir or its parent (some TB writers nest an extra subdir)
        candidates = [d.name, d.parent.name]
        env_id = exp = seed = None
        for name in candidates:
            e, ex, s = parse_run_name(name)
            if e is not None:
                env_id, exp, seed = e, ex, s
                break
        if env_id is None:
            continue

        algo = algo_from_exp(exp or "")
        game = game_from_env(env_id) or env_id

        series = load_returns(d, tag)
        if not series:
            # maybe events live one level up
            series = load_returns(d.parent, tag)
            if not series:
                continue

        results.append(dict(env_id=env_id, game=game, exp_name=exp, algo=algo, seed=seed,
                            series=series, log_dir=str(d)))
    return results

def build_grid_curves(series: List[Tuple[int,float]], grid: np.ndarray) -> np.ndarray:
    """
    Convert irregular (step, value) points into values on 'grid' by forward-fill of last observed return.
    If no observation yet, use NaN (we'll ignore in mean/CI computation).
    """
    steps = np.array([s for s,_ in series], dtype=np.int64)
    vals  = np.array([v for _,v in series], dtype=np.float32)
    out = np.full_like(grid, np.nan, dtype=np.float32)
    if steps.size == 0:
        return out
    j = 0
    last = np.nan
    for i, g in enumerate(grid):
        while j < len(steps) and steps[j] <= g:
            last = vals[j]
            j += 1
        out[i] = last
    return out

def mean_ci(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and 95% CI across runs per grid point, ignoring NaNs.
    Returns (mean, halfwidth) so you can plot mean ± halfwidth.
    """
    mean = np.nanmean(arr, axis=0)
    std = np.nanstd(arr, axis=0, ddof=1)
    n = np.sum(~np.isnan(arr), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        se = std / np.sqrt(np.maximum(n, 1))
        hw = 1.96 * se
        hw[np.isnan(mean)] = np.nan
    return mean, hw

def main():
    args = parse_args()
    runs_root = Path(args.runs_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    runs = collect_runs(runs_root, args.tag)
    if not runs:
        print(f"No runs found under {runs_root} with tag '{args.tag}'.")
        return

    # Group by game then algo
    by_game: Dict[str, Dict[str, List[Dict]]] = {}
    max_step_by_game: Dict[str, int] = {}
    for r in runs:
        game = r["game"]
        algo = r["algo"] or r["exp_name"] or "unknown"
        by_game.setdefault(game, {}).setdefault(algo, []).append(r)
        # track max step for grid
        if r["series"]:
            max_s = r["series"][-1][0]
            max_step_by_game[game] = max(max_step_by_game.get(game, 0), max_s)

    for game, algos in by_game.items():
        # Build a common grid per game
        max_step = max_step_by_game.get(game, 0)
        if max_step <= 0:
            continue
        grid = np.arange(0, max_step + 1, args.grid_step, dtype=np.int64)
        if grid.size < 2:  # ensure at least two points
            grid = np.array([0, max_step], dtype=np.int64)

        # aggregate curves for each algo
        agg_curves: Dict[str, np.ndarray] = {}
        per_algo_stack: Dict[str, np.ndarray] = {}

        for algo, runs_list in algos.items():
            curves = []
            for r in runs_list:
                gcurve = build_grid_curves(r["series"], grid)
                curves.append(gcurve)
            if not curves:
                continue
            stack = np.vstack(curves)  # [num_runs, T]
            per_algo_stack[algo] = stack
            mean, hw = mean_ci(stack)

            # Save CSV for this algo/game
            csv_path = outdir / f"{game}_{algo}_agg.csv"
            with open(csv_path, "w") as f:
                f.write("step,mean,lower,upper,n\n")
                n = np.sum(~np.isnan(stack), axis=0).astype(int)
                for s, m, h, nn in zip(grid, mean, hw, n):
                    lo = m - h if not math.isnan(m) else ""
                    up = m + h if not math.isnan(m) else ""
                    f.write(f"{s},{'' if math.isnan(m) else m},{lo},{up},{nn}\n")

        # Plot figure for this game (DQN vs SARSA if both exist)
        plt.figure()
        plotted_any = False
        legend_entries = []
        for label in ["dqn", "sarsa"]:
            if label not in per_algo_stack:
                continue
            stack = per_algo_stack[label]
            mean, hw = mean_ci(stack)
            plt.plot(grid, mean, label=label.upper())
            # shaded 95% CI
            lower = mean - hw
            upper = mean + hw
            plt.fill_between(grid, lower, upper, alpha=0.25, linewidth=0)
            plotted_any = True
            legend_entries.append(label.upper())

        if not plotted_any:
            plt.close()
            continue

        plt.xlabel("Environment steps")
        plt.ylabel("Episodic return")
        plt.title(f"{game} (MinAtar)")
        if legend_entries:
            plt.legend()
        plt.tight_layout()
        fig_path = outdir / f"{game}.png"
        plt.savefig(fig_path, dpi=args.dpi)
        plt.close()
        print(f"Saved {fig_path}")

    print(f"Done. Figures and CSVs at: {outdir}")

if __name__ == "__main__":
    main()
