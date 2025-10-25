#!/usr/bin/env python3
"""
Extract MinAtar benchmark results from TensorBoard logs (no re-runs needed).

It scans a TensorBoard `runs/` folder produced by your CleanRL scripts, reads
`charts/episodic_return` & `charts/episodic_length`, and writes two CSV files:
  - results_from_tb/runs_detailed_from_tb.csv
  - results_from_tb/summary_from_tb.csv

Assumptions:
- Run names follow CleanRL's pattern:
    {env_id}__{exp_name}__{seed}__{timestamp}
  e.g., "MinAtar/Breakout-v0__dqn__0__1729876543"
- `exp_name` usually equals the script basename (e.g., dqn, sarsa_efficient).
"""

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Tags we care about (as logged by your scripts)
RET_TAG = "charts/episodic_return"
LEN_TAG = "charts/episodic_length"

RUNNAME_RE = re.compile(r"^(?P<env>.+)__?(?P<exp>[^_]+)__?(?P<seed>\d+)__\d+$")

def parse_run_name(run_name: str):
    """
    Try to parse CleanRL-style run name:
      {env_id}__{exp_name}__{seed}__{timestamp}
    Returns (env_id, exp_name, seed) or (None, None, None) if no match.
    """
    m = RUNNAME_RE.match(run_name)
    if not m:
        return None, None, None
    env_id = m.group("env")
    exp = m.group("exp")
    seed = int(m.group("seed"))
    return env_id, exp, seed

def minatar_game_from_env_id(env_id: str) -> Optional[str]:
    # Expect "MinAtar/<Game>-v0"
    if not env_id:
        return None
    if "MinAtar/" in env_id:
        rest = env_id.split("MinAtar/")[-1]
        game = rest.split("-v")[0]
        return game
    return None

def load_scalars(evdir: Path) -> Tuple[List[Tuple[int,float]], List[Tuple[int,float]]]:
    """
    Return lists of (step, value) for return and length scalars from this event directory.
    If not found, returns empty lists.
    """
    # A single run dir may contain multiple event files; EventAccumulator handles that.
    ea = EventAccumulator(str(evdir), size_guidance={
        'scalars': 0,  # load all
    })
    try:
        ea.Reload()
    except Exception:
        return [], []

    returns = []
    lengths = []

    if RET_TAG in ea.Tags().get('scalars', []):
        for e in ea.Scalars(RET_TAG):
            returns.append((e.step, float(e.value)))

    if LEN_TAG in ea.Tags().get('scalars', []):
        for e in ea.Scalars(LEN_TAG):
            lengths.append((e.step, float(e.value)))

    # sort by step just in case
    returns.sort(key=lambda x: x[0])
    lengths.sort(key=lambda x: x[0])
    return returns, lengths

def guess_algo(exp_name: Optional[str]) -> Optional[str]:
    if not exp_name:
        return None
    exp = exp_name.lower()
    if "sarsa" in exp:
        return "sarsa"
    if "dqn" in exp:
        return "dqn"
    return exp_name  # fallback: return raw exp name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", type=str, default="runs", help="Root folder containing TB run subdirs")
    ap.add_argument("--outdir", type=str, default="results_from_tb", help="Where to write CSVs")
    args = ap.parse_args()

    runs_root = Path(args.runs_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    detailed_csv = outdir / "runs_detailed_from_tb.csv"
    summary_csv  = outdir / "summary_from_tb.csv"

    # Walk one level (TB writers typically create runs/<run_name>/events.out.tfevents...),
    # but also support deeper nesting just in case.
    run_dirs: List[Path] = []
    if runs_root.is_dir():
        # Gather leaf dirs that actually contain event files
        for p in runs_root.rglob("*"):
            if p.is_dir():
                # heuristic: contains an events file?
                if any("tfevents" in f.name for f in p.glob("*")):
                    run_dirs.append(p)

    if not run_dirs:
        print(f"No TensorBoard event files found under {runs_root}")
        return

    rows = []  # detailed rows
    # We'll aggregate by (algo, game, seed)
    agg: Dict[Tuple[str,str], List[float]] = {}

    for run_dir in sorted(run_dirs):
        run_name = run_dir.name  # typically the parent dir name is the run slug
        # Some TB setups put the events directly under runs/<slug>/; other times runs/<slug>/<hash>/
        # Attempt to find the topmost slug that matches CleanRL pattern
        env_id, exp_name, seed = parse_run_name(run_dir.parent.name)
        if env_id is None:
            env_id, exp_name, seed = parse_run_name(run_name)

        if env_id is None:
            # not a CleanRL-style run; skip gracefully
            continue

        algo = guess_algo(exp_name)
        game = minatar_game_from_env_id(env_id) or env_id

        returns, lengths = load_scalars(run_dir)
        if not returns:
            # Try the parent folder too (some writers put events at the run root)
            if run_dir != run_dir.parent:
                r2, l2 = load_scalars(run_dir.parent)
                if r2:
                    returns, lengths = r2, l2

        if not returns:
            # nothing to report for this run
            rows.append([algo, game, seed, None, None, None, None, 0, str(run_dir)])
            continue

        # best / last
        ret_values = [v for _, v in returns]
        best_ret = max(ret_values) if ret_values else None
        last_ret = ret_values[-1] if ret_values else None
        last_len = lengths[-1][1] if lengths else None

        rows.append([algo, game, seed, 0, best_ret, last_ret, last_len, len(ret_values), str(run_dir)])

        # aggregate for summary by (algo, game)
        key = (algo, game)
        if last_ret is not None:
            agg.setdefault(key, []).append(last_ret)

    # Write detailed CSV
    with detailed_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","game","seed","exit_code","best_return","last_return","last_length","total_returns_seen","log_dir"])
        for r in rows:
            w.writerow(r)

    # Write summary CSV
    with summary_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo","game","num_runs","mean_last_return","median_last_return","best_last_return"])
        for (algo, game), vals in sorted(agg.items()):
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            mean = sum(vals_sorted) / n if n else float("nan")
            median = vals_sorted[n//2] if n else float("nan")
            best = max(vals_sorted) if n else float("nan")
            w.writerow([algo, game, n, mean, median, best])

    print(f"Done.\nDetailed: {detailed_csv}\nSummary:  {summary_csv}")

if __name__ == "__main__":
    main()
