#!/usr/bin/env python3
"""
Benchmark TD(lambda) STEO vs DQN on MinAtar (Gymnasium wrapper provided by 'minatar').

Usage:
  pip install minatar
  python benchmark_minatar.py \
    --cleanrl-root ./cleanrl \
    --algos sarsa,dqn \
    --games Asterix,Breakout,Freeway,Seaquest,SpaceInvaders \
    --seeds 0,1,2 \
    --total-timesteps 1000000 \
    --lambda-horizon 5 \
    --lambda_ 0.9 \
    --batch-size 256 \
    --train-frequency 10 \
    --target-network-frequency 5000 \
    --outdir results_minatar
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_GAMES = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]

# RET_LINE_RE = re.compile(r"return=([\-0-9\.eE]+)")
# LEN_LINE_RE = re.compile(r"len=([0-9]+)")
RET_LINE_RE = re.compile(r"(?:return|episodic_return)=([\-0-9\.eE]+)")
LEN_LINE_RE = re.compile(r"(?:len|episodic_length)=([0-9]+)")

def env_id_for_game(game: str) -> str:
    return f"MinAtar/{game}-v0"

def ensure_minatar_env_available(env_id: str) -> bool:
    try:
        import gymnasium as gym  # noqa: F401
        import minatar           # noqa: F401
        env = gym.make(env_id)
        try:
            env.reset(seed=0)
        finally:
            env.close()
        return True
    except Exception as e:
        sys.stderr.write(f"[warn] Could not create env '{env_id}': {e}\n")
        return False

def run_one(cmd, timeout=0):
    print("RUN:", " ".join(cmd), flush=True)
    last_ret = None
    last_len = None
    all_returns = []

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, text=True)
    start = time.time()
    try:
        for line in proc.stdout:
            print(line, end="")
            m_ret = RET_LINE_RE.search(line)
            if m_ret:
                try:
                    val = float(m_ret.group(1)); last_ret = val; all_returns.append(val)
                except Exception:
                    pass
            m_len = LEN_LINE_RE.search(line)
            if m_len:
                try:
                    last_len = int(m_len.group(1))
                except Exception:
                    pass
            if timeout and (time.time() - start) > timeout:
                proc.kill()
                return -9, last_ret, last_len, all_returns
    finally:
        proc.wait()
    return proc.returncode, last_ret, last_len, all_returns

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cleanrl-root", type=str, default="./cleanrl")
    p.add_argument("--algos", type=str, default="sarsa,dqn", help="Comma list: sarsa,dqn")
    p.add_argument("--games", type=str, default=",".join(DEFAULT_GAMES))
    p.add_argument("--seeds", type=str, default="0,1,2")
    p.add_argument("--total-timesteps", type=int, default=1_000_000)
    p.add_argument("--learning-starts", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--train-frequency", type=int, default=10)
    p.add_argument("--start-e", type=float, default=1.0)
    p.add_argument("--end-e", type=float, default=0.05)
    p.add_argument("--exploration-fraction", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--target-network-frequency", type=int, default=500)
    p.add_argument("--lambda-horizon", type=int, default=10)
    p.add_argument("--lambda_", type=float, default=0.9)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument("--outdir", type=str, default="results_minatar")
    p.add_argument("--timeout", type=int, default=0, help="Per-run timeout seconds (0 = no timeout)")
    return p.parse_args()

def main():
    args = parse_args()

    try:
        import minatar  # noqa: F401
    except Exception:
        print("ERROR: 'minatar' not installed. Install with:\n  pip install minatar", file=sys.stderr)
        sys.exit(1)

    algos = [x.strip().lower() for x in args.algos.split(",") if x.strip()]
    games = [x.strip() for x in args.games.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    cleanrl = Path(args.cleanrl_root).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    sarsa_script = cleanrl / "sarsa_efficient.py"
    dqn_script   = cleanrl / "dqn.py"

    if "sarsa" in algos and not sarsa_script.exists():
        print(f"ERROR: {sarsa_script} not found", file=sys.stderr); sys.exit(1)
    if "dqn"   in algos and not dqn_script.exists():
        print(f"ERROR: {dqn_script} not found", file=sys.stderr); sys.exit(1)

    valid_games = []
    for g in games:
        env_id = env_id_for_game(g)
        if ensure_minatar_env_available(env_id):
            valid_games.append(g)
        else:
            print(f"[skip] {env_id} not available; skipping", file=sys.stderr)
    if not valid_games:
        print("ERROR: No valid MinAtar envs available.", file=sys.stderr)
        sys.exit(1)

    per_run_csv = outdir / "runs_detailed.csv"
    summary_csv = outdir / "summary.csv"

    def bflag(name, val: bool):
        # Tyro-style toggles: --flag (true) / --no-flag (false)
        return f"--{name}" if val else f"--no-{name}"

    with per_run_csv.open("w", newline="") as f_runs, summary_csv.open("w", newline="") as f_sum:
        runs_writer = csv.writer(f_runs)
        sum_writer  = csv.writer(f_sum)
        runs_writer.writerow(["algo", "game", "seed", "exit_code", "best_return", "last_return", "last_length", "total_returns_seen"])
        sum_writer.writerow(["algo", "game", "seeds", "mean_last_return", "median_last_return", "best_last_return"])

        for algo in algos:
            for game in valid_games:
                env_id = env_id_for_game(game)
                per_game_last_returns = []

                for seed in seeds:
                    common = [
                        "--env-id", env_id,
                        "--seed", str(seed),
                        "--total-timesteps", str(args.total_timesteps),
                        "--learning_starts", str(args.learning_starts),
                        "--batch-size", str(args.batch_size),
                        "--train-frequency", str(args.train_frequency),
                        "--start-e", str(args.start_e),
                        "--end-e", str(args.end_e),
                        "--exploration-fraction", str(args.exploration_fraction),
                        "--gamma", str(args.gamma),
                        "--tau", str(args.tau),
                        bflag("cuda", False),
                        bflag("track", False),
                        bflag("capture-video", False),
                    ]

                    if algo == "sarsa":
                        cmd = [args.python, str(sarsa_script)] + common + [
                            "--lambda-horizon", str(args.lambda_horizon),
                            "--lambda_", str(args.lambda_),
                            "--target_network_frequency", str(10*args.target_network_frequency),
                        ]
                    elif algo == "dqn":
                        cmd = [args.python, str(dqn_script)] + common + [
                            "--target_network_frequency", str(args.target_network_frequency),
                        ]
                    else:
                        continue

                    exit_code, last_ret, last_len, all_rets = run_one(cmd, timeout=args.timeout)
                    best_ret = max(all_rets) if all_rets else None
                    per_game_last_returns.append(last_ret if last_ret is not None else float("-inf"))
                    runs_writer.writerow([algo, game, seed, exit_code, best_ret, last_ret, last_len, len(all_rets)])
                    f_runs.flush()

                vals = [v for v in per_game_last_returns if v is not None and v != float("-inf")]
                if len(vals) > 0:
                    mean_last = sum(vals) / len(vals)
                    med_last  = sorted(vals)[len(vals)//2]
                    best_last = max(vals)
                else:
                    mean_last = med_last = best_last = float("nan")
                sum_writer.writerow([algo, game, ",".join(map(str, seeds)), mean_last, med_last, best_last])
                f_sum.flush()

    print(f"\nFinished. CSVs written to:\n  {per_run_csv}\n  {summary_csv}")
    print("Tip: open the CSV in a notebook to plot aggregated returns by algo/game.")

if __name__ == "__main__":
    main()
