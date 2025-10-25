#!/usr/bin/env python3
"""
Run SARSA-λ (sarsa_efficient.py) and DQN (dqn.py) on MinAtar with a common baseline config.

Baseline hyperparameters (per run):
  gamma=0.99
  learning_rate=2.5e-4
  batch_size=32
  buffer_size=100000
  target_network_frequency=1000
  train_frequency=1
  learning_starts=5000
  epsilon: 1.0 -> 0.1 over first 10% of total timesteps (exploration_fraction=0.1)

Usage example:
  python run_minatar_baselines.py \
    --cleanrl-root ./cleanrl \
    --seeds 0 1 2 \
    --total-timesteps 1000000 \
    --outdir results_minatar_rerun
"""

import argparse
import csv
import shlex
import subprocess
import sys
import time
from pathlib import Path

MINATAR_GAMES = ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]

def env_id_for(game: str) -> str:
    return f"MinAtar/{game}-v0"

def ensure_env(env_id: str) -> bool:
    try:
        import gymnasium as gym  # noqa: F401
        import minatar           # noqa: F401  # registers MinAtar envs
        env = gym.make(env_id)
        try:
            env.reset(seed=0)
        finally:
            env.close()
        return True
    except Exception as e:
        sys.stderr.write(f"[warn] cannot create env '{env_id}': {e}\n")
        return False

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cleanrl-root", type=str, default="./cleanrl", help="Path to your cleanrl repo root")
    ap.add_argument("--games", type=str, nargs="*", default=MINATAR_GAMES, help="MinAtar games to run")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2], help="Seeds to run")
    ap.add_argument("--total-timesteps", type=int, default=1_000_000)
    ap.add_argument("--outdir", type=str, default="results_minatar_rerun")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--timeout", type=int, default=0, help="Kill a run after N seconds (0 = no timeout)")
    return ap.parse_args()

def main():
    args = parse_args()

    # Require minatar
    try:
        import minatar  # noqa: F401
    except Exception:
        print("ERROR: 'minatar' not installed. Install with: pip install minatar", file=sys.stderr)
        sys.exit(1)

    cleanrl = Path(args.cleanrl_root).resolve()
    sarsa_script = cleanrl / "sarsa_efficient.py"
    dqn_script   = cleanrl / "dqn.py"
    if not sarsa_script.exists():
        print(f"ERROR: {sarsa_script} not found", file=sys.stderr); sys.exit(1)
    if not dqn_script.exists():
        print(f"ERROR: {dqn_script} not found", file=sys.stderr); sys.exit(1)

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    manifest_csv = outdir / "manifest_commands.csv"

    # Common MinAtar baseline config (applied to BOTH algos)
    gamma = 0.99
    learning_rate = 2.5e-4
    batch_size = 32
    buffer_size = 100_000
    target_network_frequency = 1000
    train_frequency = 1
    learning_starts = 5000
    start_e = 1.0
    end_e = 0.1
    exploration_fraction = 0.1  # decay over 10% of total steps

    # Validate envs
    games = []
    for g in args.games:
        env_id = env_id_for(g)
        if ensure_env(env_id):
            games.append(g)
        else:
            print(f"[skip] {env_id} not available; skipping", file=sys.stderr)
    if not games:
        print("ERROR: no valid MinAtar envs found.", file=sys.stderr)
        sys.exit(1)

    # Helper: Tyro booleans are toggles --flag / --no-flag
    def bflag(name: str, val: bool) -> str:
        return f"--{name}" if val else f"--no-{name}"

    # Record commands we executed
    with manifest_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["algo", "game", "seed", "cmdline"])

        for game in games:
            env_id = env_id_for(game)
            for seed in args.seeds:
                # Common args for both scripts
                common = [
                    "--env-id", env_id,
                    "--seed", str(seed),
                    "--total-timesteps", str(args.total_timesteps),
                    "--learning_starts", str(learning_starts),
                    "--batch-size", str(batch_size),
                    "--buffer_size", str(buffer_size),
                    "--train-frequency", str(train_frequency),
                    "--start-e", str(start_e),
                    "--end-e", str(end_e),
                    "--exploration-fraction", str(exploration_fraction),
                    "--gamma", str(gamma),
                    "--learning_rate", str(learning_rate),
                    "--target_network_frequency", str(target_network_frequency),
                    bflag("cuda", False),
                    bflag("track", False),
                    bflag("capture-video", False),
                ]

                # ----- DQN -----
                dqn_cmd = [args.python, str(dqn_script)] + common + [
                    "--target_network_frequency", "1000",
                ]
                w.writerow(["dqn", game, seed, " ".join(shlex.quote(x) for x in dqn_cmd)])
                print("\nRUN DQN:", " ".join(dqn_cmd), flush=True)
                dqn_proc = subprocess.Popen(dqn_cmd)
                try:
                    dqn_proc.wait(timeout=args.timeout if args.timeout > 0 else None)
                except subprocess.TimeoutExpired:
                    dqn_proc.kill()

                # ----- SARSA-λ -----
                sarsa_cmd = [args.python, str(sarsa_script)] + common + [
                    "--lambda-horizon", "10",   # keep a moderate λ horizon
                    "--target_network_frequency", "5000",
                    "--lambda_", "0.9",
                ]
                w.writerow(["sarsa", game, seed, " ".join(shlex.quote(x) for x in sarsa_cmd)])
                print("\nRUN SARSA:", " ".join(sarsa_cmd), flush=True)
                sarsa_proc = subprocess.Popen(sarsa_cmd)
                try:
                    sarsa_proc.wait(timeout=args.timeout if args.timeout > 0 else None)
                except subprocess.TimeoutExpired:
                    sarsa_proc.kill()

    print(f"\nAll runs launched sequentially. Manifest saved to:\n  {manifest_csv}")
    print("Tip: after training finishes, use your TB extractors/plotters to rebuild tables and figures.")
    print("     (tb_last100.py and plot_minatar_from_tb.py)")
    
if __name__ == "__main__":
    main()
