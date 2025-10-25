# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass
from collections import deque  # ← NEW

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 10000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 1000
    batch_size: int = 128
    start_e: float = 1
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10

    # ---------- NEW ----------
    n_step: int = 3
    """n-step return length; set to 1 to recover 1-step."""
    # -------------------------


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(np.array(env.single_observation_space.shape).prod(), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # ---------- NEW: n-step buffer (single env) ----------
    nstep_fifo = deque()  # holds tuples: (obs, act, rew, done, next_obs)
    gamma = args.gamma
    n_step = max(1, int(args.n_step))
    gamma_n = gamma ** n_step  # used when no terminal inside the window

    def push_nstep_if_ready(force_flush=False, infos=None):
        """
        If we have >= n_step steps OR force_flush (e.g., episode ended),
        aggregate 1 transition and push to replay buffer. Repeat until
        the window is smaller than required unless force_flush==False.
        """
        pushed = 0
        while (len(nstep_fifo) >= n_step) or (force_flush and len(nstep_fifo) > 0):
            # Aggregate from the head over up to n_step entries (or until first terminal)
            R = 0.0
            g = 1.0
            done_any = False
            s0, a0, _, _, _ = nstep_fifo[0]
            sN = nstep_fifo[-1][4]  # default next_obs after n steps
            steps_to_use = min(n_step, len(nstep_fifo))
            for k in range(steps_to_use):
                _, _, r_k, d_k, s_next_k = nstep_fifo[k]
                R += g * r_k
                g *= gamma
                if d_k and not done_any:
                    done_any = True
                    sN = s_next_k  # stop bootstrap at first true terminal
                    # after a true terminal we still finish the sum but gamma_n term will be zeroed by done_any
            # push aggregated transition
            # note: store 'done' as done_any (true if a real terminal occurred inside the window)
            rb.add(
                np.array([s0]),
                np.array([sN]),
                np.array([a0]),
                np.array([R], dtype=np.float32),
                np.array([done_any], dtype=np.bool_),
                infos if infos is not None else {},
            )
            pushed += 1
            # pop head and continue (sliding window)
            nstep_fifo.popleft()
            # if a true terminal was inside the window, the remaining tuples logically belong to next episode;
            # continuing to pop/aggregate will produce the correct flush since we are in force_flush mode at episode end
            if not force_flush:
                # only emit one aggregated transition per step during normal running
                break
        return pushed
    # -----------------------------------------------------

    # start the game
    obs, _ = envs.reset(seed=args.seed)
    effective_learning_starts = max(args.learning_starts, n_step)  # ← NEW

    for global_step in range(args.total_timesteps):
        # Action selection (ε-greedy on ONLINE net)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Step env
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Record episodic stats
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # Prepare next_obs for time-limit truncations (CleanRL pattern)
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # ---------- NEW: feed n-step FIFO instead of 1-step add ----------
        # We treat 'done' as TRUE terminal only (terminations), not truncations.
        # This matches the CleanRL choice with handle_timeout_termination=False.
        nstep_fifo.append((obs[0], actions[0], float(rewards[0]), bool(terminations[0]), real_next_obs[0]))
        # Try to push one aggregated transition if we have >= n_step
        push_nstep_if_ready(force_flush=False, infos=infos)
        # If an episode really terminated here, flush whatever remains (partial n)
        if terminations[0]:
            push_nstep_if_ready(force_flush=True, infos=infos)
            nstep_fifo.clear()
        # -----------------------------------------------------------------

        # CRUCIAL step easy to overlook
        obs = next_obs

        # Training
        if global_step > effective_learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)

                # ---------- CHANGED: n-step target for STEO-DQN ----------
                with torch.no_grad():
                    # Select with TARGET:
                    a_prime = target_network(data.next_observations).argmax(dim=1, keepdim=True)  # [B,1]
                # Evaluate with ONLINE:
                q_next_eval = q_network(data.next_observations).gather(1, a_prime).squeeze()      # [B]

                # If any true terminal occurred inside the n-step window (data.dones==1),
                # the bootstrap term must be zero (this matches our stored 'done_any').
                td_target = (
                    data.rewards.flatten()
                    + (gamma_n * (1 - data.dones.flatten()) * q_next_eval)
                )

                q_sa = q_network(data.observations).gather(1, data.actions).squeeze()
                loss = F.mse_loss(q_sa, td_target)  # Huber is a bit stabler than pure MSE
                # ---------------------------------------------------------

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", q_sa.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.dqn_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=args.end_e,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub
            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
