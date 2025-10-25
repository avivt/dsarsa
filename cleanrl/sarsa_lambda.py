# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import os
import random
import time
from dataclasses import dataclass

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
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    # -------- TD(lambda) controls --------
    lambda_: float = 0.9
    """TD(lambda) mixing parameter in [0,1). Use 0 for pure n=1; 0.9 is common."""
    lambda_horizon: int = 5
    """K: maximum lookahead length for the truncated forward-view λ-return (e.g., 3–10)."""


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


# ---------- Robust RB helpers ----------
def _rb_capacity(rb) -> int:
    for name in ("size", "max_size", "buffer_size", "capacity"):
        if hasattr(rb, name):
            val = getattr(rb, name)
            if callable(val):
                continue
            try:
                return int(val)
            except Exception:
                pass
    if hasattr(rb, "observations"):
        return int(len(rb.observations))
    raise RuntimeError("Cannot determine ReplayBuffer capacity")


def _rb_effective_filled_size(rb) -> int:
    cap = _rb_capacity(rb)
    full = getattr(rb, "full", False)
    if isinstance(full, (bool, np.bool_)) and full:
        return cap
    for name in ("pos", "ptr", "_next_idx"):
        if hasattr(rb, name):
            try:
                return int(getattr(rb, name))
            except Exception:
                pass
    if hasattr(rb, "size"):
        s = getattr(rb, "size")
        if callable(s):
            try:
                return int(s())
            except Exception:
                pass
        else:
            try:
                return int(s)
            except Exception:
                pass
    return cap


def _sample_indices(rb, batch_size: int) -> np.ndarray:
    filled = _rb_effective_filled_size(rb)
    assert int(filled) > 0, "ReplayBuffer is empty"
    return np.random.randint(0, int(filled), size=batch_size)
# ----------------------------------------------------------------------


# ---------- TD(λ) builder for STEO-DQN (forward-view, truncated) ----------
@torch.no_grad()
def build_lambda_targets_STEO(
    rb,
    start_idx_batch: torch.Tensor,   # [B]
    K: int,
    gamma: float,
    lambda_: float,
    q_net: nn.Module,
    target_net: nn.Module,
    device: torch.device,
):
    """
    Build forward-view truncated TD(λ) targets for STEO-DQN:
      select with TARGET: a* = argmax_a Q_target(s_{t+n}, a)
      evaluate with ONLINE: Q_online(s_{t+n}, a*)
    Using contiguous fragments from the ring buffer; stop early at episode boundary.
    Returns:
      td_target: [B]
      obs_t:     [B, obs_dim]
      act_t:     [B, 1]
    """
    B = start_idx_batch.shape[0]
    idx0 = start_idx_batch.detach().cpu().numpy()

    obs_arr   = rb.observations
    next_arr  = rb.next_observations
    act_arr   = rb.actions
    rew_arr   = rb.rewards
    done_arr  = rb.dones

    cap = _rb_capacity(rb)
    K = max(1, int(K))
    gamma = float(gamma)
    lambda_ = float(lambda_)

    # 1) Collect contiguous indices up to K without crossing episodes
    traj_idxs = []
    for b in range(B):
        start = int(idx0[b])
        seq = [start]
        for j in range(1, K):
            cur = seq[-1]
            d_cur = bool(np.asarray(done_arr[cur]).reshape(-1)[0])
            if d_cur:
                break
            nxt = (cur + 1) % cap
            if not np.array_equal(next_arr[cur], obs_arr[nxt]):
                break
            seq.append(nxt)
        traj_idxs.append(seq)

    maxN = max(len(s) for s in traj_idxs)

    # 2) Cum rewards and bootstrap mask
    Rn = torch.zeros(B, maxN, device=device)
    bootstrap_ok = torch.zeros(B, maxN, dtype=torch.bool, device=device)
    for b, seq in enumerate(traj_idxs):
        G = 0.0
        pow_g = 1.0
        ok = True
        for n, idx in enumerate(seq):
            r = float(np.asarray(rew_arr[idx]).reshape(-1)[0])
            d = bool(np.asarray(done_arr[idx]).reshape(-1)[0])
            G += pow_g * r
            Rn[b, n] = G
            bootstrap_ok[b, n] = ok and (not d)
            pow_g *= gamma
            if d:
                ok = False

    # 3) Build s_{t+n} batch and evaluate Q(s_{t+n}, a*)
    s_boot_list, map_b, map_n = [], [], []
    for b, seq in enumerate(traj_idxs):
        for n, idx in enumerate(seq):
            s_boot_list.append(next_arr[idx])
            map_b.append(b)
            map_n.append(n)

    s_boot_np = np.array(s_boot_list)
    M = s_boot_np.shape[0]
    s_boot_tensor = torch.as_tensor(s_boot_np, dtype=torch.float32, device=device).view(M, -1)

    # forward through nets
    q_all  = q_net(s_boot_tensor)                              # [M, A]
    a_star = torch.argmax(target_net(s_boot_tensor), dim=1)    # [M]
    q_sel  = q_all[torch.arange(M, device=device), a_star]     # [M]

    # scatter back into [B, maxN]
    Qn = torch.zeros(B, maxN, device=device)
    b_idx = torch.as_tensor(map_b, dtype=torch.long, device=device)
    n_idx = torch.as_tensor(map_n, dtype=torch.long, device=device)
    Qn.index_put_((b_idx, n_idx), q_sel)

    # 4) y^{(n)} and λ-mix
    gammas = torch.tensor([gamma ** (n + 1) for n in range(maxN)], device=device).view(1, maxN)
    y_n = Rn + gammas * Qn * bootstrap_ok.float()

    lam_pows = torch.tensor([lambda_ ** n for n in range(maxN)], device=device).view(1, maxN)
    td_target = torch.zeros(B, device=device)
    for b, seq in enumerate(traj_idxs):
        L = len(seq)
        y = y_n[b, :L]
        if L == 1:
            td_target[b] = y[0]
        else:
            w = lam_pows[0, :L]
            td_target[b] = (1.0 - lambda_) * torch.sum(w[: L - 1] * y[: L - 1]) + (lambda_ ** (L - 1)) * y[L - 1]

    # 5) s_t and a_t, flattened
    obs_t = torch.as_tensor(obs_arr[idx0], dtype=torch.float32, device=device).view(B, -1)
    act_np = act_arr[idx0]
    if isinstance(act_np, np.ndarray) and act_np.ndim == 1:
        act_t = torch.as_tensor(act_np, dtype=torch.long, device=device).unsqueeze(1)
    else:
        act_t = torch.as_tensor(act_np, dtype=torch.long, device=device).view(B, 1)
    return td_target, obs_t, act_t
# ----------------------------------------------------------------------


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

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

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

    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", float(info["episode"]["r"]), global_step)
                    writer.add_scalar("charts/episodic_length", float(info["episode"]["l"]), global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        obs = next_obs

        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                idxs_np = _sample_indices(rb, args.batch_size)
                idxs = torch.as_tensor(idxs_np, dtype=torch.long, device=device)

                td_target, obs_t, act_t = build_lambda_targets_STEO(
                    rb,
                    idxs,
                    args.lambda_horizon,
                    args.gamma,
                    args.lambda_,
                    q_network,
                    target_network,
                    device,
                )

                q_sa = q_network(obs_t).gather(1, act_t).squeeze(1)
                loss = F.mse_loss(q_sa, td_target)

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", float(loss.item()), global_step)
                    writer.add_scalar("losses/q_values", float(q_sa.mean().item()), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
            writer.add_scalar("eval/episodic_return", float(episodic_return), idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "DQN", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    writer.close()
