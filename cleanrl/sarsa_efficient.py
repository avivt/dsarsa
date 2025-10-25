# sarsa_efficient.py
# STEO-DQN (target-select, online-evaluate) + fast truncated TD(lambda) on CPU
# Uses a Torch-native replay buffer to avoid NumPy↔Torch conversions in the hot path.

import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


# --------------------------- Small utilities ---------------------------

def _as_scalar(x):
    """Turn 0-d/1-d numpy values into a Python scalar safely."""
    return np.asarray(x).reshape(-1)[0].item()


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / max(1, duration)
    return float(max(end_e, start_e + slope * t))


# --------------------------- Torch-native segment replay buffer (CPU) ---------------------------

class TorchSegmentReplayBuffer:
    """
    Single-env ring buffer stored as Torch tensors on CPU.
    Precomputes link_ok[i] that is True iff transition i can advance to i+1 within the same episode
    (no terminal at i, and next_obs[i] == obs[i+1]). Enables building [B,K] contiguous segments
    with vectorized indexing, no Python loops per-sample.
    """
    def __init__(self, capacity: int, obs_space: gym.Space):
        assert isinstance(obs_space, gym.spaces.Box), "Expect Box observations"
        self.capacity = int(capacity)
        obs_shape = tuple(obs_space.shape)

        # All storage on CPU as torch tensors
        self.obs      = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32)
        self.next_obs = torch.zeros((self.capacity, *obs_shape), dtype=torch.float32)
        self.actions  = torch.zeros((self.capacity, 1),           dtype=torch.long)
        self.rewards  = torch.zeros((self.capacity,),             dtype=torch.float32)
        self.dones    = torch.zeros((self.capacity,),             dtype=torch.bool)
        self.link_ok  = torch.zeros((self.capacity,),             dtype=torch.bool)

        self.pos = 0
        self.full = False
        self.filled = 0

    def _filled(self) -> int:
        return self.capacity if self.full else self.filled

    def add(self, obs, next_obs, action, reward, done: bool):
        """
        Expects single-env transitions (from SyncVectorEnv):
          obs:      (1, *obs_shape)
          next_obs: (1, *obs_shape)
          action:   (1,) or (1,1)
          reward:   (1,)
          done:     bool (true terminal only)
        """
        i = self.pos
        # Convert the single transition to tensors (tiny copy), stay on CPU
        o  = torch.as_tensor(obs,      dtype=torch.float32)[0]
        no = torch.as_tensor(next_obs, dtype=torch.float32)[0]
        a  = torch.as_tensor(action,   dtype=torch.long).view(-1)[0]
        r  = torch.as_tensor(reward,   dtype=torch.float32).reshape(-1)[0]
        d  = bool(done)

        self.obs[i].copy_(o)
        self.next_obs[i].copy_(no)
        self.actions[i, 0] = a
        self.rewards[i] = r
        self.dones[i] = d
        self.link_ok[i] = False  # will be set for (i -> i+1) when the next slot is written

        # Update link_ok for (prev -> i) if prev exists
        if self._filled() > 0 or i > 0:
            prev = (i - 1) % self.capacity
            prev_is_init = self.full or prev < self.filled
            if prev_is_init:
                # valid if prev not terminal and next_obs[prev] == obs[i]
                self.link_ok[prev] = (not bool(self.dones[prev])) and torch.equal(self.next_obs[prev], self.obs[i])

        # Advance ring pointer & fill state
        if not self.full:
            self.filled += 1
            if self.filled == self.capacity:
                self.full = True
        self.pos = (self.pos + 1) % self.capacity

    def sample_start_indices(self, batch_size: int) -> torch.Tensor:
        filled = self._filled()
        assert filled > 0, "ReplayBuffer is empty"
        return torch.randint(0, filled, (batch_size,), dtype=torch.long)


# --------------------------- Agent & target logic ---------------------------

class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(int(np.prod(env.single_observation_space.shape)), 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)


@torch.no_grad()
def build_lambda_targets_STEO_fast(
    rb: TorchSegmentReplayBuffer,
    start_idx: torch.Tensor,     # [B] Long (CPU)
    K: int,
    gamma: float,
    lam: float,
    q_net: nn.Module,
    target_net: nn.Module,
):
    """
    Fully vectorized truncated forward-view TD(λ) for STEO-DQN (target-select, online-eval) on CPU.
    Returns: td_target [B], obs_t [B,obs], act_t [B,1]
    """
    B = start_idx.numel()
    start_idx = start_idx.to(dtype=torch.long)
    cap = rb.capacity
    K = int(K)
    assert K >= 1

    # [B,K] contiguous index grid
    offsets = torch.arange(K, dtype=torch.long)  # [K]
    idx_mat = (start_idx.unsqueeze(1) + offsets.unsqueeze(0)) % cap  # [B,K]

    # Valid mask via link_ok prefix (advanced indexing)
    link_steps = rb.link_ok[idx_mat[:, :-1]]                                   # [B,K-1]
    prefix_ok = torch.cumprod(link_steps.to(torch.int32), dim=1).to(torch.bool)  # [B,K-1]
    valid_mask = torch.cat([torch.ones(B, 1, dtype=torch.bool), prefix_ok], dim=1)  # [B,K]

    # Gather segments directly from CPU tensors
    obs_t  = rb.obs[start_idx].view(B, -1)                # [B, obs]
    act_t  = rb.actions[start_idx]                        # [B,1]
    rew    = rb.rewards[idx_mat]                          # [B,K]
    done   = rb.dones[idx_mat]                            # [B,K]
    next_o = rb.next_obs[idx_mat].view(B * K, -1)         # [B*K, obs]

    # Discounted cumulative rewards R^{(n)} with mask
    g_pows = (gamma ** torch.arange(0, K, dtype=torch.float32)).unsqueeze(0)  # [1,K]
    disc_rew = (rew * valid_mask.float()) * g_pows
    Rn = torch.cumsum(disc_rew, dim=1)  # [B,K]

    # Bootstrap: Q(s_{t+n+1}, a*) if alive & valid up to n
    alive_incl = torch.cumprod((~done & valid_mask).to(torch.int32), dim=1).to(torch.float32)  # [B,K]
    q_t_all = q_net(next_o)                 # [B*K, A]
    t_t_all = target_net(next_o)            # [B*K, A]
    a_star  = torch.argmax(t_t_all, dim=1, keepdim=True)            # [B*K,1]  (select with target)
    q_next  = q_t_all.gather(1, a_star).squeeze(1).view(B, K)       # [B,K]    (evaluate with online)

    gamma_next = (gamma ** torch.arange(1, K + 1, dtype=torch.float32)).unsqueeze(0)  # [1,K]
    y_n = Rn + alive_incl * gamma_next * q_next  # [B,K]

    # Truncated forward-view TD(λ)
    lengths = valid_mask.sum(dim=1)  # [B], each >=1
    if K > 1:
        n_idx = torch.arange(0, K - 1)
        weights = (1.0 - lam) * (lam ** n_idx).unsqueeze(0)               # [1,K-1]
        valid_base = (n_idx.unsqueeze(0) < (lengths - 1).unsqueeze(1))    # [B,K-1]
        base_sum = (y_n[:, :-1] * weights * valid_base.float()).sum(dim=1)
    else:
        base_sum = torch.zeros(B)

    last_pos = (lengths - 1).clamp(min=0)                  # [B]
    y_last = y_n.gather(1, last_pos.view(-1, 1)).squeeze(1)
    last_w = lam ** last_pos.to(torch.float32)

    td_target = base_sum + last_w * y_last
    return td_target, obs_t, act_t


# --------------------------- CLI & main loop ---------------------------

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = False  # force CPU on Mac by default
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: Optional[str] = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""

    # Algorithm
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500_000
    learning_rate: float = 2.5e-4
    num_envs: int = 1
    buffer_size: int = 100_000
    gamma: float = 0.99
    tau: float = 1.0
    target_network_frequency: int = 500
    batch_size: int = 256
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10_000
    train_frequency: int = 10

    # TD(λ)
    lambda_: float = 0.9
    lambda_horizon: int = 5  # K (3–10 recommended)


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


if __name__ == "__main__":
    # Let PyTorch use CPU cores effectively
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    torch.set_num_interop_threads(2)

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
            settings=wandb.Settings(start_method="thread"),
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # Force CPU on Mac (you can flip to MPS/CUDA by changing args.cuda)
    device = torch.device("cpu")

    # Env
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete)

    # Nets (CPU)
    q_network = QNetwork(envs).to(device)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    # Torch-native CPU replay buffer
    rb = TorchSegmentReplayBuffer(capacity=args.buffer_size, obs_space=envs.single_observation_space)

    start_time = time.time()

    # Rollout & Train
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # ε-greedy on ONLINE net
        eps = linear_schedule(
            args.start_e, args.end_e,
            int(args.total_timesteps * args.exploration_fraction),
            global_step
        )
        if random.random() < eps:
            act = np.array([envs.single_action_space.sample()])
        else:
            with torch.no_grad():
                q = q_network(torch.as_tensor(obs, dtype=torch.float32).view(1, -1))
                act = torch.argmax(q, dim=1).cpu().numpy()

        next_obs, rew, term, trunc, infos = envs.step(act)

        # episode stats
        if "final_info" in infos:
            for fi in infos["final_info"]:
                if fi and "episode" in fi:
                    ret = float(_as_scalar(fi["episode"]["r"]))
                    length = int(_as_scalar(fi["episode"]["l"]))
                    writer.add_scalar("charts/episodic_return", ret, global_step)
                    writer.add_scalar("charts/episodic_length", length, global_step)
                    print(f"step={global_step}  return={ret:.1f}  len={length}")

        # time-limit handling: keep bootstrapping
        real_next = next_obs.copy()
        for i, tr in enumerate(trunc):
            if tr:
                real_next[i] = infos["final_observation"][i]
        done_true = bool(term[0])  # true terminals only
        rb.add(obs, real_next, act, rew, done_true)

        obs = next_obs

        # Train
        if global_step >= args.learning_starts and (global_step % args.train_frequency == 0):
            start_idx = rb.sample_start_indices(args.batch_size)
            td_target, obs_t, act_t = build_lambda_targets_STEO_fast(
                rb, start_idx, args.lambda_horizon, args.gamma, args.lambda_, q_network, target_network
            )
            q_sa = q_network(obs_t).gather(1, act_t).squeeze(1)
            loss = F.mse_loss(q_sa, td_target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
            optimizer.step()

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", float(loss.item()), global_step)
                writer.add_scalar("losses/q_values", float(q_sa.mean().item()), global_step)
                elapsed = max(1e-6, time.time() - start_time)
                learned_steps = max(0, global_step - args.learning_starts)
                sps = int(learned_steps / elapsed)
                print(f"SPS={sps}")

        # periodic/soft target update
        if global_step % args.target_network_frequency == 0:
            with torch.no_grad():
                for tp, p in zip(target_network.parameters(), q_network.parameters()):
                    tp.data.lerp_(p.data, args.tau)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.pt"
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
