from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DQNConfig:
    grid_w: int
    grid_h: int
    in_channels: int = 3
    num_actions: int = 4
    lr: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    target_update: int = 1000
    buffer_size: int = 100_000
    train_start: int = 10_000
    device: str = "auto"  # "cpu" | "cuda" | "auto"


class QNetwork(nn.Module):
    def __init__(self, in_channels: int, num_actions: int, grid_h: int, grid_w: int):
        super().__init__()
        # Small CNN for small grids
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),  # HxW
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # HxW
            nn.ReLU(inplace=True),
        )
        # Compute flatten size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, grid_h, grid_w)
            flat = self.features(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        q = self.head(x)
        return q


class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int]):
        self.capacity = capacity
        self.obs_shape = obs_shape  # (C, H, W)
        self.ptr = 0
        self.size = 0

        # Store observations as uint8 to save memory
        self.obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.next_obs = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.bool_)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs[idx],
            actions=self.actions[idx],
            rewards=self.rewards[idx],
            next_obs=self.next_obs[idx],
            dones=self.dones[idx],
        )
        return batch


class DQNAgent:
    def __init__(self, cfg: DQNConfig):
        self.cfg = cfg
        self.device = (
            torch.device("cuda") if (cfg.device == "auto" and torch.cuda.is_available()) else
            torch.device(cfg.device) if cfg.device in ("cpu", "cuda") else
            torch.device("cpu")
        )

        self.q = QNetwork(cfg.in_channels, cfg.num_actions, cfg.grid_h, cfg.grid_w).to(self.device)
        self.target_q = QNetwork(cfg.in_channels, cfg.num_actions, cfg.grid_h, cfg.grid_w).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma

        self.replay = ReplayBuffer(cfg.buffer_size, (cfg.in_channels, cfg.grid_h, cfg.grid_w))
        self.batch_size = cfg.batch_size
        self.train_start = cfg.train_start
        self.target_update = cfg.target_update

        self.train_steps = 0

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float) -> int:
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.cfg.num_actions)
        # obs: (C,H,W) uint8 -> float32 [0,1]
        obs_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0).to(self.device)
        q_values = self.q(obs_t)
        action = int(q_values.argmax(dim=1).item())
        return action

    def push(self, *args, **kwargs):
        self.replay.push(*args, **kwargs)

    def train_step(self):
        if self.replay.size < self.train_start:
            return None

        batch = self.replay.sample(self.batch_size)
        obs = torch.from_numpy(batch["obs"]).float().div(255.0).to(self.device)          # (B,C,H,W)
        next_obs = torch.from_numpy(batch["next_obs"]).float().div(255.0).to(self.device)
        actions = torch.from_numpy(batch["actions"]).long().to(self.device)              # (B,)
        rewards = torch.from_numpy(batch["rewards"]).float().to(self.device)             # (B,)
        dones = torch.from_numpy(batch["dones"]).float().to(self.device)                 # (B,)

        # Current Q(s,a)
        q_values = self.q(obs)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: action selection by online net, evaluation by target net
            next_q_values = self.q(next_obs)
            next_actions = next_q_values.argmax(dim=1, keepdim=True)  # (B,1)
            next_target_q_values = self.target_q(next_obs)
            next_q = next_target_q_values.gather(1, next_actions).squeeze(1)
            target = rewards + (1.0 - dones) * self.gamma * next_q

        loss = nn.SmoothL1Loss()(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
        self.optim.step()

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return float(loss.item())

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "model": self.q.state_dict(),
                "target_model": self.target_q.state_dict(),
                "config": self.cfg.__dict__,
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        ckpt = torch.load(path, map_location=self.device)
        self.q.load_state_dict(ckpt["model"], strict=strict)
        if "target_model" in ckpt:
            self.target_q.load_state_dict(ckpt["target_model"], strict=strict)
        else:
            self.target_q.load_state_dict(self.q.state_dict())