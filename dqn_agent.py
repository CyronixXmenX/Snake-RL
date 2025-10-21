"""
Deep Q-Network (DQN) Agent for Snake RL.

This module implements a DQN agent with experience replay, target networks,
and Double DQN for stable learning.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class DQNConfig:
    """Configuration for DQN agent and training."""
    
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
    # GPU optimization settings
    use_amp: bool = False  # Automatic Mixed Precision for faster GPU training
    pin_memory: bool = True  # Pin memory for faster data transfer to GPU
    gradient_accumulation_steps: int = 1  # Gradient accumulation for larger effective batch sizes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class QNetwork(nn.Module):
    """
    Convolutional Neural Network for Q-value approximation.
    
    Architecture:
    - 2 Conv2D layers for feature extraction
    - 2 Fully connected layers for Q-value estimation
    
    Args:
        in_channels: Number of input channels (3 for [head, body, food])
        num_actions: Number of possible actions (4 for up/down/left/right)
        grid_h: Height of the input grid
        grid_w: Width of the input grid
    """
    
    def __init__(self, in_channels: int, num_actions: int, grid_h: int, grid_w: int) -> None:
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
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Q-values for each action, shape (batch, num_actions)
        """
        x = self.features(x)
        x = x.view(x.size(0), -1)
        q = self.head(x)
        return q


class ReplayBuffer:
    """
    Experience replay buffer with memory-efficient uint8 storage.
    
    Stores transitions (state, action, reward, next_state, done) for training.
    Uses circular buffer for constant memory usage.
    
    Args:
        capacity: Maximum number of transitions to store
        obs_shape: Shape of observations (C, H, W)
    """
    
    def __init__(self, capacity: int, obs_shape: Tuple[int, int, int]) -> None:
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

    def push(
        self, 
        obs: np.ndarray, 
        action: int, 
        reward: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode terminated
        """
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of transitions uniformly.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with keys: obs, actions, rewards, next_obs, dones
        """
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = {
            "obs": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_obs": self.next_obs[idx],
            "dones": self.dones[idx],
        }
        return batch
    
    def __len__(self) -> int:
        """Return current size of the buffer."""
        return self.size


class DQNAgent:
    """
    Deep Q-Network agent with Double DQN and experience replay.
    
    Features:
    - Target network for stable learning
    - Experience replay buffer
    - Double DQN for reduced overestimation
    - Gradient clipping
    - Memory-efficient uint8 observation storage
    - GPU optimizations (mixed precision, pin memory, gradient accumulation)
    
    Args:
        cfg: Configuration object with hyperparameters
    """
    
    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.device = self._get_device(cfg.device)

        # Initialize networks
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
        
        # GPU optimizations
        self.use_amp = cfg.use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp) if self.use_amp else None
        self.pin_memory = cfg.pin_memory and self.device.type == "cuda"
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._accumulated_steps = 0
    
    def _get_device(self, device_cfg: str) -> torch.device:
        """Determine compute device based on configuration."""
        if device_cfg == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_cfg in ("cpu", "cuda"):
            return torch.device(device_cfg)
        else:
            return torch.device("cpu")

    @torch.no_grad()
    def act(self, obs: np.ndarray, epsilon: float) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            obs: Observation array of shape (C, H, W)
            epsilon: Exploration rate (0 = greedy, 1 = random)
            
        Returns:
            Selected action index
        """
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.cfg.num_actions)
        # obs: (C,H,W) uint8 -> float32 [0,1]
        obs_t = torch.from_numpy(obs).float().div(255.0).unsqueeze(0)
        if self.pin_memory:
            obs_t = obs_t.pin_memory()
        obs_t = obs_t.to(self.device, non_blocking=True)
        q_values = self.q(obs_t)
        action = int(q_values.argmax(dim=1).item())
        return action
    
    @torch.no_grad()
    def act_batch(self, observations: np.ndarray, epsilon: float) -> np.ndarray:
        """
        Select actions for a batch of observations using epsilon-greedy policy.
        
        This method is optimized for vectorized environments, processing multiple
        observations in parallel on the GPU for better utilization.
        
        Args:
            observations: Batch of observations, shape (batch_size, C, H, W)
            epsilon: Exploration rate (0 = greedy, 1 = random)
            
        Returns:
            Array of selected actions, shape (batch_size,)
        """
        batch_size = observations.shape[0]
        
        # Random exploration for each environment
        explore_mask = np.random.rand(batch_size) < epsilon
        actions = np.zeros(batch_size, dtype=np.int64)
        
        if explore_mask.all():
            # All random actions
            return np.random.randint(0, self.cfg.num_actions, size=batch_size)
        
        # Get greedy actions for non-exploring environments
        obs_t = torch.from_numpy(observations).float().div(255.0)
        if self.pin_memory:
            obs_t = obs_t.pin_memory()
        obs_t = obs_t.to(self.device, non_blocking=True)
        q_values = self.q(obs_t)
        greedy_actions = q_values.argmax(dim=1).cpu().numpy()
        
        # Combine random and greedy actions
        actions[explore_mask] = np.random.randint(0, self.cfg.num_actions, size=explore_mask.sum())
        actions[~explore_mask] = greedy_actions[~explore_mask]
        
        return actions

    def push(self, *args, **kwargs) -> None:
        """Add transition to replay buffer."""
        self.replay.push(*args, **kwargs)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (if buffer is sufficiently filled).
        
        Samples a batch from replay buffer, computes TD error using Double DQN,
        and updates the Q-network. Periodically updates target network.
        Supports mixed precision training and gradient accumulation.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if self.replay.size < self.train_start:
            return None

        batch = self.replay.sample(self.batch_size)
        
        # Convert to tensors with optional pin memory for faster transfer
        obs = torch.from_numpy(batch["obs"]).float().div(255.0)
        next_obs = torch.from_numpy(batch["next_obs"]).float().div(255.0)
        actions = torch.from_numpy(batch["actions"]).long()
        rewards = torch.from_numpy(batch["rewards"]).float()
        dones = torch.from_numpy(batch["dones"]).float()
        
        # Pin memory if enabled for faster GPU transfer
        if self.pin_memory:
            obs = obs.pin_memory()
            next_obs = next_obs.pin_memory()
            actions = actions.pin_memory()
            rewards = rewards.pin_memory()
            dones = dones.pin_memory()
        
        # Transfer to device (non-blocking if pinned memory)
        obs = obs.to(self.device, non_blocking=self.pin_memory)
        next_obs = next_obs.to(self.device, non_blocking=self.pin_memory)
        actions = actions.to(self.device, non_blocking=self.pin_memory)
        rewards = rewards.to(self.device, non_blocking=self.pin_memory)
        dones = dones.to(self.device, non_blocking=self.pin_memory)

        # Forward pass with optional automatic mixed precision
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
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
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps

        # Backward pass with optional gradient scaling for mixed precision
        if self.use_amp and self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        self._accumulated_steps += 1
        
        # Only step optimizer after accumulating gradients
        if self._accumulated_steps >= self.gradient_accumulation_steps:
            if self.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=10.0)
                self.optim.step()
            
            self.optim.zero_grad(set_to_none=True)
            self._accumulated_steps = 0

        self.train_steps += 1
        if self.train_steps % self.target_update == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return float(loss.item() * self.gradient_accumulation_steps)

    def save(self, path: str) -> None:
        """
        Save agent state to disk.
        
        Args:
            path: File path to save checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            "model": self.q.state_dict(),
            "target_model": self.target_q.state_dict(),
            "optimizer": self.optim.state_dict(),
            "config": self.cfg.to_dict(),
            "train_steps": self.train_steps,
        }
        if self.use_amp and self.scaler is not None:
            checkpoint["scaler"] = self.scaler.state_dict()
        torch.save(checkpoint, path)

    def load(self, path: str, strict: bool = True) -> None:
        """
        Load agent state from disk.
        
        Args:
            path: File path to load checkpoint from
            strict: Whether to strictly enforce state dict matching
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["model"], strict=strict)
        if "target_model" in ckpt:
            self.target_q.load_state_dict(ckpt["target_model"], strict=strict)
        else:
            self.target_q.load_state_dict(self.q.state_dict())
        if "optimizer" in ckpt:
            self.optim.load_state_dict(ckpt["optimizer"])
        if "train_steps" in ckpt:
            self.train_steps = ckpt["train_steps"]
        if "scaler" in ckpt and self.use_amp and self.scaler is not None:
            self.scaler.load_state_dict(ckpt["scaler"])