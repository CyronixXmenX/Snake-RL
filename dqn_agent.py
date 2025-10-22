"""
Deep Q-Network (DQN) Agent for Snake RL.

This module implements a DQN agent with experience replay, target networks,
and Double DQN for stable learning.
"""

from __future__ import annotations

import os
import platform
import warnings
from collections import deque
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def _is_torch_compile_supported() -> bool:
    """
    Check if torch.compile is supported on the current platform.
    
    torch.compile requires Triton which is not available on Windows.
    
    Returns:
        True if torch.compile is available and supported, False otherwise.
    """
    # Check if torch.compile exists (PyTorch 2.0+)
    if not hasattr(torch, 'compile'):
        return False
    
    # Triton is not available on Windows
    if platform.system() == "Windows":
        return False
    
    # Check if triton is available
    try:
        import triton
        return True
    except ImportError:
        return False


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
    # DQN architecture options
    dueling: bool = True  # Use Dueling DQN architecture
    double_dqn: bool = True  # Use Double DQN (always enabled in current implementation)
    hidden_size: int = 256  # Size of hidden layers
    n_step: int = 1  # N-step returns (1 = standard TD)
    # GPU optimization settings
    use_amp: bool = False  # Automatic Mixed Precision for faster GPU training
    pin_memory: bool = False  # Pin memory for faster data transfer to GPU (may add overhead)
    gradient_accumulation_steps: int = 1  # Gradient accumulation for larger effective batch sizes
    compile_model: bool = False  # Use torch.compile for optimized execution (PyTorch 2.0+)
    gradient_steps: int = 1  # Number of gradient steps per train_step call
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class QNetwork(nn.Module):
    """
    Convolutional Neural Network for Q-value approximation.
    
    Architecture:
    - 2 Conv2D layers for feature extraction
    - 2 Fully connected layers for Q-value estimation
    - Optional Dueling architecture with value and advantage streams
    
    Args:
        in_channels: Number of input channels (3 for [head, body, food])
        num_actions: Number of possible actions (4 for up/down/left/right)
        grid_h: Height of the input grid
        grid_w: Width of the input grid
        dueling: Whether to use Dueling DQN architecture
        hidden_size: Size of hidden layers (default: 256)
    """
    
    def __init__(
        self, 
        in_channels: int, 
        num_actions: int, 
        grid_h: int, 
        grid_w: int,
        dueling: bool = True,
        hidden_size: int = 256
    ) -> None:
        super().__init__()
        self.dueling = dueling
        self.num_actions = num_actions
        
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
        
        if self.dueling:
            # Dueling architecture: separate value and advantage streams
            # Shared feature layer
            self.feature_layer = nn.Sequential(
                nn.Linear(flat, hidden_size),
                nn.ReLU(inplace=True),
            )
            
            # Value stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size // 2, 1)
            )
            
            # Advantage stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size // 2, num_actions)
            )
        else:
            # Standard DQN architecture
            self.head = nn.Sequential(
                nn.Linear(flat, hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_size, num_actions),
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
        
        if self.dueling:
            features = self.feature_layer(x)
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            # Using mean instead of max for stability
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
            return q
        else:
            q = self.head(x)
            return q


class ReplayBuffer:
    """
    Experience replay buffer with memory-efficient uint8 storage.
    
    Stores transitions (state, action, reward, next_state, done) for training.
    Uses circular buffer for constant memory usage.
    Supports n-step returns for improved learning.
    
    Args:
        capacity: Maximum number of transitions to store
        obs_shape: Shape of observations (C, H, W)
        device: Device for storing tensors ('cpu' or 'cuda')
        pin_memory: Whether to pin memory for faster GPU transfer
        n_step: Number of steps for n-step returns (1 = standard TD)
        gamma: Discount factor for n-step returns
    """
    
    def __init__(
        self, 
        capacity: int, 
        obs_shape: Tuple[int, int, int],
        device: Optional[torch.device] = None,
        pin_memory: bool = False,
        n_step: int = 1,
        gamma: float = 0.99
    ) -> None:
        self.capacity = capacity
        self.obs_shape = obs_shape  # (C, H, W)
        self.ptr = 0
        self.size = 0
        self.device = device or torch.device("cpu")
        self.use_gpu = self.device.type == "cuda"
        # Only pin memory if CUDA is available and we're on CPU
        self.pin_memory = pin_memory and not self.use_gpu and torch.cuda.is_available()
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer: deque = deque(maxlen=n_step)

        if self.use_gpu:
            # Store data directly on GPU for zero-copy sampling
            self.obs = torch.zeros((capacity,) + obs_shape, dtype=torch.uint8, device=self.device)
            self.next_obs = torch.zeros((capacity,) + obs_shape, dtype=torch.uint8, device=self.device)
            self.actions = torch.zeros((capacity,), dtype=torch.int64, device=self.device)
            self.rewards = torch.zeros((capacity,), dtype=torch.float32, device=self.device)
            self.dones = torch.zeros((capacity,), dtype=torch.bool, device=self.device)
        else:
            # Store on CPU (with optional pinning for faster transfer)
            if self.pin_memory:
                self.obs = torch.zeros((capacity,) + obs_shape, dtype=torch.uint8).pin_memory()
                self.next_obs = torch.zeros((capacity,) + obs_shape, dtype=torch.uint8).pin_memory()
                self.actions = torch.zeros((capacity,), dtype=torch.int64).pin_memory()
                self.rewards = torch.zeros((capacity,), dtype=torch.float32).pin_memory()
                self.dones = torch.zeros((capacity,), dtype=torch.bool).pin_memory()
            else:
                self.obs = torch.zeros((capacity,) + obs_shape, dtype=torch.uint8)
                self.next_obs = torch.zeros((capacity,) + obs_shape, dtype=torch.uint8)
                self.actions = torch.zeros((capacity,), dtype=torch.int64)
                self.rewards = torch.zeros((capacity,), dtype=torch.float32)
                self.dones = torch.zeros((capacity,), dtype=torch.bool)

    def push(
        self, 
        obs: np.ndarray, 
        action: int, 
        reward: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> None:
        """
        Add a transition to the buffer with n-step return support.
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode terminated
        """
        # Add to n-step buffer
        self.n_step_buffer.append((obs, action, reward, next_obs, done))
        
        # Only push to main buffer when we have n steps or episode ends
        if len(self.n_step_buffer) < self.n_step and not done:
            return
        
        # Compute n-step return
        n_step_obs, n_step_action = self.n_step_buffer[0][:2]
        n_step_reward = 0.0
        n_step_next_obs = next_obs
        n_step_done = done
        
        for i, (_, _, r, next_o, d) in enumerate(self.n_step_buffer):
            n_step_reward += (self.gamma ** i) * r
            if d:
                n_step_next_obs = next_o
                n_step_done = True
                break
        
        # Store the n-step transition
        if self.use_gpu:
            self.obs[self.ptr].copy_(torch.from_numpy(n_step_obs))
            self.next_obs[self.ptr].copy_(torch.from_numpy(n_step_next_obs))
        else:
            self.obs[self.ptr] = torch.from_numpy(n_step_obs)
            self.next_obs[self.ptr] = torch.from_numpy(n_step_next_obs)
        
        self.actions[self.ptr] = int(n_step_action)
        self.rewards[self.ptr] = float(n_step_reward)
        self.dones[self.ptr] = bool(n_step_done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        # Clear n-step buffer on episode end
        if done:
            self.n_step_buffer.clear()

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions uniformly.
        
        Returns tensors on the same device as the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Dictionary with keys: obs, actions, rewards, next_obs, dones (all torch.Tensor)
        """
        # Generate random indices on the same device for GPU efficiency
        if self.use_gpu:
            idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        else:
            idx = torch.randint(0, self.size, (batch_size,))
        
        # Index directly into tensors - zero copy on GPU, minimal overhead on CPU
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
        self.q = QNetwork(
            cfg.in_channels, 
            cfg.num_actions, 
            cfg.grid_h, 
            cfg.grid_w,
            dueling=cfg.dueling,
            hidden_size=cfg.hidden_size
        ).to(self.device)
        self.target_q = QNetwork(
            cfg.in_channels, 
            cfg.num_actions, 
            cfg.grid_h, 
            cfg.grid_w,
            dueling=cfg.dueling,
            hidden_size=cfg.hidden_size
        ).to(self.device)
        self.target_q.load_state_dict(self.q.state_dict())
        self.target_q.eval()

        # Compile models for optimized execution (PyTorch 2.0+)
        if cfg.compile_model:
            if not _is_torch_compile_supported():
                platform_name = platform.system()
                if platform_name == "Windows":
                    warnings.warn(
                        "torch.compile is not supported on Windows (requires Triton). "
                        "Falling back to eager mode. Training will still work but without "
                        "the ~20-30% speedup from torch.compile. "
                        "Set compile_model=false in your config to suppress this warning.",
                        UserWarning,
                        stacklevel=2
                    )
                else:
                    warnings.warn(
                        "torch.compile requested but Triton is not available. "
                        "Falling back to eager mode. Install triton for ~20-30% speedup: "
                        "pip install triton",
                        UserWarning,
                        stacklevel=2
                    )
            else:
                try:
                    self.q = torch.compile(self.q, mode="reduce-overhead")
                    self.target_q = torch.compile(self.target_q, mode="reduce-overhead")
                except Exception as e:
                    warnings.warn(
                        f"torch.compile failed: {e}. Falling back to eager mode.",
                        UserWarning,
                        stacklevel=2
                    )

        self.optim = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.gamma = cfg.gamma

        # Initialize GPU-optimized replay buffer
        buffer_device = self.device if self.device.type == "cuda" else torch.device("cpu")
        self.replay = ReplayBuffer(
            cfg.buffer_size, 
            (cfg.in_channels, cfg.grid_h, cfg.grid_w),
            device=buffer_device,
            pin_memory=cfg.pin_memory,
            n_step=cfg.n_step,
            gamma=cfg.gamma
        )
        self.batch_size = cfg.batch_size
        self.train_start = cfg.train_start
        self.target_update = cfg.target_update
        self.gradient_steps_per_call = cfg.gradient_steps

        self.train_steps = 0
        
        # GPU optimizations
        self.use_gpu = self.device.type == "cuda"
        self.use_amp = cfg.use_amp and self.use_gpu
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp) if self.use_amp else None
        self.pin_memory = cfg.pin_memory and self.use_gpu
        self.gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._accumulated_steps = 0
        
        # CUDA stream for async operations (only on GPU)
        self.stream = torch.cuda.Stream() if self.use_gpu else None
        
        # Loss function (reused for efficiency)
        self.loss_fn = nn.SmoothL1Loss()
        
        # Enable TF32 for faster training on Ampere+ GPUs
        if self.use_gpu and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Enable cuDNN benchmarking for faster convolutions
        if self.use_gpu:
            torch.backends.cudnn.benchmark = True
    
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
        
        # Convert to tensor and normalize efficiently
        # Use contiguous() for better memory access patterns
        obs_t = torch.from_numpy(obs).float().div_(255.0).unsqueeze(0).contiguous()
        obs_t = obs_t.to(self.device, non_blocking=True)
        
        q_values = self.q(obs_t)
        action = int(q_values.argmax(dim=1).item())
        return action
    
    def push(self, *args, **kwargs) -> None:
        """Add transition to replay buffer."""
        self.replay.push(*args, **kwargs)

    def train_step(self) -> Optional[float]:
        """
        Perform training steps (if buffer is sufficiently filled).
        
        Samples batches from replay buffer, computes TD error using Double DQN,
        and updates the Q-network. Performs multiple gradient steps per call.
        Periodically updates target network.
        Supports mixed precision training and gradient accumulation.
        
        Returns:
            Average loss value if training occurred, None otherwise
        """
        if self.replay.size < self.train_start:
            return None

        total_loss = 0.0
        
        # Perform multiple gradient steps per call for better GPU utilization
        for _ in range(self.gradient_steps_per_call):
            # Sample batch - already on correct device (GPU or CPU)
            batch = self.replay.sample(self.batch_size)
            
            # Convert uint8 observations to float32 [0, 1] - keep on same device
            # Use contiguous() to ensure efficient memory layout
            obs = batch["obs"].float().div_(255.0).contiguous()
            next_obs = batch["next_obs"].float().div_(255.0).contiguous()
            actions = batch["actions"]
            rewards = batch["rewards"]
            dones = batch["dones"].float()
            
            # If buffer is on CPU but model is on GPU, transfer now (async if pinned)
            if not self.use_gpu or self.replay.device.type == "cpu":
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
                    
                    # Adjust gamma for n-step returns (already incorporated in n-step reward computation)
                    # For n-step, the effective gamma is gamma^n
                    effective_gamma = self.gamma ** self.cfg.n_step
                    next_q = next_target_q_values.gather(1, next_actions).squeeze(1)
                    target = rewards + (1.0 - dones) * effective_gamma * next_q

                loss = self.loss_fn(q_sa, target)
                
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

            total_loss += float(loss.item() * self.gradient_accumulation_steps)

        return total_loss / self.gradient_steps_per_call

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