"""
Fast-first DQN training script for Snake RL.

Implements a DQN training loop optimized for rapid iteration (≤5 minutes by default)
with comprehensive logging and optional performance modes.

Key Features:
- Fast defaults: batch_size=256, gradient_steps=2, total_steps=50k, max_seconds=300
- Wall-clock timeout (max_seconds) and step limit (total_steps)
- CSV metrics + TensorBoard logging with timing instrumentation
- Pinned memory + non_blocking GPU transfers
- Optional GPU utilization monitoring
- Double DQN + Dueling architecture
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

import numpy as np
import torch
from tqdm import trange

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Install with: pip install tensorboard")

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig


class GPUMonitor:
    """Monitor GPU utilization using pynvml (optional)."""
    
    def __init__(self, device_index: int = 0):
        self.enabled = PYNVML_AVAILABLE and torch.cuda.is_available()
        self.device_index = device_index
        self.handle = None
        
        if self.enabled:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as e:
                warnings.warn(f"Failed to initialize GPU monitor: {e}")
                self.enabled = False
    
    def get_utilization(self) -> float:
        """Get GPU utilization percentage (0-100)."""
        if not self.enabled or self.handle is None:
            return 0.0
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return float(util.gpu)
        except Exception:
            return 0.0
    
    def __del__(self):
        if self.enabled and self.handle is not None:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class CSVLogger:
    """CSV logger for metrics with exact schema."""
    
    HEADER = [
        "step", "episodes", "episode_return_mean", "episode_length_mean",
        "steps_per_sec", "updates_per_sec", "samples_per_sec",
        "time_env_ms_per_step", "time_learn_ms_per_update",
        "replay_size", "epsilon", "loss_q", "td_error_mean",
        "gpu_util", "device", "batch_size", "gradient_steps",
        "n_envs", "n_step", "seed"
    ]
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file = open(filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.HEADER)
        self.writer.writeheader()
        self.file.flush()
    
    def log(self, row: Dict[str, Any]):
        """Write a row to CSV. Missing keys are left blank."""
        # Ensure all header fields are present
        full_row = {k: row.get(k, "") for k in self.HEADER}
        self.writer.writerow(full_row)
        self.file.flush()
    
    def close(self):
        self.file.close()


class Timer:
    """Simple timer for measuring elapsed time."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = None
        self.elapsed = 0.0
        self.count = 0
    
    def start(self):
        self.start_time = time.perf_counter()
    
    def stop(self):
        if self.start_time is not None:
            self.elapsed += time.perf_counter() - self.start_time
            self.count += 1
            self.start_time = None
    
    def average_ms(self) -> float:
        """Get average time in milliseconds."""
        if self.count == 0:
            return 0.0
        return (self.elapsed / self.count) * 1000.0
    
    def total_seconds(self) -> float:
        """Get total elapsed time in seconds."""
        return self.elapsed


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Calculate epsilon for epsilon-greedy exploration with linear decay."""
    if decay_steps <= 0:
        return end
    t = min(step / decay_steps, 1.0)
    return start + (end - start) * t


def main() -> None:
    """Main training loop with fast-first defaults and comprehensive logging."""
    parser = argparse.ArgumentParser(
        description="Fast-first DQN training for Snake RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Device and stopping conditions
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"],
                        help="Device to use (auto=prefer CUDA if available)")
    parser.add_argument("--total_steps", type=int, default=50000,
                        help="Total environment steps (stop condition 1)")
    parser.add_argument("--max_seconds", type=int, default=300,
                        help="Maximum wall-clock time in seconds (stop condition 2)")
    
    # Environment configuration
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel environments (default 1 for fast-first)")
    parser.add_argument("--grid_w", type=int, default=24,
                        help="Grid width")
    parser.add_argument("--grid_h", type=int, default=20,
                        help="Grid height")
    
    # DQN hyperparameters (fast-first defaults)
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for learning")
    parser.add_argument("--gradient_steps", type=int, default=2,
                        help="Gradient steps per training call")
    parser.add_argument("--n_step", type=int, default=1,
                        help="N-step returns (1=standard TD)")
    parser.add_argument("--train_freq", type=int, default=4,
                        help="Train every N env steps")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--buffer_size", type=int, default=100000,
                        help="Replay buffer size")
    parser.add_argument("--train_start", type=int, default=10000,
                        help="Start training after N steps")
    parser.add_argument("--target_update", type=int, default=10000,
                        help="Target network update interval")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden layer size")
    
    # Exploration
    parser.add_argument("--eps_start", type=float, default=1.0,
                        help="Initial epsilon")
    parser.add_argument("--eps_end", type=float, default=0.01,
                        help="Final epsilon")
    parser.add_argument("--eps_decay_steps", type=int, default=40000,
                        help="Epsilon decay duration")
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=1000,
                        help="Log metrics every N steps")
    parser.add_argument("--log_dir", type=str, default="runs",
                        help="Base directory for logs")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name (default: timestamp)")
    
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Optional optimizations (default OFF for fast-first)
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Enable automatic mixed precision (AMP)")
    parser.add_argument("--compile", action="store_true", default=False,
                        help="Enable torch.compile")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable detailed profiling")
    
    # Environment rewards
    parser.add_argument("--step_penalty", type=float, default=-0.01,
                        help="Penalty per step")
    parser.add_argument("--food_reward", type=float, default=1.0,
                        help="Reward for eating food")
    parser.add_argument("--death_reward", type=float, default=-1.0,
                        help="Penalty for dying")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.cuda.is_available():
            warnings.warn("CUDA is available but using CPU. Use --device cuda for faster training.")
    else:
        device = args.device
    
    # Setup logging directory
    if args.exp_name is None:
        exp_name = f"fast_{int(time.time())}"
    else:
        exp_name = args.exp_name
    
    run_dir = Path(args.log_dir) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup CSV logger
    csv_logger = CSVLogger(str(run_dir / "metrics.csv"))
    
    # Setup TensorBoard
    writer = None
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter(log_dir=str(run_dir))
    
    # Setup GPU monitoring
    gpu_monitor = GPUMonitor() if device == "cuda" else None
    
    # Print configuration
    print(f"{'='*70}")
    print(f"Fast-First DQN Training for Snake RL")
    print(f"{'='*70}")
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")
    print(f"  Environment: {args.grid_w}x{args.grid_h} grid, n_envs={args.n_envs}")
    print(f"\nDQN Settings (fast-first):")
    print(f"  batch_size={args.batch_size}, gradient_steps={args.gradient_steps}")
    print(f"  n_step={args.n_step}, train_freq={args.train_freq}")
    print(f"  Learning rate: {args.lr}")
    print(f"\nStopping Conditions:")
    print(f"  total_steps={args.total_steps} OR max_seconds={args.max_seconds}")
    print(f"\nLogging:")
    print(f"  Directory: {run_dir}")
    print(f"  Interval: every {args.log_interval} steps")
    print(f"\nOptimizations:")
    print(f"  AMP: {args.use_amp}, Compile: {args.compile}, Profile: {args.profile}")
    print(f"{'='*70}\n")
    
    # Create environment
    env = SnakeEnv(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        step_penalty=args.step_penalty,
        food_reward=args.food_reward,
        death_reward=args.death_reward,
        render_mode="none"
    )
    
    # Create DQN agent
    cfg = DQNConfig(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        train_start=args.train_start,
        device=device,
        dueling=True,  # Dueling DQN on by default
        double_dqn=True,  # Double DQN on by default
        hidden_size=args.hidden_size,
        n_step=args.n_step,
        use_amp=args.use_amp,
        pin_memory=(device == "cuda"),  # Use pinned memory for GPU
        gradient_accumulation_steps=1,
        compile_model=args.compile,
        gradient_steps=args.gradient_steps,
    )
    agent = DQNAgent(cfg)
    
    print(f"Agent initialized on device: {agent.device}")
    if agent.use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")
    print()
    
    # Training state
    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_len = 0
    
    returns = deque(maxlen=100)
    lengths = deque(maxlen=100)
    episode_count = 0
    
    # Timing
    env_timer = Timer()
    learn_timer = Timer()
    
    # Metrics for logging
    steps_in_interval = 0
    updates_in_interval = 0
    interval_start_time = time.perf_counter()
    
    # Wall-clock timeout
    training_start_time = time.perf_counter()
    
    # Training loop
    pbar = trange(args.total_steps, desc="Training", unit="step")
    
    for step in pbar:
        # Check wall-clock timeout
        elapsed = time.perf_counter() - training_start_time
        if elapsed >= args.max_seconds:
            print(f"\n⏱️  Reached max_seconds={args.max_seconds}. Stopping.")
            # Log final state before breaking
            if steps_in_interval > 0:
                interval_end_time = time.perf_counter()
                interval_duration = interval_end_time - interval_start_time
                
                steps_per_sec = steps_in_interval / interval_duration if interval_duration > 0 else 0.0
                updates_per_sec = updates_in_interval / interval_duration if interval_duration > 0 else 0.0
                samples_per_sec = updates_per_sec * args.batch_size
                
                time_env_ms = env_timer.average_ms()
                time_learn_ms = learn_timer.average_ms()
                
                gpu_util = gpu_monitor.get_utilization() if gpu_monitor else 0.0
                
                csv_row = {
                    "step": step,
                    "episodes": episode_count,
                    "episode_return_mean": np.mean(returns) if returns else "",
                    "episode_length_mean": np.mean(lengths) if lengths else "",
                    "steps_per_sec": f"{steps_per_sec:.2f}",
                    "updates_per_sec": f"{updates_per_sec:.2f}",
                    "samples_per_sec": f"{samples_per_sec:.2f}",
                    "time_env_ms_per_step": f"{time_env_ms:.4f}",
                    "time_learn_ms_per_update": f"{time_learn_ms:.4f}",
                    "replay_size": len(agent.replay),
                    "epsilon": f"{epsilon:.4f}",
                    "loss_q": f"{loss:.6f}" if loss is not None else "",
                    "td_error_mean": "",
                    "gpu_util": f"{gpu_util:.2f}" if gpu_util > 0 else "",
                    "device": device,
                    "batch_size": args.batch_size,
                    "gradient_steps": args.gradient_steps,
                    "n_envs": args.n_envs,
                    "n_step": args.n_step,
                    "seed": args.seed,
                }
                csv_logger.log(csv_row)
            break
        
        # Epsilon schedule
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        
        # Environment step
        env_timer.start()
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.push(obs, action, reward, next_obs, terminated)
        
        obs = next_obs
        ep_return += reward
        ep_len += 1
        env_timer.stop()
        
        steps_in_interval += 1
        
        if done:
            returns.append(ep_return)
            lengths.append(ep_len)
            episode_count += 1
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0
        
        # Training step
        loss = None
        if step % args.train_freq == 0 and agent.replay.size >= agent.train_start:
            learn_timer.start()
            # Perform gradient_steps updates
            for _ in range(args.gradient_steps):
                loss = agent.train_step()
            learn_timer.stop()
            updates_in_interval += args.gradient_steps
        
        # Update progress bar
        avg_ret = np.mean(returns) if returns else 0.0
        avg_len = int(np.mean(lengths)) if lengths else 0
        pbar.set_postfix(
            eps=f"{epsilon:.3f}",
            ret=f"{avg_ret:.2f}",
            len=avg_len,
            loss=f"{(loss or 0):.4f}",
        )
        
        # Logging
        if (step + 1) % args.log_interval == 0 or (step + 1) == args.total_steps:
            interval_end_time = time.perf_counter()
            interval_duration = interval_end_time - interval_start_time
            
            # Calculate metrics
            steps_per_sec = steps_in_interval / interval_duration if interval_duration > 0 else 0.0
            updates_per_sec = updates_in_interval / interval_duration if interval_duration > 0 else 0.0
            samples_per_sec = updates_per_sec * args.batch_size
            
            time_env_ms = env_timer.average_ms()
            time_learn_ms = learn_timer.average_ms()
            
            gpu_util = gpu_monitor.get_utilization() if gpu_monitor else 0.0
            
            # CSV logging
            csv_row = {
                "step": step + 1,
                "episodes": episode_count,
                "episode_return_mean": np.mean(returns) if returns else "",
                "episode_length_mean": np.mean(lengths) if lengths else "",
                "steps_per_sec": f"{steps_per_sec:.2f}",
                "updates_per_sec": f"{updates_per_sec:.2f}",
                "samples_per_sec": f"{samples_per_sec:.2f}",
                "time_env_ms_per_step": f"{time_env_ms:.4f}",
                "time_learn_ms_per_update": f"{time_learn_ms:.4f}",
                "replay_size": len(agent.replay),
                "epsilon": f"{epsilon:.4f}",
                "loss_q": f"{loss:.6f}" if loss is not None else "",
                "td_error_mean": "",  # Not tracked in current implementation
                "gpu_util": f"{gpu_util:.2f}" if gpu_util > 0 else "",
                "device": device,
                "batch_size": args.batch_size,
                "gradient_steps": args.gradient_steps,
                "n_envs": args.n_envs,
                "n_step": args.n_step,
                "seed": args.seed,
            }
            csv_logger.log(csv_row)
            
            # TensorBoard logging
            if writer:
                if returns:
                    writer.add_scalar("episode/return_mean", np.mean(returns), step + 1)
                    writer.add_scalar("episode/length_mean", np.mean(lengths), step + 1)
                
                writer.add_scalar("perf/steps_per_sec", steps_per_sec, step + 1)
                writer.add_scalar("perf/updates_per_sec", updates_per_sec, step + 1)
                writer.add_scalar("perf/samples_per_sec", samples_per_sec, step + 1)
                
                writer.add_scalar("time/env_ms_per_step", time_env_ms, step + 1)
                writer.add_scalar("time/learn_ms_per_update", time_learn_ms, step + 1)
                
                if loss is not None:
                    writer.add_scalar("loss/q", loss, step + 1)
                
                if gpu_util > 0:
                    writer.add_scalar("sys/gpu_util", gpu_util, step + 1)
            
            # Reset interval metrics
            steps_in_interval = 0
            updates_in_interval = 0
            env_timer.reset()
            learn_timer.reset()
            interval_start_time = time.perf_counter()
    
    # Cleanup
    env.close()
    csv_logger.close()
    if writer:
        writer.close()
    
    total_time = time.perf_counter() - training_start_time
    
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Total episodes: {episode_count}")
    if returns:
        print(f"Final avg return: {np.mean(returns):.2f}")
        print(f"Final avg length: {np.mean(lengths):.1f}")
    print(f"\nLogs saved to: {run_dir}")
    print(f"  - CSV metrics: {run_dir / 'metrics.csv'}")
    if TENSORBOARD_AVAILABLE:
        print(f"  - TensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
