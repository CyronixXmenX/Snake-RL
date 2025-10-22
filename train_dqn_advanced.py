"""
Advanced training script for high-throughput DQN on Snake environment.

Features:
- Vectorized environments for higher throughput
- GPU profiling and utilization monitoring
- Multiple gradient steps per environment step
- Comprehensive logging (TensorBoard)
- Support for various DQN variants (Dueling, n-step)
- Performance metrics (FPS, GPU util, env/learner time split)
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Tuple, Optional
import warnings

import numpy as np
import torch
from tqdm import trange

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    warnings.warn("pynvml not available. GPU utilization monitoring disabled. Install with: pip install pynvml")

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    warnings.warn("TensorBoard not available. Logging disabled. Install with: pip install tensorboard")

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig
from config_utils import load_config, merge_config_with_args, add_training_arguments


class GPUMonitor:
    """Monitor GPU utilization using pynvml."""
    
    def __init__(self, device_index: int = 0):
        self.enabled = PYNVML_AVAILABLE and torch.cuda.is_available()
        self.device_index = device_index
        
        if self.enabled:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            except Exception as e:
                warnings.warn(f"Failed to initialize GPU monitor: {e}")
                self.enabled = False
    
    def get_utilization(self) -> Tuple[float, float]:
        """Get GPU and memory utilization percentages."""
        if not self.enabled:
            return 0.0, 0.0
        
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            return float(util.gpu), float(util.memory)
        except Exception:
            return 0.0, 0.0
    
    def __del__(self):
        if self.enabled:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


class VectorizedEnvWrapper:
    """
    Simple vectorized environment wrapper for multiple Snake environments.
    Runs environments sequentially but provides batched interface.
    For true parallelism, use gymnasium.vector.AsyncVectorEnv.
    """
    
    def __init__(self, env_fns, n_envs: int):
        self.envs = [env_fn() for env_fn in env_fns[:n_envs]]
        self.n_envs = n_envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
    
    def reset(self, seed: Optional[int] = None):
        obs_list = []
        info_list = []
        for i, env in enumerate(self.envs):
            env_seed = None if seed is None else seed + i
            obs, info = env.reset(seed=env_seed)
            obs_list.append(obs)
            info_list.append(info)
        return np.array(obs_list), info_list
    
    def step(self, actions):
        obs_list = []
        reward_list = []
        terminated_list = []
        truncated_list = []
        info_list = []
        
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(truncated)
            info_list.append(info)
        
        return (
            np.array(obs_list),
            np.array(reward_list),
            np.array(terminated_list),
            np.array(truncated_list),
            info_list
        )
    
    def close(self):
        for env in self.envs:
            env.close()


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Calculate epsilon for epsilon-greedy exploration with linear decay."""
    if decay_steps <= 0:
        return end
    t = min(step / decay_steps, 1.0)
    return start + (end - start) * t


def evaluate(agent: DQNAgent, env: SnakeEnv, episodes: int = 5) -> Tuple[float, float]:
    """
    Evaluate agent performance without exploration.
    
    Returns:
        Tuple of (average return, average length)
    """
    total_return = 0.0
    total_length = 0.0
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
        
        total_return += ep_return
        total_length += ep_length
    
    return total_return / episodes, total_length / episodes


def main() -> None:
    """Main training loop with profiling and monitoring."""
    parser = argparse.ArgumentParser(
        description="High-throughput DQN training for Snake RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Add existing training arguments
    add_training_arguments(parser)
    
    # Advanced training options
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--train_freq", type=int, default=4,
                        help="Train every N environment steps")
    parser.add_argument("--gradient_steps", type=int, default=1,
                        help="Number of gradient steps per training call")
    parser.add_argument("--n_step", type=int, default=1,
                        help="N-step returns (1=standard TD, 3-5 recommended)")
    parser.add_argument("--dueling", action="store_true", default=True,
                        help="Use Dueling DQN architecture")
    parser.add_argument("--no_dueling", action="store_false", dest="dueling",
                        help="Disable Dueling DQN")
    parser.add_argument("--hidden_size", type=int, default=512,
                        help="Hidden layer size")
    
    # Profiling and logging
    parser.add_argument("--log_dir", type=str, default="runs/dqn_baseline",
                        help="TensorBoard log directory")
    parser.add_argument("--profile", action="store_true",
                        help="Enable detailed profiling")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="Log metrics every N steps")
    
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
        print(f"✓ Loaded configuration from: {args.config}")
    
    # Setup TensorBoard logging
    writer = None
    if TENSORBOARD_AVAILABLE:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=args.log_dir)
        print(f"✓ TensorBoard logging enabled: {args.log_dir}")
        print(f"  Run: tensorboard --logdir={args.log_dir}")
    
    # Setup GPU monitoring
    gpu_monitor = None
    if args.device in ("cuda", "auto") and torch.cuda.is_available():
        gpu_monitor = GPUMonitor(device_index=0)
        if gpu_monitor.enabled:
            print("✓ GPU monitoring enabled")
    
    # Create vectorized environments
    print(f"\n{'='*60}")
    print(f"High-Throughput DQN Training")
    print(f"{'='*60}")
    
    def make_env():
        return SnakeEnv(
            grid_w=args.grid_w,
            grid_h=args.grid_h,
            step_penalty=args.step_penalty,
            food_reward=args.food_reward,
            death_reward=args.death_reward,
            distance_reward_scale=args.distance_reward_scale,
            loop_penalty=args.loop_penalty,
            exploration_reward_scale=args.exploration_reward_scale,
            loop_detection_window=args.loop_detection_window,
            render_mode="none",
        )
    
    if args.n_envs > 1:
        env = VectorizedEnvWrapper([make_env for _ in range(args.n_envs)], args.n_envs)
        print(f"✓ Using {args.n_envs} vectorized environments")
    else:
        env = make_env()
        print("✓ Using single environment")
    
    # Create evaluation environment
    eval_env = make_env()
    
    # Print configuration
    print(f"\nEnvironment: {args.grid_w}x{args.grid_h} grid")
    print(f"Device: {args.device}")
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  → Auto-selected: {device_name}")
    
    print(f"\nDQN Configuration:")
    print(f"  Architecture: {'Dueling' if args.dueling else 'Standard'} DQN")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Learning rate: {args.lr}")
    print(f"  N-step returns: {args.n_step}")
    print(f"  Gradient steps: {args.gradient_steps}")
    print(f"  Train frequency: every {args.train_freq} env steps")
    
    print(f"\nGPU Optimizations:")
    print(f"  Mixed precision (AMP): {args.use_amp}")
    print(f"  Pin memory: {args.pin_memory}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  torch.compile: {args.compile_model}")
    
    print(f"\nTraining:")
    print(f"  Total steps: {args.total_steps:,}")
    print(f"  Epsilon: {args.eps_start} → {args.eps_end} over {args.eps_decay_steps:,} steps")
    print(f"{'='*60}\n")
    
    # Create agent
    cfg = DQNConfig(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        train_start=args.train_start,
        device=args.device,
        dueling=args.dueling,
        hidden_size=args.hidden_size,
        n_step=args.n_step,
        use_amp=args.use_amp,
        pin_memory=args.pin_memory,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        compile_model=args.compile_model,
        gradient_steps=args.gradient_steps,
    )
    agent = DQNAgent(cfg)
    
    print(f"Agent initialized on device: {agent.device}")
    if agent.use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
    print()
    
    # Setup checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_latest.pth")
    best_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_best.pth")
    
    # Training state
    if args.n_envs > 1:
        obs, _ = env.reset(seed=args.seed)
        ep_returns = np.zeros(args.n_envs)
        ep_lengths = np.zeros(args.n_envs, dtype=int)
    else:
        obs, _ = env.reset(seed=args.seed)
        ep_return = 0.0
        ep_len = 0
    
    returns = deque(maxlen=100)
    lengths = deque(maxlen=100)
    best_avg_return = -float('inf')
    
    # Profiling metrics
    env_time = 0.0
    learner_time = 0.0
    env_steps = 0
    train_steps_count = 0
    
    # Training loop
    t_start = time.time()
    pbar = trange(args.total_steps, desc="Training", unit="step")
    
    for step in pbar:
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        
        # Environment step
        t_env_start = time.time()
        if args.n_envs > 1:
            actions = np.array([agent.act(o, epsilon=epsilon) for o in obs])
            next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
            dones = np.logical_or(terminateds, truncateds)
            
            # Push transitions to replay buffer
            for i in range(args.n_envs):
                agent.push(obs[i], actions[i], rewards[i], next_obs[i], terminateds[i])
                ep_returns[i] += rewards[i]
                ep_lengths[i] += 1
                
                if dones[i]:
                    returns.append(ep_returns[i])
                    lengths.append(ep_lengths[i])
                    ep_returns[i] = 0.0
                    ep_lengths[i] = 0
            
            obs = next_obs
            env_steps += args.n_envs
        else:
            action = agent.act(obs, epsilon=epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.push(obs, action, reward, next_obs, terminated)
            
            obs = next_obs
            ep_return += reward
            ep_len += 1
            env_steps += 1
            
            if done:
                returns.append(ep_return)
                lengths.append(ep_len)
                obs, _ = env.reset()
                ep_return = 0.0
                ep_len = 0
        
        env_time += time.time() - t_env_start
        
        # Training step
        loss = None
        if step % args.train_freq == 0 and agent.replay.size >= agent.train_start:
            t_learner_start = time.time()
            loss = agent.train_step()
            learner_time += time.time() - t_learner_start
            train_steps_count += 1
        
        # Update progress bar
        avg_ret = np.mean(returns) if returns else 0.0
        avg_len = int(np.mean(lengths)) if lengths else 0
        pbar.set_postfix(
            eps=f"{epsilon:.3f}",
            ret=f"{avg_ret:.2f}",
            len=avg_len,
            loss=f"{(loss or 0):.4f}",
            buf=f"{len(agent.replay)}/{agent.replay.capacity}"
        )
        
        # Logging
        if writer and (step + 1) % args.log_interval == 0:
            writer.add_scalar("train/epsilon", epsilon, step + 1)
            writer.add_scalar("train/buffer_size", len(agent.replay), step + 1)
            
            if returns:
                writer.add_scalar("train/return_mean", np.mean(returns), step + 1)
                writer.add_scalar("train/return_std", np.std(returns), step + 1)
                writer.add_scalar("train/length_mean", np.mean(lengths), step + 1)
            
            if loss is not None:
                writer.add_scalar("train/loss", loss, step + 1)
            
            # Performance metrics
            if env_steps > 0:
                elapsed = time.time() - t_start
                fps = env_steps / elapsed
                writer.add_scalar("perf/env_fps", fps, step + 1)
                writer.add_scalar("perf/env_time_pct", 100 * env_time / elapsed, step + 1)
                writer.add_scalar("perf/learner_time_pct", 100 * learner_time / elapsed, step + 1)
            
            # GPU utilization
            if gpu_monitor and gpu_monitor.enabled:
                gpu_util, mem_util = gpu_monitor.get_utilization()
                writer.add_scalar("perf/gpu_utilization", gpu_util, step + 1)
                writer.add_scalar("perf/gpu_memory_utilization", mem_util, step + 1)
        
        # Periodic evaluation
        if (step + 1) % args.eval_interval == 0:
            eval_return, eval_length = evaluate(agent, eval_env, episodes=args.eval_episodes)
            
            print(f"\n[Step {step + 1:,}] Eval: return={eval_return:.2f}, length={eval_length:.1f}")
            
            if writer:
                writer.add_scalar("eval/return", eval_return, step + 1)
                writer.add_scalar("eval/length", eval_length, step + 1)
            
            if eval_return > best_avg_return:
                best_avg_return = eval_return
                agent.save(best_ckpt)
                print(f"  ★ New best model saved: {best_ckpt}")
            else:
                agent.save(latest_ckpt)
                print(f"  Checkpoint saved: {latest_ckpt}")
    
    # Final save and summary
    agent.save(latest_ckpt)
    env.close()
    eval_env.close()
    
    total_time = time.time() - t_start
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best evaluation return: {best_avg_return:.2f}")
    
    if args.profile:
        print(f"\nPerformance Profile:")
        print(f"  Environment steps: {env_steps:,}")
        print(f"  Training steps: {train_steps_count:,}")
        print(f"  Environment FPS: {env_steps / total_time:.1f}")
        print(f"  Environment time: {env_time:.1f}s ({100*env_time/total_time:.1f}%)")
        print(f"  Learner time: {learner_time:.1f}s ({100*learner_time/total_time:.1f}%)")
        
        if gpu_monitor and gpu_monitor.enabled:
            final_gpu_util, final_mem_util = gpu_monitor.get_utilization()
            print(f"  Final GPU utilization: {final_gpu_util:.1f}%")
            print(f"  Final GPU memory: {final_mem_util:.1f}%")
    
    print(f"{'='*60}\n")
    
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
