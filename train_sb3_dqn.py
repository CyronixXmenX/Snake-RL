"""
Stable-Baselines3 based DQN training for Snake RL.

This provides a simpler, well-tested alternative using SB3's DQN or QR-DQN.
Useful for baseline comparisons and rapid prototyping.

Features:
- Easy-to-use SB3 interface
- Support for DQN and QR-DQN (distributional)
- Vectorized environments
- TensorBoard logging
- GPU acceleration
"""

from __future__ import annotations

import argparse
import os
import warnings
from typing import Callable

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

try:
    from sb3_contrib import QRDQN
    QRDQN_AVAILABLE = True
except ImportError:
    QRDQN_AVAILABLE = False
    warnings.warn("sb3-contrib not available. QR-DQN disabled. Install with: pip install sb3-contrib")

from snake_env import SnakeEnv

# Import for custom feature extractor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn


class SmallCNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for small grids.
    
    SB3's default CNN is designed for Atari (84x84), which is too large
    for our small Snake grid (20x24). This extractor uses smaller kernels
    and stride to preserve spatial information.
    """
    
    def __init__(self, observation_space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


def make_env(rank: int, seed: int, **env_kwargs) -> Callable:
    """
    Create a function that instantiates a monitored Snake environment.
    
    Args:
        rank: Index of the environment
        seed: Base random seed
        **env_kwargs: Additional environment arguments
        
    Returns:
        Function that creates the environment
    """
    def _init():
        env = SnakeEnv(**env_kwargs)
        env.reset(seed=seed + rank)
        env = Monitor(env)  # Wrap with Monitor for automatic logging
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN/QR-DQN on Snake using Stable-Baselines3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm
    parser.add_argument("--algo", type=str, default="dqn", choices=["dqn", "qrdqn"],
                        help="Algorithm to use")
    
    # Environment
    parser.add_argument("--grid_w", type=int, default=24, help="Grid width")
    parser.add_argument("--grid_h", type=int, default=20, help="Grid height")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="Number of parallel environments")
    parser.add_argument("--vec_env_type", type=str, default="dummy", 
                        choices=["dummy", "subproc"],
                        help="Vectorized environment type (subproc for true parallelism)")
    
    # Training
    parser.add_argument("--total_timesteps", type=int, default=2_000_000,
                        help="Total training timesteps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for training")
    parser.add_argument("--buffer_size", type=int, default=1_000_000,
                        help="Replay buffer size")
    parser.add_argument("--learning_starts", type=int, default=50_000,
                        help="Steps before learning starts")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--target_update_interval", type=int, default=10_000,
                        help="Target network update interval")
    parser.add_argument("--train_freq", type=int, default=4,
                        help="Update frequency (in steps)")
    parser.add_argument("--gradient_steps", type=int, default=16,
                        help="Gradient steps per update")
    
    # Exploration
    parser.add_argument("--exploration_fraction", type=float, default=0.2,
                        help="Fraction of training for epsilon decay")
    parser.add_argument("--exploration_final_eps", type=float, default=0.01,
                        help="Final epsilon value")
    
    # Device
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="runs/sb3_dqn",
                        help="Log directory for TensorBoard")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Checkpoint directory")
    parser.add_argument("--save_freq", type=int, default=50_000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--eval_freq", type=int, default=25_000,
                        help="Evaluate every N steps")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                        help="Number of evaluation episodes")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Check algorithm availability
    if args.algo == "qrdqn" and not QRDQN_AVAILABLE:
        print("ERROR: QR-DQN requested but sb3-contrib not available.")
        print("Install with: pip install sb3-contrib")
        return
    
    print(f"\n{'='*70}")
    print(f"Stable-Baselines3 {args.algo.upper()} Training")
    print(f"{'='*70}\n")
    
    # Create directories
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Environment kwargs
    env_kwargs = {
        "grid_w": args.grid_w,
        "grid_h": args.grid_h,
        "step_penalty": -0.01,
        "food_reward": 1.0,
        "death_reward": -1.0,
        "distance_reward_scale": 0.1,
        "loop_penalty": -0.05,
        "exploration_reward_scale": 0.02,
        "loop_detection_window": 8,
        "render_mode": "none",
    }
    
    # Create vectorized environment
    print(f"Creating environment...")
    print(f"  Grid: {args.grid_w}x{args.grid_h}")
    print(f"  Parallel envs: {args.n_envs}")
    print(f"  Vec env type: {args.vec_env_type}")
    
    if args.n_envs == 1:
        env = DummyVecEnv([make_env(0, args.seed, **env_kwargs)])
        eval_env = DummyVecEnv([make_env(0, args.seed + 1000, **env_kwargs)])
    else:
        if args.vec_env_type == "subproc":
            # True parallel environments (separate processes)
            env = SubprocVecEnv([
                make_env(i, args.seed, **env_kwargs) 
                for i in range(args.n_envs)
            ])
            eval_env = SubprocVecEnv([
                make_env(i, args.seed + 1000, **env_kwargs)
                for i in range(min(args.n_envs, 4))  # Use fewer envs for eval
            ])
        else:
            # Sequential environments (single process)
            env = DummyVecEnv([
                make_env(i, args.seed, **env_kwargs) 
                for i in range(args.n_envs)
            ])
            eval_env = DummyVecEnv([
                make_env(i, args.seed + 1000, **env_kwargs)
                for i in range(min(args.n_envs, 4))
            ])
    
    print(f"✓ Environments created\n")
    
    # Model configuration
    print(f"Configuring {args.algo.upper()} model...")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Buffer size: {args.buffer_size:,}")
    print(f"  Learning starts: {args.learning_starts:,}")
    print(f"  Target update interval: {args.target_update_interval:,}")
    print(f"  Train freq: {args.train_freq}")
    print(f"  Gradient steps: {args.gradient_steps}")
    print(f"  Device: {args.device}")
    
    # Network architecture for SB3
    # Use custom CNN architecture suitable for small grids
    policy_kwargs = dict(
        features_extractor_class=SmallCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=[512],  # Additional hidden layers after feature extraction
        activation_fn=torch.nn.ReLU,
        normalize_images=False,  # Already normalized in our env
    )
    
    # Create model
    AlgoClass = QRDQN if args.algo == "qrdqn" else DQN
    
    model = AlgoClass(
        "CnnPolicy",  # Use CNN policy with custom feature extractor
        env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        device=args.device,
        seed=args.seed,
        verbose=1,
    )
    
    print(f"✓ Model created\n")
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),  # Adjust for n_envs
        save_path=args.checkpoint_dir,
        name_prefix=f"sb3_{args.algo}_snake",
        save_replay_buffer=False,  # Save space
        save_vecnormalize=False,
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    
    callbacks = [checkpoint_callback, eval_callback]
    
    # Training
    print(f"Starting training for {args.total_timesteps:,} timesteps...")
    print(f"TensorBoard: tensorboard --logdir {args.log_dir}")
    print(f"{'='*70}\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, f"sb3_{args.algo}_snake_final")
    model.save(final_model_path)
    print(f"\n✓ Final model saved to: {final_model_path}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
