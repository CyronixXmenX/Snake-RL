"""
GPU-optimized training script for DQN agent on Snake environment.

This version uses vectorized environments and multiple training steps per
environment step to maximize GPU utilization.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
from tqdm import trange

from snake_env import SnakeEnv
from vec_env import VectorizedSnakeEnv
from dqn_agent import DQNAgent, DQNConfig
from config_utils import load_config, merge_config_with_args, add_training_arguments
from logger_utils import setup_logger, TrainingLogger


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """
    Calculate epsilon for epsilon-greedy exploration with linear decay.
    
    Args:
        step: Current training step
        start: Initial epsilon value
        end: Final epsilon value
        decay_steps: Number of steps over which to decay
        
    Returns:
        Current epsilon value
    """
    if decay_steps <= 0:
        return end
    t = min(step / decay_steps, 1.0)
    return start + (end - start) * t


def evaluate(agent: DQNAgent, env: SnakeEnv, episodes: int = 5) -> float:
    """
    Evaluate agent performance without exploration.
    
    Args:
        agent: Trained DQN agent
        env: Snake environment instance
        episodes: Number of episodes to evaluate
        
    Returns:
        Average episode return
    """
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            done = terminated or truncated
        total += ret
    return total / episodes


def main() -> None:
    """Main training loop with GPU optimizations."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Snake environment (GPU-optimized)")
    add_training_arguments(parser)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (default: logs/training.log)")
    parser.add_argument("--no_console_log", action="store_true",
                        help="Disable console logging")
    parser.add_argument("--num_envs", type=int, default=8,
                        help="Number of parallel environments (default: 8)")
    parser.add_argument("--train_freq", type=int, default=4,
                        help="Number of training steps per environment step (default: 4)")
    args = parser.parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
        print(f"Loaded configuration from: {args.config}")
    
    # Setup logging
    log_file = args.log_file or os.path.join(args.checkpoint_dir, "training.log")
    logger = setup_logger(
        name="snake_rl",
        log_file=log_file,
        console=not args.no_console_log
    )
    train_logger = TrainingLogger(logger)
    
    # Log configuration
    train_logger.log_config({
        "Environment": {
            "grid_width": args.grid_w,
            "grid_height": args.grid_h,
            "step_penalty": args.step_penalty,
            "food_reward": args.food_reward,
            "death_reward": args.death_reward,
            "num_parallel_envs": args.num_envs,
        },
        "DQN": {
            "learning_rate": args.lr,
            "gamma": args.gamma,
            "batch_size": args.batch_size,
            "target_update": args.target_update,
            "buffer_size": args.buffer_size,
            "train_start": args.train_start,
        },
        "Training": {
            "total_steps": args.total_steps,
            "seed": args.seed,
            "device": args.device,
            "train_freq": args.train_freq,
        },
        "Exploration": {
            "epsilon_start": args.eps_start,
            "epsilon_end": args.eps_end,
            "epsilon_decay_steps": args.eps_decay_steps,
        },
        "GPU Optimization": {
            "use_amp": args.use_amp,
            "pin_memory": args.pin_memory,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
        }
    })

    # Create vectorized environments for parallel data collection
    vec_env = VectorizedSnakeEnv(
        num_envs=args.num_envs,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        step_penalty=args.step_penalty,
        food_reward=args.food_reward,
        death_reward=args.death_reward,
    )
    
    # Create single environment for evaluation
    eval_env = SnakeEnv(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        step_penalty=args.step_penalty,
        food_reward=args.food_reward,
        death_reward=args.death_reward,
        render_mode="none",
    )
    eval_env.reset(seed=args.seed)

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
        use_amp=args.use_amp,
        pin_memory=args.pin_memory,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    agent = DQNAgent(cfg)
    train_logger.log_training_start(args.total_steps, str(agent.device))
    
    logger.info(f"Using {args.num_envs} parallel environments")
    logger.info(f"Training {args.train_freq} steps per environment step")
    logger.info(f"Effective samples per iteration: {args.num_envs * args.train_freq}")

    # Setup checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_latest.pth")
    best_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_best.pth")

    # Training state
    observations = vec_env.reset(seed=args.seed)
    ep_returns = [0.0] * args.num_envs
    ep_lens = [0] * args.num_envs
    returns = deque(maxlen=100)
    lengths = deque(maxlen=100)
    best_avg_return = -float('inf')

    # Training loop
    t0 = time.time()
    total_env_steps = 0
    pbar = trange(args.total_steps, desc="Training", unit="step")
    
    for step in pbar:
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        
        # Collect experiences from all parallel environments
        # Use batch action inference for better GPU utilization
        actions = agent.act_batch(observations, epsilon=epsilon)
        next_observations, rewards, terminateds, truncateds, infos = vec_env.step(actions)
        
        # Store transitions for all environments
        for i in range(args.num_envs):
            agent.push(observations[i], actions[i], rewards[i], next_observations[i], terminateds[i])
            ep_returns[i] += rewards[i]
            ep_lens[i] += 1
            
            if terminateds[i] or truncateds[i]:
                returns.append(ep_returns[i])
                lengths.append(ep_lens[i])
                ep_returns[i] = 0.0
                ep_lens[i] = 0
        
        observations = next_observations
        total_env_steps += args.num_envs
        
        # Perform multiple training steps to keep GPU busy
        losses = []
        for _ in range(args.train_freq):
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
        
        avg_loss = np.mean(losses) if losses else 0.0

        # Update progress bar
        avg_ret = np.mean(returns) if returns else 0.0
        avg_len = int(np.mean(lengths)) if lengths else 0
        pbar.set_postfix(
            eps=f"{epsilon:.3f}", 
            avg_return=f"{avg_ret:.2f}", 
            avg_len=avg_len, 
            loss=f"{avg_loss:.4f}",
            env_steps=total_env_steps
        )

        # Periodic evaluation and checkpointing
        if (step + 1) % args.eval_interval == 0:
            eval_ret = evaluate(agent, eval_env, episodes=args.eval_episodes)
            train_logger.log_eval(step + 1, eval_ret, args.eval_episodes)
            if eval_ret > best_avg_return:
                best_avg_return = eval_ret
                agent.save(best_ckpt)
                train_logger.log_checkpoint(best_ckpt, is_best=True)
            else:
                agent.save(latest_ckpt)
                train_logger.log_checkpoint(latest_ckpt, is_best=False)

    # Final save
    agent.save(latest_ckpt)
    train_logger.log_checkpoint(latest_ckpt, is_best=False)
    vec_env.close()
    eval_env.close()
    
    dt = time.time() - t0
    logger.info(f"Total environment steps collected: {total_env_steps}")
    logger.info(f"Environment steps per second: {total_env_steps / dt:.2f}")
    train_logger.log_training_end(dt / 60, best_avg_return)


if __name__ == "__main__":
    main()
