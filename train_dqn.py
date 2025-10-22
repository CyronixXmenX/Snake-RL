"""
Training script for DQN agent on Snake environment.

Trains a Deep Q-Network agent with configurable hyperparameters and
saves checkpoints periodically. Supports loading configuration from YAML files.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from typing import Tuple

import numpy as np
from tqdm import trange

from snake_env import SnakeEnv
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
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train DQN agent on Snake environment")
    add_training_arguments(parser)
    parser.add_argument("--log_file", type=str, default=None,
                        help="Path to log file (default: logs/training.log)")
    parser.add_argument("--no_console_log", action="store_true",
                        help="Disable console logging")
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
            "distance_reward_scale": args.distance_reward_scale,
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
            "compile_model": args.compile_model,
        }
    })

    # Create environment
    env = SnakeEnv(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        step_penalty=args.step_penalty,
        food_reward=args.food_reward,
        death_reward=args.death_reward,
        distance_reward_scale=args.distance_reward_scale,
        render_mode="none",
    )
    env.reset(seed=args.seed)

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
        compile_model=args.compile_model,
    )
    agent = DQNAgent(cfg)
    train_logger.log_training_start(args.total_steps, str(agent.device))

    # Setup checkpoints
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_latest.pth")
    best_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_best.pth")

    # Training state
    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_len = 0
    returns = deque(maxlen=100)
    lengths = deque(maxlen=100)
    best_avg_return = -float('inf')

    # Training loop
    t0 = time.time()
    pbar = trange(args.total_steps, desc="Training", unit="step")
    
    for step in pbar:
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.push(obs, action, reward, next_obs, terminated)
        loss = agent.train_step()

        obs = next_obs
        ep_return += reward
        ep_len += 1

        if done:
            returns.append(ep_return)
            lengths.append(ep_len)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0

        # Update progress bar
        avg_ret = np.mean(returns) if returns else 0.0
        avg_len = int(np.mean(lengths)) if lengths else 0
        pbar.set_postfix(
            eps=f"{epsilon:.3f}", 
            avg_return=f"{avg_ret:.2f}", 
            avg_len=avg_len, 
            loss=f"{(loss or 0):.4f}"
        )

        # Periodic evaluation and checkpointing
        if (step + 1) % args.eval_interval == 0:
            eval_ret = evaluate(agent, env, episodes=args.eval_episodes)
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
    env.close()
    
    dt = time.time() - t0
    train_logger.log_training_end(dt / 60, best_avg_return)


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


if __name__ == "__main__":
    main()