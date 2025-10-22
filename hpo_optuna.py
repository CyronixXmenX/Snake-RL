"""
Hyperparameter optimization for Snake DQN using Optuna.

Searches over learning rate, batch size, network architecture, n-step,
and other key hyperparameters to find optimal configuration.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Any

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

try:
    import torch
    from dqn_agent import DQNAgent, DQNConfig
    from snake_env import SnakeEnv
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    exit(1)


def objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        args: Command line arguments
        
    Returns:
        Mean evaluation return (to maximize)
    """
    # Sample hyperparameters
    lr = trial.suggest_float("learning_rate", 1e-5, 3e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024, 2048])
    hidden_size = trial.suggest_categorical("hidden_size", [256, 512, 768, 1024])
    n_step = trial.suggest_int("n_step", 1, 5)
    gradient_steps = trial.suggest_categorical("gradient_steps", [4, 8, 16, 32])
    target_update = trial.suggest_categorical("target_update", [5000, 10000, 20000])
    gamma = trial.suggest_float("gamma", 0.95, 0.999)
    
    # Exploration schedule
    eps_decay_fraction = trial.suggest_float("eps_decay_fraction", 0.1, 0.5)
    eps_final = trial.suggest_float("eps_final", 0.01, 0.1)
    
    # Reward shaping (optional)
    if args.optimize_rewards:
        distance_reward_scale = trial.suggest_float("distance_reward_scale", 0.0, 0.3)
        loop_penalty = trial.suggest_float("loop_penalty", -0.1, 0.0)
    else:
        distance_reward_scale = 0.1
        loop_penalty = -0.05
    
    # Create environment
    env = SnakeEnv(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        step_penalty=-0.01,
        food_reward=1.0,
        death_reward=-1.0,
        distance_reward_scale=distance_reward_scale,
        loop_penalty=loop_penalty,
        exploration_reward_scale=0.02,
        loop_detection_window=8,
        render_mode="none",
    )
    
    # Create agent with trial hyperparameters
    cfg = DQNConfig(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        lr=lr,
        gamma=gamma,
        batch_size=batch_size,
        target_update=target_update,
        buffer_size=args.buffer_size,
        train_start=args.train_start,
        device=args.device,
        dueling=True,
        hidden_size=hidden_size,
        n_step=n_step,
        gradient_steps=gradient_steps,
        use_amp=args.use_amp,
        compile_model=False,  # Disable for HPO speed
    )
    
    agent = DQNAgent(cfg)
    
    # Training parameters
    eps_decay_steps = int(args.trial_steps * eps_decay_fraction)
    
    # Training loop
    obs, _ = env.reset(seed=trial.number)
    ep_return = 0.0
    ep_len = 0
    returns = []
    
    for step in range(args.trial_steps):
        # Epsilon schedule
        epsilon = max(
            eps_final,
            1.0 - (1.0 - eps_final) * min(step / eps_decay_steps, 1.0)
        )
        
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        agent.push(obs, action, reward, next_obs, terminated)
        agent.train_step()
        
        obs = next_obs
        ep_return += reward
        ep_len += 1
        
        if done:
            returns.append(ep_return)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0
        
        # Report intermediate results for pruning
        if step > 0 and step % args.report_interval == 0:
            if len(returns) >= 5:
                intermediate_value = float(np.mean(returns[-10:]))
                trial.report(intermediate_value, step)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    env.close()
                    raise optuna.TrialPruned()
    
    env.close()
    
    # Return mean of last episodes as objective
    if len(returns) < 5:
        return -1000.0  # Failed to complete any episodes
    
    final_performance = float(np.mean(returns[-20:]))  # Average of last 20 episodes
    return final_performance


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter optimization for Snake DQN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # HPO settings
    parser.add_argument("--n_trials", type=int, default=30,
                        help="Number of Optuna trials")
    parser.add_argument("--trial_steps", type=int, default=200_000,
                        help="Training steps per trial")
    parser.add_argument("--n_jobs", type=int, default=1,
                        help="Number of parallel jobs (trials)")
    parser.add_argument("--study_name", type=str, default="snake_dqn_hpo",
                        help="Name of Optuna study")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (e.g., sqlite:///optuna.db)")
    
    # Environment
    parser.add_argument("--grid_w", type=int, default=24, help="Grid width")
    parser.add_argument("--grid_h", type=int, default=20, help="Grid height")
    
    # Fixed hyperparameters
    parser.add_argument("--buffer_size", type=int, default=200_000,
                        help="Replay buffer size (fixed)")
    parser.add_argument("--train_start", type=int, default=20_000,
                        help="Steps before training starts (fixed)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device to use")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use mixed precision (GPU only)")
    
    # Optimization scope
    parser.add_argument("--optimize_rewards", action="store_true",
                        help="Include reward shaping in HPO search space")
    
    # Pruning
    parser.add_argument("--report_interval", type=int, default=10_000,
                        help="Steps between intermediate reports for pruning")
    
    # Seed
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampler")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"Hyperparameter Optimization for Snake DQN")
    print(f"{'='*70}\n")
    print(f"Study name: {args.study_name}")
    print(f"Number of trials: {args.n_trials}")
    print(f"Steps per trial: {args.trial_steps:,}")
    print(f"Parallel jobs: {args.n_jobs}")
    print(f"Device: {args.device}")
    
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  → Auto-selected: {device}")
        args.device = device
    
    print(f"Optimize rewards: {args.optimize_rewards}")
    print(f"\nSearch space:")
    print(f"  learning_rate: [1e-5, 3e-4] (log scale)")
    print(f"  batch_size: [256, 512, 1024, 2048]")
    print(f"  hidden_size: [256, 512, 768, 1024]")
    print(f"  n_step: [1, 5]")
    print(f"  gradient_steps: [4, 8, 16, 32]")
    print(f"  target_update: [5000, 10000, 20000]")
    print(f"  gamma: [0.95, 0.999]")
    print(f"  eps_decay_fraction: [0.1, 0.5]")
    print(f"  eps_final: [0.01, 0.1]")
    
    if args.optimize_rewards:
        print(f"  distance_reward_scale: [0.0, 0.3]")
        print(f"  loop_penalty: [-0.1, 0.0]")
    
    print(f"\n{'='*70}\n")
    
    # Create study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=args.report_interval,
        interval_steps=args.report_interval,
    )
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",  # Maximize mean return
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    
    # Run optimization
    print("Starting optimization...\n")
    
    try:
        study.optimize(
            lambda trial: objective(trial, args),
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
    
    # Results
    print(f"\n{'='*70}")
    print("Optimization Results")
    print(f"{'='*70}\n")
    
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    
    if len(study.trials) > 0:
        print(f"\nBest trial:")
        trial = study.best_trial
        print(f"  Value (mean return): {trial.value:.3f}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save best params
        output_file = f"{args.study_name}_best_params.txt"
        with open(output_file, "w") as f:
            f.write(f"Best trial value: {trial.value:.3f}\n")
            f.write("\nBest hyperparameters:\n")
            for key, value in trial.params.items():
                f.write(f"{key}: {value}\n")
        
        print(f"\n✓ Best parameters saved to: {output_file}")
        
        # Optionally generate config file
        config_file = f"{args.study_name}_best_config.yaml"
        try:
            import yaml
            config = {
                "dqn": {
                    "learning_rate": trial.params.get("learning_rate", 1e-4),
                    "batch_size": trial.params.get("batch_size", 1024),
                    "hidden_size": trial.params.get("hidden_size", 512),
                    "n_step": trial.params.get("n_step", 3),
                    "gradient_steps": trial.params.get("gradient_steps", 16),
                    "target_update": trial.params.get("target_update", 10000),
                    "gamma": trial.params.get("gamma", 0.99),
                    "buffer_size": args.buffer_size,
                    "train_start": args.train_start,
                },
                "exploration": {
                    "epsilon_start": 1.0,
                    "epsilon_end": trial.params.get("eps_final", 0.01),
                    "epsilon_decay_steps": int(2_000_000 * trial.params.get("eps_decay_fraction", 0.2)),
                },
                "environment": {
                    "grid_width": args.grid_w,
                    "grid_height": args.grid_h,
                    "distance_reward_scale": trial.params.get("distance_reward_scale", 0.1),
                    "loop_penalty": trial.params.get("loop_penalty", -0.05),
                },
            }
            with open(config_file, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            print(f"✓ Best config saved to: {config_file}")
        except ImportError:
            print("  (yaml not available, skipping config file generation)")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
