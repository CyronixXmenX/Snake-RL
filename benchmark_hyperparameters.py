"""
Hyperparameter Benchmark for Snake RL.

This script performs a brute-force grid search over various hyperparameter
combinations to find optimal settings for training. It runs short training
sessions with different configurations and reports the best-performing settings.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np
from tqdm import tqdm

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig


@dataclass
class BenchmarkConfig:
    """Configuration for hyperparameter benchmark."""
    
    # Benchmark settings
    benchmark_steps: int = 50000  # Steps per configuration
    eval_interval: int = 10000  # Evaluation frequency
    eval_episodes: int = 5  # Episodes per evaluation
    n_runs_per_config: int = 1  # Number of runs per configuration (for stability)
    
    # Grid search ranges
    learning_rates: List[float] = None
    batch_sizes: List[int] = None
    gamma_values: List[float] = None
    epsilon_decay_steps: List[int] = None
    distance_reward_scales: List[float] = None
    buffer_sizes: List[int] = None
    target_update_intervals: List[int] = None
    
    # Fixed parameters
    grid_w: int = 24
    grid_h: int = 20
    step_penalty: float = -0.01
    food_reward: float = 1.0
    death_reward: float = -1.0
    train_start: int = 5000
    device: str = "auto"
    seed_base: int = 42
    
    # Output settings
    output_dir: str = "benchmark_results"
    save_best_model: bool = True
    
    def __post_init__(self):
        """Set default values for grid search ranges."""
        if self.learning_rates is None:
            self.learning_rates = [0.0001, 0.0002, 0.0005]
        if self.batch_sizes is None:
            self.batch_sizes = [32, 64, 128]
        if self.gamma_values is None:
            self.gamma_values = [0.95, 0.99]
        if self.epsilon_decay_steps is None:
            self.epsilon_decay_steps = [10000, 20000, 30000]
        if self.distance_reward_scales is None:
            self.distance_reward_scales = [0.0, 0.05, 0.1, 0.2]
        if self.buffer_sizes is None:
            self.buffer_sizes = [50000]
        if self.target_update_intervals is None:
            self.target_update_intervals = [1000]


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    
    config: Dict[str, Any]
    final_avg_return: float
    best_eval_return: float
    training_time: float
    avg_episode_length: float
    total_episodes: int
    seed: int


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    """Calculate epsilon with linear decay."""
    if decay_steps <= 0:
        return end
    t = min(step / decay_steps, 1.0)
    return start + (end - start) * t


def evaluate(agent: DQNAgent, env: SnakeEnv, episodes: int = 5) -> Tuple[float, float]:
    """
    Evaluate agent performance.
    
    Returns:
        Tuple of (average_return, average_length)
    """
    total_return = 0.0
    total_length = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_length = 0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
        total_return += ep_return
        total_length += ep_length
    return total_return / episodes, total_length / episodes


def run_single_benchmark(
    hyperparams: Dict[str, Any],
    benchmark_cfg: BenchmarkConfig,
    seed: int,
    verbose: bool = False
) -> BenchmarkResult:
    """
    Run training with a specific hyperparameter configuration.
    
    Args:
        hyperparams: Dictionary of hyperparameters to test
        benchmark_cfg: Benchmark configuration
        seed: Random seed for this run
        verbose: Whether to show progress bar
        
    Returns:
        BenchmarkResult with training metrics
    """
    # Create environment
    env = SnakeEnv(
        grid_w=benchmark_cfg.grid_w,
        grid_h=benchmark_cfg.grid_h,
        step_penalty=benchmark_cfg.step_penalty,
        food_reward=benchmark_cfg.food_reward,
        death_reward=benchmark_cfg.death_reward,
        distance_reward_scale=hyperparams['distance_reward_scale'],
        render_mode="none",
    )
    env.reset(seed=seed)
    
    # Create agent
    agent_cfg = DQNConfig(
        grid_w=benchmark_cfg.grid_w,
        grid_h=benchmark_cfg.grid_h,
        lr=hyperparams['lr'],
        gamma=hyperparams['gamma'],
        batch_size=hyperparams['batch_size'],
        target_update=hyperparams['target_update'],
        buffer_size=hyperparams['buffer_size'],
        train_start=benchmark_cfg.train_start,
        device=benchmark_cfg.device,
    )
    agent = DQNAgent(agent_cfg)
    
    # Training state
    obs, _ = env.reset(seed=seed)
    ep_return = 0.0
    ep_len = 0
    returns = deque(maxlen=100)
    lengths = deque(maxlen=100)
    best_eval_return = -float('inf')
    total_episodes = 0
    
    # Training loop
    start_time = time.time()
    epsilon_start = 1.0
    epsilon_end = 0.05
    
    iterator = tqdm(range(benchmark_cfg.benchmark_steps), disable=not verbose, 
                   desc=f"lr={hyperparams['lr']:.4f}, bs={hyperparams['batch_size']}")
    
    for step in iterator:
        epsilon = linear_epsilon(step, epsilon_start, epsilon_end, 
                                hyperparams['epsilon_decay_steps'])
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        agent.push(obs, action, reward, next_obs, terminated)
        agent.train_step()
        
        obs = next_obs
        ep_return += reward
        ep_len += 1
        
        if done:
            returns.append(ep_return)
            lengths.append(ep_len)
            total_episodes += 1
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0
        
        # Periodic evaluation
        if (step + 1) % benchmark_cfg.eval_interval == 0:
            eval_return, eval_length = evaluate(agent, env, benchmark_cfg.eval_episodes)
            best_eval_return = max(best_eval_return, eval_return)
            
            if verbose:
                avg_ret = np.mean(returns) if returns else 0.0
                iterator.set_postfix(
                    eps=f"{epsilon:.3f}",
                    avg_ret=f"{avg_ret:.2f}",
                    eval_ret=f"{eval_return:.2f}"
                )
    
    training_time = time.time() - start_time
    final_avg_return = float(np.mean(returns)) if returns else 0.0
    avg_length = float(np.mean(lengths)) if lengths else 0.0
    
    env.close()
    
    return BenchmarkResult(
        config=hyperparams,
        final_avg_return=final_avg_return,
        best_eval_return=best_eval_return,
        training_time=training_time,
        avg_episode_length=avg_length,
        total_episodes=total_episodes,
        seed=seed
    )


def run_benchmark_suite(benchmark_cfg: BenchmarkConfig, verbose: bool = True) -> List[BenchmarkResult]:
    """
    Run full benchmark suite over all hyperparameter combinations.
    
    Args:
        benchmark_cfg: Benchmark configuration
        verbose: Whether to show detailed progress
        
    Returns:
        List of BenchmarkResult objects
    """
    # Generate all hyperparameter combinations
    param_grid = {
        'lr': benchmark_cfg.learning_rates,
        'batch_size': benchmark_cfg.batch_sizes,
        'gamma': benchmark_cfg.gamma_values,
        'epsilon_decay_steps': benchmark_cfg.epsilon_decay_steps,
        'distance_reward_scale': benchmark_cfg.distance_reward_scales,
        'buffer_size': benchmark_cfg.buffer_sizes,
        'target_update': benchmark_cfg.target_update_intervals,
    }
    
    # Create all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    total_configs = len(combinations) * benchmark_cfg.n_runs_per_config
    
    print(f"\n{'='*80}")
    print(f"HYPERPARAMETER BENCHMARK")
    print(f"{'='*80}")
    print(f"Total configurations: {len(combinations)}")
    print(f"Runs per configuration: {benchmark_cfg.n_runs_per_config}")
    print(f"Steps per run: {benchmark_cfg.benchmark_steps}")
    print(f"Total runs: {total_configs}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, combo in enumerate(combinations, 1):
        hyperparams = dict(zip(keys, combo))
        
        print(f"\n[{i}/{len(combinations)}] Testing configuration:")
        for key, value in hyperparams.items():
            print(f"  {key}: {value}")
        
        # Run multiple times for stability
        run_results = []
        for run in range(benchmark_cfg.n_runs_per_config):
            seed = benchmark_cfg.seed_base + run
            show_progress = verbose and (run == 0)  # Only show progress for first run
            
            if benchmark_cfg.n_runs_per_config > 1:
                print(f"  Run {run + 1}/{benchmark_cfg.n_runs_per_config} (seed={seed})")
            
            result = run_single_benchmark(hyperparams, benchmark_cfg, seed, show_progress)
            run_results.append(result)
            results.append(result)
        
        # Report average performance across runs
        avg_final_return = np.mean([r.final_avg_return for r in run_results])
        avg_best_eval = np.mean([r.best_eval_return for r in run_results])
        std_final_return = np.std([r.final_avg_return for r in run_results])
        
        print(f"  Results: avg_return={avg_final_return:.3f}±{std_final_return:.3f}, "
              f"best_eval={avg_best_eval:.3f}")
    
    return results


def save_results(results: List[BenchmarkResult], output_dir: str) -> None:
    """Save benchmark results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to serializable format
    results_data = []
    for result in results:
        result_dict = asdict(result)
        results_data.append(result_dict)
    
    # Save to file with timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"benchmark_results_{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


def print_summary(results: List[BenchmarkResult], top_n: int = 5) -> None:
    """
    Print summary of benchmark results.
    
    Args:
        results: List of benchmark results
        top_n: Number of top configurations to display
    """
    print(f"\n{'='*80}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    # Group results by configuration (average across runs)
    config_results = {}
    for result in results:
        config_key = json.dumps(result.config, sort_keys=True)
        if config_key not in config_results:
            config_results[config_key] = []
        config_results[config_key].append(result)
    
    # Calculate average performance for each configuration
    config_summary = []
    for config_key, run_results in config_results.items():
        config = run_results[0].config
        avg_final_return = np.mean([r.final_avg_return for r in run_results])
        std_final_return = np.std([r.final_avg_return for r in run_results])
        avg_best_eval = np.mean([r.best_eval_return for r in run_results])
        avg_time = np.mean([r.training_time for r in run_results])
        
        config_summary.append({
            'config': config,
            'avg_final_return': avg_final_return,
            'std_final_return': std_final_return,
            'avg_best_eval': avg_best_eval,
            'avg_time': avg_time,
            'n_runs': len(run_results)
        })
    
    # Sort by average final return
    config_summary.sort(key=lambda x: x['avg_final_return'], reverse=True)
    
    print(f"TOP {top_n} CONFIGURATIONS (by average return):")
    print(f"{'-'*80}\n")
    
    for i, summary in enumerate(config_summary[:top_n], 1):
        print(f"#{i} Configuration:")
        print(f"  Average Return: {summary['avg_final_return']:.3f} ± {summary['std_final_return']:.3f}")
        print(f"  Best Eval Return: {summary['avg_best_eval']:.3f}")
        print(f"  Training Time: {summary['avg_time']:.1f}s")
        print(f"  Parameters:")
        for key, value in summary['config'].items():
            print(f"    {key}: {value}")
        print()
    
    # Print best single run
    best_run = max(results, key=lambda r: r.final_avg_return)
    print(f"{'-'*80}")
    print(f"BEST SINGLE RUN:")
    print(f"  Final Average Return: {best_run.final_avg_return:.3f}")
    print(f"  Best Eval Return: {best_run.best_eval_return:.3f}")
    print(f"  Seed: {best_run.seed}")
    print(f"  Parameters:")
    for key, value in best_run.config.items():
        print(f"    {key}: {value}")
    print(f"{'='*80}\n")


def main():
    """Main entry point for hyperparameter benchmark."""
    parser = argparse.ArgumentParser(
        description="Benchmark hyperparameters for Snake RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with default parameters (small grid search)
  python benchmark_hyperparameters.py --benchmark_steps 20000
  
  # Comprehensive search with more steps
  python benchmark_hyperparameters.py --benchmark_steps 100000 --n_runs 3
  
  # Custom parameter ranges
  python benchmark_hyperparameters.py --lr 0.0001 0.0002 --batch_size 32 64 128
  
  # Use specific device
  python benchmark_hyperparameters.py --device cuda --benchmark_steps 50000
        """
    )
    
    # Benchmark settings
    parser.add_argument("--benchmark_steps", type=int, default=50000,
                       help="Training steps per configuration")
    parser.add_argument("--eval_interval", type=int, default=10000,
                       help="Steps between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=5,
                       help="Episodes per evaluation")
    parser.add_argument("--n_runs", type=int, default=1,
                       help="Number of runs per configuration (for stability)")
    
    # Grid search ranges
    parser.add_argument("--lr", type=float, nargs='+', default=None,
                       help="Learning rates to test (default: 0.0001 0.0002 0.0005)")
    parser.add_argument("--batch_size", type=int, nargs='+', default=None,
                       help="Batch sizes to test (default: 32 64 128)")
    parser.add_argument("--gamma", type=float, nargs='+', default=None,
                       help="Gamma values to test (default: 0.95 0.99)")
    parser.add_argument("--epsilon_decay", type=int, nargs='+', default=None,
                       help="Epsilon decay steps to test (default: 10000 20000 30000)")
    parser.add_argument("--distance_reward", type=float, nargs='+', default=None,
                       help="Distance reward scales to test (default: 0.0 0.05 0.1 0.2)")
    
    # Fixed parameters
    parser.add_argument("--grid_w", type=int, default=24,
                       help="Grid width")
    parser.add_argument("--grid_h", type=int, default=20,
                       help="Grid height")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cpu", "cuda"],
                       help="Device to use for training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Base random seed")
    
    # Output settings
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress progress bars")
    parser.add_argument("--top_n", type=int, default=5,
                       help="Number of top configurations to display")
    
    args = parser.parse_args()
    
    # Create benchmark configuration
    benchmark_cfg = BenchmarkConfig(
        benchmark_steps=args.benchmark_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        n_runs_per_config=args.n_runs,
        learning_rates=args.lr,
        batch_sizes=args.batch_size,
        gamma_values=args.gamma,
        epsilon_decay_steps=args.epsilon_decay,
        distance_reward_scales=args.distance_reward,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        device=args.device,
        seed_base=args.seed,
        output_dir=args.output_dir,
    )
    
    # Run benchmark
    results = run_benchmark_suite(benchmark_cfg, verbose=not args.quiet)
    
    # Save and display results
    save_results(results, benchmark_cfg.output_dir)
    print_summary(results, top_n=args.top_n)


if __name__ == "__main__":
    main()
