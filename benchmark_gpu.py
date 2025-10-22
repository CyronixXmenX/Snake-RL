"""
Benchmark script to measure GPU utilization and training performance.

This script compares different GPU optimization settings to demonstrate
the performance improvements.
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Any

import numpy as np
import torch

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig


def benchmark_training(
    cfg: DQNConfig,
    num_steps: int = 1000,
    warmup_steps: int = 100
) -> Dict[str, Any]:
    """
    Benchmark training performance with given configuration.
    
    Args:
        cfg: DQN configuration
        num_steps: Number of training steps to benchmark
        warmup_steps: Number of warmup steps before timing
        
    Returns:
        Dictionary with benchmark results
    """
    # Create environment and agent
    env = SnakeEnv(
        grid_w=cfg.grid_w,
        grid_h=cfg.grid_h,
        render_mode="none"
    )
    agent = DQNAgent(cfg)
    
    # Fill buffer with random transitions
    print(f"Filling replay buffer to {cfg.train_start} transitions...")
    obs, _ = env.reset()
    for _ in range(cfg.train_start):
        action = np.random.randint(0, cfg.num_actions)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        agent.push(obs, action, reward, next_obs, terminated)
        obs = next_obs if not done else env.reset()[0]
    
    # Warmup
    print(f"Warming up for {warmup_steps} steps...")
    for _ in range(warmup_steps):
        agent.train_step()
    
    # Synchronize GPU before timing
    if agent.use_gpu:
        torch.cuda.synchronize()
    
    # Benchmark training steps
    print(f"Benchmarking {num_steps} training steps...")
    start_time = time.time()
    
    for _ in range(num_steps):
        agent.train_step()
    
    # Synchronize GPU after timing
    if agent.use_gpu:
        torch.cuda.synchronize()
    
    elapsed_time = time.time() - start_time
    steps_per_sec = num_steps / elapsed_time
    
    # Get memory usage
    if agent.use_gpu:
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2  # MB
    else:
        memory_allocated = 0
        memory_reserved = 0
    
    results = {
        "device": str(agent.device),
        "use_amp": agent.use_amp,
        "pin_memory": agent.pin_memory,
        "compile_model": cfg.compile_model,
        "batch_size": cfg.batch_size,
        "buffer_device": str(agent.replay.device),
        "elapsed_time": elapsed_time,
        "steps_per_sec": steps_per_sec,
        "memory_allocated_mb": memory_allocated,
        "memory_reserved_mb": memory_reserved,
    }
    
    env.close()
    return results


def print_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "="*70)
    print("BENCHMARK RESULTS")
    print("="*70)
    print(f"Device:              {results['device']}")
    print(f"Batch Size:          {results['batch_size']}")
    print(f"Buffer Device:       {results['buffer_device']}")
    print(f"Use AMP:             {results['use_amp']}")
    print(f"Pin Memory:          {results['pin_memory']}")
    print(f"Compile Model:       {results['compile_model']}")
    print("-"*70)
    print(f"Elapsed Time:        {results['elapsed_time']:.2f} seconds")
    print(f"Steps/Second:        {results['steps_per_sec']:.2f}")
    if results['memory_allocated_mb'] > 0:
        print(f"GPU Memory (alloc):  {results['memory_allocated_mb']:.2f} MB")
        print(f"GPU Memory (total):  {results['memory_reserved_mb']:.2f} MB")
    print("="*70)


def main() -> None:
    """Run benchmarks with different configurations."""
    parser = argparse.ArgumentParser(description="Benchmark GPU performance")
    parser.add_argument("--num_steps", type=int, default=1000,
                        help="Number of training steps to benchmark")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--grid_w", type=int, default=24,
                        help="Grid width")
    parser.add_argument("--grid_h", type=int, default=20,
                        help="Grid height")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size")
    parser.add_argument("--compare", action="store_true",
                        help="Compare multiple configurations")
    args = parser.parse_args()
    
    if args.compare:
        # Compare different configurations
        configs = [
            ("Baseline (CPU)", {
                "device": "cpu",
                "use_amp": False,
                "pin_memory": False,
                "compile_model": False,
            }),
        ]
        
        # Add GPU configs if CUDA is available
        if torch.cuda.is_available():
            configs.extend([
                ("GPU (no optimizations)", {
                    "device": "cuda",
                    "use_amp": False,
                    "pin_memory": False,
                    "compile_model": False,
                }),
                ("GPU + AMP", {
                    "device": "cuda",
                    "use_amp": True,
                    "pin_memory": False,
                    "compile_model": False,
                }),
                ("GPU + AMP + Pin Memory", {
                    "device": "cuda",
                    "use_amp": True,
                    "pin_memory": True,
                    "compile_model": False,
                }),
                ("GPU + All Optimizations", {
                    "device": "cuda",
                    "use_amp": True,
                    "pin_memory": True,
                    "compile_model": True,
                }),
            ])
        
        all_results = []
        for name, config_overrides in configs:
            print(f"\n{'='*70}")
            print(f"Running: {name}")
            print(f"{'='*70}")
            
            cfg = DQNConfig(
                grid_w=args.grid_w,
                grid_h=args.grid_h,
                batch_size=args.batch_size,
                train_start=500,
                **config_overrides
            )
            
            results = benchmark_training(cfg, args.num_steps, args.warmup_steps)
            results["config_name"] = name
            all_results.append(results)
            print_results(results)
        
        # Print comparison table
        print("\n" + "="*70)
        print("PERFORMANCE COMPARISON")
        print("="*70)
        print(f"{'Configuration':<35} {'Steps/Sec':>15} {'Speedup':>10}")
        print("-"*70)
        
        baseline_speed = all_results[0]["steps_per_sec"]
        for result in all_results:
            speedup = result["steps_per_sec"] / baseline_speed
            print(f"{result['config_name']:<35} {result['steps_per_sec']:>15.2f} {speedup:>10.2f}x")
        print("="*70)
        
    else:
        # Single benchmark run
        device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = DQNConfig(
            grid_w=args.grid_w,
            grid_h=args.grid_h,
            batch_size=args.batch_size,
            device=device,
            use_amp=True if device == "cuda" else False,
            pin_memory=True if device == "cuda" else False,
            compile_model=True,
            train_start=500,
        )
        
        results = benchmark_training(cfg, args.num_steps, args.warmup_steps)
        print_results(results)


if __name__ == "__main__":
    main()
