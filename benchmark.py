"""
Performance benchmarking utilities for Snake RL.

Provides tools to measure and compare performance of environment and agent.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, Any, List
import numpy as np

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig


class Benchmark:
    """
    Simple benchmarking utility for measuring execution time.
    """
    
    def __init__(self, name: str = "Benchmark"):
        self.name = name
        self.times: List[float] = []
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        self.times.append(elapsed)
    
    def avg_time(self) -> float:
        """Get average execution time in seconds."""
        return np.mean(self.times) if self.times else 0.0
    
    def total_time(self) -> float:
        """Get total execution time in seconds."""
        return sum(self.times)
    
    def report(self) -> str:
        """Generate a report string."""
        if not self.times:
            return f"{self.name}: No measurements"
        
        avg = self.avg_time() * 1000  # Convert to ms
        total = self.total_time()
        count = len(self.times)
        
        return (
            f"{self.name}:\n"
            f"  Runs: {count}\n"
            f"  Total: {total:.3f}s\n"
            f"  Average: {avg:.3f}ms\n"
            f"  Min: {min(self.times)*1000:.3f}ms\n"
            f"  Max: {max(self.times)*1000:.3f}ms"
        )


def benchmark_environment(
    grid_w: int = 24, 
    grid_h: int = 20, 
    episodes: int = 100
) -> Dict[str, float]:
    """
    Benchmark environment performance.
    
    Args:
        grid_w: Grid width
        grid_h: Grid height
        episodes: Number of episodes to run
        
    Returns:
        Dictionary with benchmark results
    """
    env = SnakeEnv(grid_w=grid_w, grid_h=grid_h, render_mode="none")
    
    reset_bench = Benchmark("Environment reset")
    step_bench = Benchmark("Environment step")
    
    total_steps = 0
    start_time = time.perf_counter()
    
    for _ in range(episodes):
        with reset_bench:
            obs, _ = env.reset()
        
        done = False
        while not done:
            action = env.action_space.sample()
            with step_bench:
                obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_steps += 1
    
    total_time = time.perf_counter() - start_time
    
    env.close()
    
    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "total_time_s": total_time,
        "steps_per_second": total_steps / total_time,
        "avg_reset_time_ms": reset_bench.avg_time() * 1000,
        "avg_step_time_ms": step_bench.avg_time() * 1000,
    }


def benchmark_agent(
    grid_w: int = 24,
    grid_h: int = 20,
    episodes: int = 100,
    device: str = "auto"
) -> Dict[str, float]:
    """
    Benchmark agent performance.
    
    Args:
        grid_w: Grid width
        grid_h: Grid height
        episodes: Number of episodes to run
        device: Device to use for benchmarking ('auto', 'cpu', 'cuda')
        
    Returns:
        Dictionary with benchmark results
    """
    env = SnakeEnv(grid_w=grid_w, grid_h=grid_h, render_mode="none")
    cfg = DQNConfig(grid_w=grid_w, grid_h=grid_h, device=device)
    agent = DQNAgent(cfg)
    
    inference_bench = Benchmark("Agent inference")
    training_bench = Benchmark("Agent training step")
    
    total_steps = 0
    start_time = time.perf_counter()
    
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        
        while not done:
            with inference_bench:
                action = agent.act(obs, epsilon=0.1)
            
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.push(obs, action, reward, next_obs, done)
            
            # Train every step if buffer has enough samples
            if agent.replay.size >= agent.train_start:
                with training_bench:
                    agent.train_step()
            
            obs = next_obs
            total_steps += 1
    
    total_time = time.perf_counter() - start_time
    
    env.close()
    
    return {
        "episodes": episodes,
        "total_steps": total_steps,
        "total_time_s": total_time,
        "steps_per_second": total_steps / total_time,
        "avg_inference_time_ms": inference_bench.avg_time() * 1000,
        "avg_training_time_ms": training_bench.avg_time() * 1000 if training_bench.times else 0,
        "training_steps": len(training_bench.times),
        "device": str(agent.device),
    }


def print_benchmark_results(results: Dict[str, Any], title: str = "Benchmark Results") -> None:
    """
    Pretty print benchmark results.
    
    Args:
        results: Dictionary with benchmark metrics
        title: Title for the report
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    for key, value in results.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print(f"{'='*60}\n")


def main():
    """Run comprehensive benchmarks."""
    import torch
    
    print("Running Snake RL Performance Benchmarks...")
    print("This may take a minute...\n")
    
    # Benchmark environment
    env_results = benchmark_environment(episodes=100)
    print_benchmark_results(env_results, "Environment Performance")
    
    # Benchmark agent on CPU
    print("Benchmarking agent on CPU...")
    agent_results_cpu = benchmark_agent(episodes=20, device="cpu")
    print_benchmark_results(agent_results_cpu, "Agent Performance (CPU)")
    
    # Benchmark agent on GPU if available
    if torch.cuda.is_available():
        print("Benchmarking agent on GPU...")
        agent_results_gpu = benchmark_agent(episodes=20, device="cuda")
        print_benchmark_results(agent_results_gpu, "Agent Performance (GPU)")
        
        # Calculate speedup
        cpu_training_time = agent_results_cpu["avg_training_time_ms"]
        gpu_training_time = agent_results_gpu["avg_training_time_ms"]
        if cpu_training_time > 0 and gpu_training_time > 0:
            speedup = cpu_training_time / gpu_training_time
            print(f"\n{'='*60}")
            print(f"GPU Speedup: {speedup:.2f}x faster than CPU")
            print(f"{'='*60}\n")
    else:
        print("\nGPU not available. Skipping GPU benchmarks.")
        print("To enable GPU acceleration:")
        print("1. Ensure CUDA is installed")
        print("2. Install PyTorch with CUDA support")
        print("3. Set device='cuda' or device='auto' in config\n")
    
    print("Benchmarks complete!")


if __name__ == "__main__":
    main()
