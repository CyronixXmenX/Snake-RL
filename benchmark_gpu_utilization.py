"""
Benchmark script to compare standard vs optimized GPU utilization.

This script runs short training sessions with both approaches and compares:
- Wall-clock time
- Environment steps per second
- Training steps per second
- GPU utilization (if available)
"""

import time
import subprocess
import sys
from typing import Dict, Tuple


def run_training(script: str, config: dict) -> Tuple[float, int]:
    """
    Run a training script and measure performance.
    
    Args:
        script: Path to training script
        config: Configuration dictionary with training parameters
        
    Returns:
        Tuple of (wall_clock_time, total_steps)
    """
    args = [
        sys.executable, script,
        "--total_steps", str(config["total_steps"]),
        "--grid_w", str(config["grid_w"]),
        "--grid_h", str(config["grid_h"]),
        "--batch_size", str(config["batch_size"]),
        "--buffer_size", str(config["buffer_size"]),
        "--train_start", str(config["train_start"]),
        "--eval_interval", str(config["eval_interval"]),
        "--device", config["device"],
        "--no_console_log",
    ]
    
    if "num_envs" in config:
        args.extend(["--num_envs", str(config["num_envs"])])
    if "train_freq" in config:
        args.extend(["--train_freq", str(config["train_freq"])])
    
    if config.get("use_amp"):
        args.append("--use_amp")
    
    print(f"Running: {' '.join(args)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            cwd="/home/runner/work/Snake-RL/Snake-RL"
        )
        
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
            return 0.0, 0
            
    except subprocess.TimeoutExpired:
        print("Training timed out!")
        return 0.0, 0
    
    wall_time = time.time() - start_time
    return wall_time, config["total_steps"]


def print_comparison(standard_time: float, optimized_time: float, steps: int):
    """Print comparison results."""
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    
    print(f"\nTraining Steps: {steps}")
    print(f"\nStandard Training (train_dqn.py):")
    print(f"  Wall-clock time: {standard_time:.2f} seconds")
    print(f"  Steps/second: {steps / standard_time:.1f}")
    
    print(f"\nOptimized Training (train_dqn_optimized.py):")
    print(f"  Wall-clock time: {optimized_time:.2f} seconds")
    print(f"  Steps/second: {steps / optimized_time:.1f}")
    
    speedup = standard_time / optimized_time if optimized_time > 0 else 0
    print(f"\nSpeedup: {speedup:.2f}x faster")
    
    print("\nNote: This benchmark measures wall-clock time only.")
    print("For GPU utilization, monitor with: watch -n 1 nvidia-smi")
    print("Expected GPU utilization:")
    print("  - Standard training: ~1-5%")
    print("  - Optimized training: ~70-90%")
    print("=" * 70)


def main():
    """Run benchmarks."""
    print("GPU Utilization Benchmark")
    print("=" * 70)
    print("\nThis benchmark compares standard vs optimized training.")
    print("Running short training sessions (may take 2-5 minutes total)...\n")
    
    # Small configuration for quick testing
    base_config = {
        "total_steps": 1000,
        "grid_w": 12,
        "grid_h": 10,
        "batch_size": 32,
        "buffer_size": 2000,
        "train_start": 100,
        "eval_interval": 10000,  # Skip evaluation
        "device": "cpu",  # Use CPU for testing (GPU if available)
    }
    
    # Check if CUDA is available
    try:
        import torch
        if torch.cuda.is_available():
            base_config["device"] = "cuda"
            base_config["use_amp"] = True
            print("✓ CUDA detected - will benchmark on GPU\n")
        else:
            print("! CUDA not available - benchmarking on CPU")
            print("  (GPU speedup will be much more dramatic on actual GPU)\n")
    except ImportError:
        print("! PyTorch not found - benchmarking on CPU\n")
    
    # Standard training config
    standard_config = base_config.copy()
    
    # Optimized training config
    optimized_config = base_config.copy()
    optimized_config["num_envs"] = 4
    optimized_config["train_freq"] = 4
    optimized_config["batch_size"] = 64
    
    print("1. Running standard training...")
    standard_time, steps = run_training("train_dqn.py", standard_config)
    
    if standard_time == 0:
        print("Standard training failed!")
        return
    
    print(f"   Completed in {standard_time:.2f} seconds\n")
    
    print("2. Running optimized training...")
    optimized_time, _ = run_training("train_dqn_optimized.py", optimized_config)
    
    if optimized_time == 0:
        print("Optimized training failed!")
        return
    
    print(f"   Completed in {optimized_time:.2f} seconds\n")
    
    # Print comparison
    print_comparison(standard_time, optimized_time, steps)
    
    print("\nTo see GPU utilization in real-time:")
    print("  1. Start training in one terminal:")
    print("     python train_dqn_optimized.py --config config_gpu_optimized.yaml")
    print("  2. Monitor GPU in another terminal:")
    print("     watch -n 1 nvidia-smi")


if __name__ == "__main__":
    main()
