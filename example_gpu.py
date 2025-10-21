#!/usr/bin/env python3
"""
Example: Using GPU Optimizations

This script demonstrates how to use GPU optimization features in Snake RL.
"""

import torch
from dqn_agent import DQNAgent, DQNConfig
from snake_env import SnakeEnv


def main():
    print("=" * 60)
    print("Snake RL - GPU Optimization Example")
    print("=" * 60)
    print()
    
    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"✓ CUDA is available")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        device = "cuda"
    else:
        print("✗ CUDA not available, using CPU")
        print("  GPU optimizations will be automatically disabled")
        device = "cpu"
    print()
    
    # Create configuration with GPU optimizations
    print("Configuration:")
    cfg = DQNConfig(
        grid_w=24,
        grid_h=20,
        device=device,
        batch_size=128,  # Larger batch size for GPU
        lr=0.0002,  # Higher LR for larger batch
        use_amp=True,  # Enable mixed precision
        pin_memory=True,  # Enable pinned memory
        gradient_accumulation_steps=2,  # Accumulate gradients
    )
    
    print(f"  Device: {device}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  Use AMP: {cfg.use_amp}")
    print(f"  Pin memory: {cfg.pin_memory}")
    print(f"  Gradient accumulation: {cfg.gradient_accumulation_steps}")
    print(f"  Effective batch size: {cfg.batch_size * cfg.gradient_accumulation_steps}")
    print()
    
    # Create agent
    print("Creating agent...")
    agent = DQNAgent(cfg)
    print(f"✓ Agent created on {agent.device}")
    print(f"  AMP enabled: {agent.use_amp}")
    print(f"  Pin memory: {agent.pin_memory}")
    print()
    
    # Create environment
    print("Creating environment...")
    env = SnakeEnv(grid_w=cfg.grid_w, grid_h=cfg.grid_h, render_mode="none")
    print("✓ Environment created")
    print()
    
    # Run a few training steps
    print("Running sample training loop...")
    obs, _ = env.reset()
    
    # Collect initial experience
    print("  Collecting experience...")
    for _ in range(cfg.train_start):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.push(obs, action, reward, next_obs, terminated or truncated)
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    
    print(f"  Buffer filled: {len(agent.replay)}/{cfg.buffer_size}")
    
    # Train for a few steps
    print("  Training...")
    import time
    start_time = time.time()
    
    for step in range(100):
        action = agent.act(obs, epsilon=0.1)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.push(obs, action, reward, next_obs, terminated or truncated)
        loss = agent.train_step()
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    
    elapsed = time.time() - start_time
    steps_per_sec = 100 / elapsed
    
    print(f"  Completed 100 training steps")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {steps_per_sec:.1f} steps/second")
    print()
    
    # Cleanup
    env.close()
    
    print("=" * 60)
    print("Example completed successfully! ✓")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Train with config: python train_dqn.py --config config_gpu.yaml")
    print("  2. See GPU_OPTIMIZATION_GUIDE.md for detailed documentation")
    print("  3. Run benchmark: python benchmark.py")


if __name__ == "__main__":
    main()
