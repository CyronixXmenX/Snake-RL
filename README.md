# Snake RL (DQN)

Train a Deep Q-Network (DQN) to play Snake on a grid world that mirrors the Pygame version's mechanics.

📚 **[Quick Start Guide](QUICKSTART.md)** | 🚀 **[Optimizations](OPTIMIZATIONS.md)** | 🎮 **[GPU Guide](GPU_OPTIMIZATION_GUIDE.md)** | 📝 **[Changelog](CHANGELOG.md)**

## Features
- Gymnasium environment (`SnakeEnv`) with:
  - 3-channel observation: [head, body, food] on a H×W grid
  - Discrete actions: 0=Up, 1=Down, 2=Left, 3=Right
  - Reverse-direction input is ignored (like the Pygame game)
  - Rewards: +1 (eat), -1 (death), -0.01 (step)
  - Optimized collision detection with cached snake positions
- DQN with:
  - CNN backbone suitable for small grids (default 24×20)
  - Experience replay with uint8 observation storage for memory efficiency
  - Double DQN for reduced Q-value overestimation
  - Target network, epsilon-greedy exploration, gradient clipping
  - Checkpointing to `checkpoints/` with optimizer state
  - YAML configuration file support
- GPU Optimizations:
  - Automatic Mixed Precision (AMP) for 2-3x faster GPU training
  - Pinned memory for faster CPU-to-GPU data transfers
  - Non-blocking data transfers for better GPU utilization
  - Gradient accumulation for larger effective batch sizes
  - See [GPU Optimization Guide](GPU_OPTIMIZATION_GUIDE.md) for details
- Code Quality:
  - Comprehensive docstrings for all classes and methods
  - Full type hints throughout the codebase
  - Improved error handling and validation

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. (Optional) If you want to visualize evaluation with Pygame later, ensure your system has a display (or use a virtual framebuffer on headless servers).

3. (Optional) For GPU acceleration, ensure CUDA is installed and PyTorch detects your GPU:
   ```bash
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## Configuration
You can configure training parameters using either command-line arguments or a YAML configuration file.

### Using Config File (Recommended)
Create a `config.yaml` file (see `config.yaml` for an example):
```bash
python train_dqn.py --config config.yaml
```

### Using Command-Line Arguments
```bash
python train_dqn.py --grid_w 24 --grid_h 20 --total_steps 500000 --lr 0.0001
```

Command-line arguments override config file settings.

## Train
```bash
python train_dqn.py \
  --grid_w 24 --grid_h 20 \
  --total_steps 500000 \
  --batch_size 64 \
  --lr 1e-4 \
  --gamma 0.99 \
  --buffer_size 100000 \
  --train_start 10000 \
  --target_update 1000 \
  --eps_start 1.0 --eps_end 0.05 --eps_decay_steps 200000 \
  --step_penalty -0.01 --food_reward 1.0 --death_reward -1.0
```

Or use a configuration file:
```bash
python train_dqn.py --config config.yaml
```

### GPU-Accelerated Training

For 10-20x faster training on NVIDIA GPUs:
```bash
# Using command line
python train_dqn.py --use_amp --batch_size 128

# Or use the GPU-optimized config
python train_dqn.py --config config_gpu.yaml

# Run the GPU example
python example_gpu.py
```

See [GPU Optimization Guide](GPU_OPTIMIZATION_GUIDE.md) for detailed information.

- Checkpoints are written to `checkpoints/dqn_snake_latest.pth` and best average score to `checkpoints/dqn_snake_best.pth`.
- Training logs are saved to `checkpoints/training.log` by default.
- Default grid is 24×20 to match the Pygame version.

## Evaluate (watch the trained agent)
```bash
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --episodes 5 --render True
```

- Without `--render True`, it runs headless and prints scores.
- With `--render True`, it opens a Pygame window to visualize.
- Fixed `--render` argument parsing to properly accept True/False values.

## Files
- `snake_env.py` — Gymnasium environment for Snake
- `dqn_agent.py` — DQN model, replay buffer, agent utilities
- `train_dqn.py` — Training loop with config file support
- `evaluate_dqn.py` — Run trained model with optional rendering
- `config_utils.py` — Configuration management utilities
- `logger_utils.py` — Structured logging for training
- `config.yaml` — Example configuration file
- `config_gpu.yaml` — GPU-optimized configuration example
- `example_gpu.py` — Example script demonstrating GPU optimizations
- `requirements.txt` — Python dependencies
- `main.py` — Manual play Snake game (Pygame)
- `benchmark.py` — Performance benchmarking utilities

## Tips
- Training from scratch can take time. Use CPU or GPU; PyTorch will auto-detect GPU if available.
- For GPU acceleration, see the [GPU Optimization Guide](GPU_OPTIMIZATION_GUIDE.md).
- Enable `--use_amp` flag for 2-3x faster training on modern GPUs.
- You can reduce grid size (e.g., 12×10) to speed up learning initially.
- Reward shaping matters. The provided defaults are balanced for stability, but you can adjust `--step_penalty`, `--food_reward`, and `--death_reward`.

## Compatibility with your Pygame Snake
- Mechanics mirror the Pygame game: grid movement, no direct reverse, food spawn not on snake, same collision rules.
- This environment trains headless. To reuse your existing `main.py` for visualization, you can adapt it to read actions from the agent each tick. Alternatively, use `evaluate_dqn.py` to visualize directly.