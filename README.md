# Snake RL with Deep Q-Network (DQN)

Train a Deep Q-Network agent to play Snake using reinforcement learning. The implementation uses PyTorch and follows best practices for stable DQN training.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Agent

**CPU Training:**
```bash
python train_dqn.py --config config.yaml
```

**GPU Training (15-20x faster):**
```bash
python train_dqn.py --config config_gpu.yaml
```

**Quick Test (10k steps):**
```bash
python train_dqn.py --total_steps 10000 --eval_interval 5000
```

### 3. Evaluate Your Agent
```bash
# Watch the agent play (with visualization)
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --episodes 5 --render True

# Quick performance test (no visualization)
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --episodes 10 --render False
```

### 4. Play Manually (Optional)
```bash
python main.py
```
Use arrow keys or WASD to control the snake.

---

## 📊 Features

### Environment
- **Gymnasium-compatible** Snake environment with optimized collision detection
- **3-channel observation**: [head, body, food] on a configurable H×W grid
- **Discrete actions**: 0=Up, 1=Down, 2=Left, 3=Right
- **Smart action handling**: Reverse-direction moves are ignored (prevents instant death)
- **Configurable rewards**: 
  - +1.0 for eating food (default)
  - -1.0 for dying (collision)
  - -0.01 per step (time penalty)
  - **Distance-based reward shaping**: Optional dense rewards for moving toward food (improves learning)

### DQN Agent
- **CNN architecture** optimized for small grid environments
- **Experience replay** with memory-efficient uint8 storage
- **Double DQN** to reduce Q-value overestimation
- **Target network** for stable learning
- **Epsilon-greedy exploration** with linear decay
- **Gradient clipping** to prevent instability
- **Checkpointing** with best model tracking

### GPU Optimization
- **Automatic device detection** (CPU/CUDA)
- **GPU-optimized replay buffer** with zero-copy sampling (data stored directly on GPU)
- **Mixed Precision Training (AMP)** for 2-3x speedup on modern GPUs
- **Pinned memory** for async CPU-GPU transfers
- **torch.compile** support for JIT compilation (PyTorch 2.0+, 20-30% speedup)
- **TF32 acceleration** on Ampere+ GPUs (RTX 30xx+)
- **cuDNN benchmarking** for optimized convolution kernels
- **Gradient accumulation** for larger effective batch sizes
- **CUDA streams** for concurrent GPU operations

---

## ⚙️ Configuration

### Using Config Files (Recommended)

Config files provide a clean way to manage hyperparameters:

```bash
# CPU-optimized (balanced speed/memory)
python train_dqn.py --config config.yaml

# GPU-optimized (faster training)
python train_dqn.py --config config_gpu.yaml
```

### Creating Custom Config

Create your own `my_config.yaml`:

```yaml
environment:
  grid_width: 24
  grid_height: 20
  step_penalty: -0.01
  food_reward: 1.0
  death_reward: -1.0
  distance_reward_scale: 0.1  # Reward shaping for guiding toward food

dqn:
  learning_rate: 0.0001
  batch_size: 64
  buffer_size: 100000
  gamma: 0.99
  target_update: 1000
  train_start: 10000

training:
  total_steps: 500000
  device: auto  # auto, cpu, or cuda
  checkpoint_dir: checkpoints
  eval_interval: 10000
  eval_episodes: 5

exploration:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_steps: 200000

gpu_optimization:
  use_amp: false  # Enable for 2-3x speedup on modern GPUs (RTX 20xx+)
  pin_memory: false  # Usually not needed
  gradient_accumulation_steps: 1  # Increase for larger effective batch size
  compile_model: false  # Enable for 20-30% speedup on PyTorch 2.0+
```

### Command-Line Overrides

Override any config setting via command line:
```bash
python train_dqn.py --config config.yaml --total_steps 100000 --lr 0.0002 --batch_size 128
```

---

## 💻 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA support for faster training

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CyronixXmenX/Snake-RL.git
   cd Snake-RL
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "from snake_env import SnakeEnv; from dqn_agent import DQNAgent; print('✓ Installation successful!')"
   ```

### GPU Setup (Optional)

**Check if GPU is available:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**If GPU is available**, you can use GPU training immediately with `--config config_gpu.yaml`.

**If GPU is not detected:**
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA: https://pytorch.org/get-started/locally/

---

## 🎯 Usage Examples

### Training Options

**Option 1: Quick CPU Training**
```bash
python train_dqn.py --config config.yaml
```

**Option 2: GPU Training (Recommended if available)**
```bash
python train_dqn.py --config config_gpu.yaml
```

**Option 3: Custom Parameters**
```bash
python train_dqn.py \
  --grid_w 24 \
  --grid_h 20 \
  --total_steps 500000 \
  --batch_size 64 \
  --lr 0.0001 \
  --device auto
```

**Option 4: GPU with All Optimizations (Maximum Speed)**
```bash
python train_dqn.py --config config_gpu.yaml
# or manually:
python train_dqn.py \
  --device cuda \
  --use_amp \
  --pin_memory \
  --compile_model \
  --batch_size 128 \
  --lr 0.0002 \
  --total_steps 500000
```

**Option 5: Benchmark GPU Performance**
```bash
# Single run benchmark
python benchmark_gpu.py --num_steps 1000 --batch_size 128

# Compare different configurations
python benchmark_gpu.py --compare --num_steps 1000
```

**Option 6: Find Best Hyperparameters**
```bash
# Quick hyperparameter search
python benchmark_hyperparameters.py --benchmark_steps 20000

# Comprehensive search with multiple runs
python benchmark_hyperparameters.py --benchmark_steps 100000 --n_runs 3

# Custom parameter ranges
python benchmark_hyperparameters.py --lr 0.0001 0.0002 --batch_size 32 64 128
```

See [BENCHMARK_GUIDE.md](BENCHMARK_GUIDE.md) for detailed usage instructions.

### Training Output

During training, you'll see:
- Progress bar with current training step
- Average episode return (higher is better)
- Average episode length (longer means better survival)
- Training loss
- Current exploration rate (epsilon)

Checkpoints are automatically saved to:
- `checkpoints/dqn_snake_latest.pth` - Most recent model
- `checkpoints/dqn_snake_best.pth` - Best performing model

Training logs are saved to `checkpoints/training.log`. Monitor in real-time:
```bash
tail -f checkpoints/training.log
```

### Evaluation Options

**Watch the agent play (with visualization):**
```bash
python evaluate_dqn.py \
  --model checkpoints/dqn_snake_best.pth \
  --episodes 5 \
  --render True \
  --step_delay 0.1
```

**Quick performance test (no visualization):**
```bash
python evaluate_dqn.py \
  --model checkpoints/dqn_snake_best.pth \
  --episodes 20 \
  --render False
```

---

## 🚀 GPU Training Guide

### Performance Expectations

With the GPU optimizations in this implementation:

- **CPU Training**: ~5-10 steps/second (baseline)
- **GPU Training (basic)**: ~50-100 steps/second (5-10x faster)
- **GPU Training + AMP**: ~100-200 steps/second (10-20x faster)
- **GPU Training + All Optimizations**: ~150-250 steps/second (15-25x faster)

> **Note**: Performance depends heavily on hardware. Modern NVIDIA GPUs with Tensor Cores (RTX 20xx+, V100+) provide the best speedups. The GPU-optimized replay buffer stores data directly on GPU, eliminating expensive CPU-GPU transfers during training.

### GPU Settings Explained

#### `use_amp` (Automatic Mixed Precision)
- **What it does**: Uses 16-bit floats for some operations instead of 32-bit
- **When to use**: Always enable on modern GPUs (RTX 20xx+, Tesla V100+)
- **Benefit**: 2-3x faster training, reduced memory usage
- **Drawback**: None on supported hardware

#### `pin_memory`
- **What it does**: Pre-allocates page-locked memory for faster CPU-to-GPU transfers
- **When to use**: When replay buffer is on CPU but model is on GPU
- **Default**: Enabled in config_gpu.yaml
- **Note**: Not needed when buffer is on GPU (zero-copy sampling)

#### `gradient_accumulation_steps`
- **What it does**: Simulates larger batch sizes by accumulating gradients
- **When to use**: When you want larger batch size but have limited GPU memory
- **Example**: `batch_size=32` with `gradient_accumulation_steps=4` = effective batch size of 128

#### `compile_model` (PyTorch 2.0+)
- **What it does**: JIT compiles the model for optimized execution
- **When to use**: Always enable on PyTorch 2.0+ for additional 20-30% speedup
- **Note**: First run is slower due to compilation, subsequent runs are faster
- **Benefit**: Optimized CUDA kernels, graph optimizations

### GPU Replay Buffer

The replay buffer can store data directly on GPU or CPU:

- **GPU Buffer** (default on CUDA): Zero-copy sampling, no transfer overhead
- **CPU Buffer + Pin Memory**: Async transfers to GPU during training
- **CPU Buffer**: Standard transfers (slowest but uses less GPU memory)

To use GPU buffer, the model must be on GPU (`device=cuda`). The buffer will automatically use GPU storage.

### GPU Memory Optimization

**For 4-6GB GPUs:**
```bash
python train_dqn.py \
  --device cuda \
  --use_amp \
  --batch_size 32 \
  --buffer_size 50000
```

**For 8GB+ GPUs:**
```bash
python train_dqn.py --config config_gpu.yaml
```

**Out of Memory? Try this:**
```bash
python train_dqn.py \
  --device cuda \
  --use_amp \
  --batch_size 32 \
  --gradient_accumulation_steps 4
```

---

## 📁 Project Structure

```
Snake-RL/
├── train_dqn.py                 # Main training script
├── evaluate_dqn.py              # Model evaluation script
├── benchmark_gpu.py             # GPU performance benchmarking
├── benchmark_hyperparameters.py # Hyperparameter grid search
├── main.py                      # Manual play (Pygame)
├── snake_env.py                 # Gymnasium Snake environment
├── dqn_agent.py                 # DQN agent implementation
├── config_utils.py              # Configuration management
├── logger_utils.py              # Training logging utilities
├── config.yaml                  # CPU training config
├── config_gpu.yaml              # GPU training config
├── benchmark_config_example.yaml # Benchmark parameter ranges
├── requirements.txt             # Python dependencies
├── checkpoints/                 # Saved model checkpoints
├── BENCHMARK_GUIDE.md           # Hyperparameter benchmark guide
└── README.md                    # This file
```

---

## 💡 Tips & Best Practices

### Training Tips
- **Start small**: Try `--total_steps 100000` first to verify everything works
- **Monitor progress**: Use `tail -f checkpoints/training.log` to watch training
- **Enable GPU**: Use `--config config_gpu.yaml` for 15-20x speedup
- **Smaller grids learn faster**: Try 12×10 grid first, then scale up

### Hyperparameter Tips
- **Learning rate**: Start with 0.0001, increase to 0.0002 for GPU training
- **Batch size**: 64 for CPU, 128-256 for GPU
- **Buffer size**: 100k is usually sufficient, use 50k if memory is limited
- **Epsilon decay**: Longer decay (200k+ steps) usually works better

### Reward Tuning
Adjust rewards to shape agent behavior:
```bash
python train_dqn.py \
  --step_penalty -0.01 \          # Encourages efficiency
  --food_reward 1.0 \              # Reward for eating
  --death_reward -1.0 \            # Penalty for dying
  --distance_reward_scale 0.1      # Guides agent toward food (0 to disable)
```

**Note**: The `distance_reward_scale` parameter provides dense reward signals by rewarding the agent for moving closer to food. This significantly improves learning by giving immediate feedback. Set to 0.0 to disable reward shaping.

---

## 🔧 Troubleshooting

### Training is slow
```bash
# Enable GPU and AMP
python train_dqn.py --config config_gpu.yaml
```

### CUDA Out of Memory
```bash
# Reduce batch size and buffer
python train_dqn.py --batch_size 32 --buffer_size 50000

# Or use gradient accumulation
python train_dqn.py --batch_size 32 --gradient_accumulation_steps 4
```

### Agent not learning
- Check that rewards are appropriate (not all negative)
- Increase training steps (500k+ may be needed)
- Verify epsilon decay is not too fast
- Try different random seeds

### Pygame display issues (on headless server)
```bash
# Disable rendering during evaluation
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --render False
```

### Import/Module errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Open issues for bugs or feature requests
- Submit pull requests with improvements
- Share your training results and configurations

---

## 📄 License

See LICENSE file for details.

---

## 🎮 Game Mechanics

This implementation mirrors the classic Snake game:
- Snake moves continuously in the current direction
- Cannot reverse directly into itself (ignored as invalid move)
- Dies on collision with walls or its own body
- Grows by one segment when eating food
- Food spawns randomly in empty cells
- Episode ends on death or maximum steps reached

---

## 📈 Expected Results

With default settings, you should see:
- **Episode return** gradually increase from -10 to 5+
- **Average length** increase from 10-20 steps to 50+ steps
- **Success rate** improve as agent learns to avoid walls and itself
- Training typically takes 200k-500k steps for good performance

Typical learning curve:
- 0-50k steps: Random exploration, mostly dying quickly
- 50k-150k steps: Learning basic survival and food collection
- 150k-300k steps: Improving efficiency and longer survival
- 300k+ steps: Near-optimal play on the grid size

---

## 🔗 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gymnasium API](https://gymnasium.farama.org/)
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Original Deep Q-Learning
- [Double DQN Paper](https://arxiv.org/abs/1509.06461) - Improved Q-Learning

---

**Happy Training! 🐍🎮**
