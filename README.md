# Snake RL (DQN)

Train a Deep Q-Network (DQN) to play Snake on a grid world that mirrors the Pygame version's mechanics.

📚 **[Quick Start Guide](QUICKSTART.md)** | 🚀 **[Optimizations](OPTIMIZATIONS.md)** | 🎮 **[GPU Guide](GPU_OPTIMIZATION_GUIDE.md)** | ⚡ **[GPU Utilization](GPU_UTILIZATION_IMPROVEMENTS.md)** | 📝 **[Changelog](CHANGELOG.md)**

---

## 🚀 Quick Start (Copy & Run)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Training (Choose One)

**Basic CPU Training:**
```bash
python train_dqn.py --config config.yaml
```

**GPU-Accelerated Training (15-20x faster):**
```bash
python train_dqn.py --config config_gpu.yaml
```

**Maximum GPU Utilization (70-90% GPU usage, 8x faster than standard GPU):**
```bash
python train_dqn_optimized.py --config config_gpu_optimized.yaml
```

**Custom Quick Training (100k steps):**
```bash
python train_dqn.py --total_steps 100000 --eval_interval 5000
```

### 3. Watch Your Trained Agent
```bash
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --episodes 5 --render True
```

### 4. Play Manually (Optional)
```bash
python main.py
```
Use arrow keys or WASD to control the snake.

---

## ⚙️ Configuration

### Using Config Files (Recommended)
```bash
# CPU-optimized training
python train_dqn.py --config config.yaml

# GPU-optimized training
python train_dqn.py --config config_gpu.yaml
```

### Creating Custom Config
Create your own `my_config.yaml`:
```yaml
environment:
  grid_width: 24
  grid_height: 20

dqn:
  learning_rate: 0.0001
  batch_size: 64
  buffer_size: 100000

training:
  total_steps: 500000
  device: auto  # auto, cpu, or cuda

gpu_optimization:
  use_amp: false
  pin_memory: true
```

### Command-Line Overrides
Command-line arguments override config file settings:
```bash
python train_dqn.py --config config.yaml --total_steps 100000 --lr 0.0002
```

## 📊 Features
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
  - Vectorized environments for parallel data collection (NEW!)
  - Batch action inference for efficient GPU utilization (NEW!)
  - Multiple training steps per environment step (NEW!)
  - 70-90% GPU utilization vs 1.5% with standard training (NEW!)
  - See [GPU Optimization Guide](GPU_OPTIMIZATION_GUIDE.md) for details
  - See [GPU Utilization Improvements](GPU_UTILIZATION_IMPROVEMENTS.md) for maximum performance
- Code Quality:
  - Comprehensive docstrings for all classes and methods
  - Full type hints throughout the codebase
  - Improved error handling and validation

## 💻 Setup & Installation

### Prerequisites
- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA support for faster training

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CyronixXmenX/snake-ml.git
   cd snake-ml
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**
   ```bash
   python -c "from snake_env import SnakeEnv; from dqn_agent import DQNAgent; print('✓ Installation successful!')"
   ```

### GPU Setup (Optional - For 15-20x Speedup)

**Check if GPU is available:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**If GPU is available, you're ready to use GPU training!** The scripts will automatically use GPU when available.

**If GPU is not detected:**
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Reinstall PyTorch with CUDA support: https://pytorch.org/get-started/locally/

## 🎯 Usage

### Training Your Agent

#### Option 1: Quick Start with Config (Recommended)
```bash
# CPU training (works everywhere)
python train_dqn.py --config config.yaml

# GPU training (15-20x faster if you have NVIDIA GPU)
python train_dqn.py --config config_gpu.yaml
```

#### Option 2: Command Line Arguments
```bash
python train_dqn.py \
  --grid_w 24 --grid_h 20 \
  --total_steps 500000 \
  --batch_size 64 \
  --lr 0.0001 \
  --gamma 0.99
```

#### Option 3: GPU-Optimized Training (Maximum Performance)
```bash
# All GPU optimizations enabled
python train_dqn.py \
  --device cuda \
  --use_amp \
  --pin_memory \
  --batch_size 128 \
  --lr 0.0002 \
  --total_steps 500000
```

**Training Progress:**
- Checkpoints saved to `checkpoints/dqn_snake_latest.pth`
- Best model saved to `checkpoints/dqn_snake_best.pth`
- Training logs in `checkpoints/training.log`
- Monitor with: `tail -f checkpoints/training.log`

### Evaluating Your Agent

#### Watch the Agent Play (with visualization)
```bash
python evaluate_dqn.py \
  --model checkpoints/dqn_snake_best.pth \
  --episodes 5 \
  --render True \
  --step_delay 0.1
```

#### Quick Performance Test (no visualization)
```bash
python evaluate_dqn.py \
  --model checkpoints/dqn_snake_best.pth \
  --episodes 10 \
  --render False
```

### Testing GPU Performance

Run the GPU example to verify optimizations:
```bash
python example_gpu.py
```

Run benchmarks to compare CPU vs GPU:
```bash
python benchmark.py
```

## 🚀 GPU Training (15-20x Faster!)

### ⚡ Maximum GPU Utilization (NEW!)

**Problem**: Standard training uses only ~1.5% of GPU capacity.  
**Solution**: Use the optimized training script for 70-90% GPU utilization!

```bash
# Achieves 70-90% GPU utilization (vs 1.5% with standard training)
python train_dqn_optimized.py --config config_gpu_optimized.yaml
```

**Features**:
- Vectorized parallel environments (8 environments running simultaneously)
- Batch action inference (process all environments in one GPU call)
- Multiple training steps per environment step (keeps GPU busy)
- Expected speedup: **8x faster** than standard GPU training, **50x faster** than CPU

📖 **Quick Start**: [QUICKSTART_GPU_OPTIMIZATION.md](QUICKSTART_GPU_OPTIMIZATION.md)  
📚 **Full Guide**: [GPU_UTILIZATION_IMPROVEMENTS.md](GPU_UTILIZATION_IMPROVEMENTS.md)

### Ready-to-Use GPU Commands

#### Quick GPU Training
```bash
# Use pre-configured GPU settings
python train_dqn.py --config config_gpu.yaml
```

#### Maximum Performance GPU Training
```bash
python train_dqn.py \
  --device cuda \
  --use_amp \
  --pin_memory \
  --batch_size 128 \
  --lr 0.0002 \
  --total_steps 500000 \
  --gradient_accumulation_steps 1
```

#### GPU Training with Limited Memory
```bash
# For GPUs with 4-6GB memory
python train_dqn.py \
  --device cuda \
  --use_amp \
  --batch_size 32 \
  --buffer_size 50000 \
  --gradient_accumulation_steps 4
```

### GPU Features Explained

- **`--use_amp`**: Automatic Mixed Precision - 2-3x faster training
- **`--pin_memory`**: Faster CPU-to-GPU data transfer
- **`--gradient_accumulation_steps`**: Simulate larger batch sizes
- **`--device cuda`**: Force GPU usage (auto-detected by default)

### Check GPU Status
```bash
# Simple check
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Detailed check
python example_gpu.py
```

**For detailed GPU optimization information, see [GPU Optimization Guide](GPU_OPTIMIZATION_GUIDE.md)**

## 📁 Project Files

- `train_dqn.py` — Main training script
- `train_dqn_optimized.py` — GPU-optimized training script (70-90% GPU utilization)
- `evaluate_dqn.py` — Evaluate trained models
- `main.py` — Manual play Snake game (Pygame)
- `snake_env.py` — Gymnasium environment for Snake
- `vec_env.py` — Vectorized environments for parallel data collection
- `dqn_agent.py` — DQN model, replay buffer, agent utilities
- `config.yaml` — CPU training configuration
- `config_gpu.yaml` — GPU-optimized configuration
- `config_gpu_optimized.yaml` — Maximum GPU utilization configuration
- `example_gpu.py` — GPU optimization demo
- `benchmark.py` — Performance benchmarking
- `config_utils.py` — Configuration management
- `logger_utils.py` — Training logging utilities
- `requirements.txt` — Python dependencies

## 💡 Tips & Best Practices
- **Start Small**: Try training with `--total_steps 100000` first to verify everything works
- **Monitor Training**: Use `tail -f checkpoints/training.log` to watch progress in real-time
- **GPU Acceleration**: Enable `--use_amp` for 2-3x faster training on modern NVIDIA GPUs
- **Smaller Grids Learn Faster**: Try 12×10 grid for faster learning, then scale to 24×20
- **Reward Shaping**: Adjust `--step_penalty`, `--food_reward`, `--death_reward` for different behaviors
- **Save GPU Memory**: Use `--gradient_accumulation_steps 4` with smaller `--batch_size 32` if you get OOM errors

## 🔧 Troubleshooting

### Training is slow
```bash
# Enable GPU optimizations
python train_dqn.py --config config_gpu.yaml

# Or manually enable AMP
python train_dqn.py --use_amp --batch_size 128
```

### CUDA Out of Memory
```bash
# Reduce batch size and buffer
python train_dqn.py --batch_size 32 --buffer_size 50000

# Or use gradient accumulation
python train_dqn.py --batch_size 32 --gradient_accumulation_steps 4
```

### Pygame Display Issues (Headless Server)
```bash
# Disable rendering
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --render False
```

### Import/Module Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 🎮 Compatibility with Pygame Snake
- Mechanics mirror the Pygame game: grid movement, no direct reverse, food spawn not on snake, same collision rules
- Training is headless for speed. Use `evaluate_dqn.py --render True` to visualize the trained agent
- You can adapt `main.py` to read actions from the agent for live visualization

## 📚 Additional Resources

- **[QUICKSTART.md](QUICKSTART.md)** - Get started in under 5 minutes
- **[GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md)** - Detailed GPU optimization guide
- **[OPTIMIZATIONS.md](OPTIMIZATIONS.md)** - Performance improvements explained
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and updates

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📄 License

See LICENSE file for details.