# Snake RL with Deep Q-Network (DQN)

Train a Deep Q-Network agent to play Snake using reinforcement learning. The implementation uses PyTorch and follows best practices for stable DQN training with a **fast-first** approach optimized for rapid iteration.

## 🚀 Quick Start (Fast Mode - ≤5 minutes)

The fastest way to train an agent and see results:

```bash
make fast
```

This runs a fast training session with:
- **Duration**: ≤5 minutes (50,000 steps or 300 seconds, whichever comes first)
- **Settings**: batch_size=256, gradient_steps=2, n_step=1, train_freq=4
- **Output**: Logs saved to `runs/fast/<timestamp>/`

### View Training Metrics

```bash
make tensorboard
```

Then open http://localhost:6006 in your browser to see:
- Episode returns and lengths over time
- Training performance (steps/sec, updates/sec, samples/sec)
- Timing breakdown (env stepping vs learning)
- GPU utilization (if available)
- Loss and other training metrics

### Check CSV Metrics

```bash
cat runs/fast/*/metrics.csv | head -5
```

The CSV contains detailed per-interval metrics including timing, performance, and hyperparameters.

---

## 📊 Performance Mode (Opt-in)

For better sample efficiency and GPU utilization at the cost of slower iteration:

```bash
make perf
```

This uses heavier update settings:
- **Batch size**: 1024 (vs 256 in fast mode)
- **Gradient steps**: 8 (vs 2 in fast mode)
- **N-step returns**: 3 (vs 1 in fast mode)
- **AMP**: Enabled (mixed precision for faster GPU training)
- **Duration**: Up to 1 hour or 500,000 steps

**Trade-offs:**
- ✅ Higher samples/sec (more data processed per second during updates)
- ✅ Better GPU utilization (especially on modern GPUs)
- ✅ Often better sample efficiency (learns faster per environment step)
- ❌ Lower environment steps/sec (env waits longer for updates)
- ❌ Requires more memory

---

## ⚙️ Configuration Flags

You can customize training with these flags:

### Core Settings
- `--device {auto|cuda|cpu}` - Device selection (auto prefers CUDA if available)
- `--total_steps N` - Maximum environment steps (default: 50000)
- `--max_seconds N` - Wall-clock timeout in seconds (default: 300)
- `--seed N` - Random seed for reproducibility (default: 42)

### DQN Hyperparameters (Fast-First Defaults)
- `--batch_size N` - Batch size for learning (default: 256, try 512-1024 for perf mode)
- `--gradient_steps N` - Gradient updates per training call (default: 2, try 4-8 for perf)
- `--n_step N` - N-step returns (default: 1, try 3 for perf mode)
- `--train_freq N` - Train every N env steps (default: 4)
- `--lr F` - Learning rate (default: 0.0001)
- `--buffer_size N` - Replay buffer size (default: 100000)

### Logging
- `--log_interval N` - Log metrics every N steps (default: 1000)
- `--log_dir DIR` - Base directory for logs (default: runs)
- `--exp_name NAME` - Experiment name (default: timestamp)

### Optional Optimizations (Default OFF)
- `--use_amp` - Enable automatic mixed precision (recommended for GPU)
- `--compile` - Enable torch.compile for ~20% speedup (PyTorch 2.0+, Linux/macOS only)
- `--profile` - Enable detailed profiling

### Example: Custom Fast Run
```bash
python train_dqn_advanced.py \
  --device cuda \
  --total_steps 100000 \
  --max_seconds 600 \
  --batch_size 512 \
  --gradient_steps 4 \
  --log_interval 2000
```

---

## 📈 Understanding Metrics

### CSV Schema
The `metrics.csv` file contains the following columns (in order):

| Column | Description |
|--------|-------------|
| `step` | Current training step |
| `episodes` | Total episodes completed |
| `episode_return_mean` | Average return over last 100 episodes |
| `episode_length_mean` | Average episode length over last 100 episodes |
| `steps_per_sec` | Environment steps per second (rolling average) |
| `updates_per_sec` | Optimizer updates per second (rolling average) |
| `samples_per_sec` | Samples processed per second (updates × batch_size) |
| `time_env_ms_per_step` | Average time per env step in milliseconds |
| `time_learn_ms_per_update` | Average time per optimizer update in milliseconds |
| `replay_size` | Current replay buffer size |
| `epsilon` | Current exploration rate |
| `loss_q` | Q-value loss |
| `td_error_mean` | Mean TD error (if tracked) |
| `gpu_util` | GPU utilization percentage (if available) |
| `device` | Device used (cpu/cuda) |
| `batch_size` | Batch size used |
| `gradient_steps` | Gradient steps per training call |
| `n_envs` | Number of parallel environments |
| `n_step` | N-step returns |
| `seed` | Random seed |

### Key Metrics to Watch

**Episode Return** (`episode_return_mean`): Higher is better. Should gradually increase from negative values (dying quickly) to positive values (surviving and eating food).

**Steps per Second** (`steps_per_sec`): Higher means faster training iteration. Fast mode should achieve 50-200 steps/sec on CPU, 200-1000+ on GPU.

**Updates per Second** (`updates_per_sec`): Number of gradient updates per second. Fast mode prioritizes this being small but frequent.

**Samples per Second** (`samples_per_sec`): Total samples processed per second (updates × batch_size). Performance mode optimizes for this metric.

**Timing Split**: Compare `time_env_ms_per_step` vs `time_learn_ms_per_update` to see where time is spent.

---

## 🔧 Troubleshooting

### Low steps/sec (slow training)
- **Reduce** `gradient_steps` (try 1 or 2)
- **Reduce** `batch_size` (try 128 or 256)
- Keep `n_envs=1` (vectorization often slower for tiny Snake env)
- Disable `--use_amp`, `--compile`, `--profile`
- Check if another process is using GPU/CPU

### Low GPU utilization (<30%)
- **Increase** `batch_size` (try 512, 1024, or higher)
- **Increase** `gradient_steps` (try 4, 8, or 16)
- Prefer "fewer, fatter" updates: increase `train_freq` to 8 or 16, match with higher `gradient_steps`
- Enable `--use_amp` for mixed precision
- Consider increasing `--hidden_size` (try 512 or 1024)

### Agent not learning
- Check that `episode_return_mean` is improving over time
- Increase `total_steps` (try 200k-500k)
- Verify epsilon decay isn't too fast (check `epsilon` column in CSV)
- Try different `--seed` values
- Ensure rewards are balanced (not all negative)

### CUDA out of memory
- Reduce `batch_size` (try 128 or 64)
- Reduce `buffer_size` (try 50000)
- Disable `--use_amp` (counter-intuitive but can help in some cases)

### torch.compile warning on Windows
- Disable `--compile` flag (not supported on Windows due to missing Triton)
- This is expected behavior; training still works without compile

---

## 💻 Installation

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
   python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
   ```

4. **Run a quick test:**
   ```bash
   make fast
   ```

---

## 📊 Features

### Fast-First Training Philosophy
- **Rapid iteration**: Default settings optimized for ≤5 minute runs
- **Wall-clock timeout**: Training stops at either `total_steps` OR `max_seconds` (whichever comes first)
- **Comprehensive logging**: CSV metrics + TensorBoard with timing instrumentation
- **Optional performance mode**: Opt-in heavier settings for better sample efficiency

### Environment
- **Gymnasium-compatible** Snake environment with optimized collision detection
- **3-channel observation**: [head, body, food] on a configurable H×W grid
- **Discrete actions**: 0=Up, 1=Down, 2=Left, 3=Right
- **Smart action handling**: Reverse-direction moves are ignored (prevents instant death)
- **Configurable rewards**: 
  - +1.0 for eating food (default)
  - -1.0 for dying (collision)
  - -0.01 per step (time penalty)

### DQN Agent (Fast-First Baseline)
- **Double DQN** to reduce Q-value overestimation (always enabled)
- **Dueling architecture** for better value estimation (on by default)
- **CNN architecture** optimized for small grid environments
- **Experience replay** with memory-efficient uint8 storage
- **Pinned memory** + `non_blocking=True` for efficient GPU transfers
- **Target network** with hard updates (default: every 10k steps)
- **Epsilon-greedy exploration** with linear decay (1.0 → 0.01)
- **Gradient clipping** to prevent instability
- **N-step returns** (configurable, default: 1 for speed)

### Logging & Metrics
- **CSV logging**: Detailed metrics with exact schema (step, episodes, returns, timing, etc.)
- **TensorBoard**: Real-time visualization of training progress
- **Timing instrumentation**: Separate tracking for env stepping vs learner updates
- **Performance metrics**: steps/sec, updates/sec, samples/sec
- **Optional GPU monitoring**: Utilization tracking via pynvml

---

## 🔬 Reproducibility

For reproducible results, use the `--seed` flag:

```bash
python train_dqn_advanced.py --seed 42
```

The training script sets seeds for:
- Python's `random` module
- NumPy
- PyTorch (including CUDA if available)
- Environment resets

**Note**: Even with the same seed, results may vary slightly across different hardware (especially GPU models) due to floating-point precision differences and non-deterministic GPU operations.

---

## ⚙️ Alternative Training Scripts

This repository also includes other training scripts for different use cases:

### Basic Training (Config-based)
```bash
python train_dqn.py --config config.yaml
```

Uses YAML config files for hyperparameters. See `config.yaml` and `config_gpu.yaml` for examples.

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
  loop_penalty: -0.05  # Penalty for circular movement patterns
  exploration_reward_scale: 0.02  # Reward for visiting new/less-visited cells
  loop_detection_window: 8  # Number of recent positions to track for loop detection

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
- **When to use**: Always enable on PyTorch 2.0+ for additional 20-30% speedup (Linux/macOS only)
- **Note**: First run is slower due to compilation, subsequent runs are faster
- **Benefit**: Optimized CUDA kernels, graph optimizations
- **Windows limitation**: Not supported on Windows (requires Triton). The agent will automatically fall back to eager mode with a warning if enabled on Windows

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
  --step_penalty -0.01 \               # Encourages efficiency
  --food_reward 1.0 \                  # Reward for eating
  --death_reward -1.0 \                # Penalty for dying
  --distance_reward_scale 0.1 \        # Guides agent toward food (0 to disable)
  --loop_penalty -0.05 \               # Penalizes circular movement
  --exploration_reward_scale 0.02 \    # Rewards exploring new areas
  --loop_detection_window 8            # Number of positions to track for loops
```

**Note**: 
- The `distance_reward_scale` parameter provides dense reward signals by rewarding the agent for moving closer to food. This significantly improves learning by giving immediate feedback. Set to 0.0 to disable reward shaping.
- The `loop_penalty` parameter helps prevent the snake from going in circles by penalizing revisiting recent positions. Set to 0.0 to disable.
- The `exploration_reward_scale` parameter encourages the snake to explore different areas of the grid using a heatmap system. Higher values promote more exploration. Set to 0.0 to disable.
- The `loop_detection_window` parameter controls how many recent positions are tracked for loop detection. Larger values detect larger loops but use more memory.

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

### torch.compile warning on Windows
If you see a warning about torch.compile not being supported on Windows:
```bash
# Disable compile_model in your config
python train_dqn.py --config config.yaml --compile_model False
```
Or create a Windows-specific config file with `compile_model: false`. This is expected behavior as Triton (required for torch.compile) is not available on Windows.

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

## 🚄 High-Throughput Training (NEW!)

We now provide advanced training scripts optimized for maximum GPU utilization and throughput:

### Advanced DQN Training

High-throughput training with vectorized environments, profiling, and GPU monitoring:

```bash
# High-throughput training (recommended for GPU)
python train_dqn_advanced.py \
  --config config_high_throughput.yaml \
  --profile \
  --log_dir runs/high_throughput

# Custom high-throughput setup
python train_dqn_advanced.py \
  --device cuda \
  --batch_size 1024 \
  --n_envs 8 \
  --gradient_steps 16 \
  --n_step 3 \
  --hidden_size 512 \
  --total_steps 2000000 \
  --profile
```

**Features:**
- ✅ **Dueling DQN** architecture by default
- ✅ **N-step returns** (n=1-5) for faster learning
- ✅ **Multiple gradient steps** per environment step for better GPU utilization
- ✅ **Vectorized environments** (8-16 parallel envs)
- ✅ **Large batch sizes** (512-4096) for GPU efficiency
- ✅ **GPU utilization monitoring** with pynvml
- ✅ **TensorBoard logging** with comprehensive metrics
- ✅ **Performance profiling** (env/learner time split, FPS)

**Expected Performance:**
- GPU utilization: **>50% during training steps** (peaks can be higher)
- Throughput: **500-1000+ steps/sec** (depends on GPU)
- Speedup: **20-50x faster** than CPU baseline
- Mean return target: **>5.0 within 2M steps**

### Stable-Baselines3 Training

Use the battle-tested SB3 library for baseline comparisons:

```bash
# Standard DQN with SB3
python train_sb3_dqn.py \
  --algo dqn \
  --batch_size 1024 \
  --n_envs 4 \
  --gradient_steps 16 \
  --device cuda

# QR-DQN (distributional, often stronger)
python train_sb3_dqn.py \
  --algo qrdqn \
  --batch_size 2048 \
  --n_envs 8 \
  --gradient_steps 32 \
  --device cuda
```

**Why use SB3?**
- ✅ Production-ready, well-tested implementation
- ✅ Support for QR-DQN (distributional DQN)
- ✅ Easy to use with minimal code
- ✅ Great for baselines and comparisons

### Hyperparameter Optimization

Automatically find the best hyperparameters using Optuna:

```bash
# Run HPO study (30 trials)
python hpo_optuna.py \
  --n_trials 30 \
  --trial_steps 200000 \
  --device cuda \
  --study_name snake_hpo

# Advanced HPO (optimize rewards too)
python hpo_optuna.py \
  --n_trials 50 \
  --trial_steps 200000 \
  --optimize_rewards \
  --n_jobs 2 \
  --device cuda
```

**Search space includes:**
- Learning rate, batch size, hidden size
- N-step returns, gradient steps
- Target update interval, gamma
- Exploration schedule
- Optional: reward shaping parameters

---

## 📈 Performance Targets

Based on the high-throughput configuration, you should achieve:

| Metric | Target | Notes |
|--------|--------|-------|
| **GPU Utilization** | >50% during updates | Peaks can be 70-90% |
| **Throughput** | 500-1000 steps/sec | GPU-dependent |
| **Mean Return** | >5.0 @ 2M steps | 24x20 grid |
| **Episode Length** | 50+ steps | After convergence |
| **Speedup vs CPU** | 20-50x | With AMP + compile |

### Optimization Tips

**Maximize GPU Utilization:**
1. Use large batch sizes (1024-4096)
2. Multiple gradient steps per env step (8-32)
3. Vectorized environments (8-16 parallel)
4. Enable AMP (mixed precision training)
5. Enable torch.compile (PyTorch 2.0+)
6. Keep data on GPU (replay buffer on GPU)

**If GPU utilization is low (<30%):**
- Increase batch size
- Increase gradient steps
- Increase network size (hidden_size)
- Reduce train_freq (train more often)
- Check for CPU bottlenecks in env stepping

**If training is unstable:**
- Reduce learning rate
- Use smaller batch sizes initially
- Increase learning_starts (prefill buffer more)
- Enable gradient clipping (already enabled at 10.0)
- Reduce n_step (try n=1 or n=3)

---

## 🔧 Advanced Configuration

### High-Throughput Config Template

See `config_high_throughput.yaml` for a complete example:

```yaml
dqn:
  learning_rate: 0.0003
  batch_size: 1024  # Scale to 2048-4096 for modern GPUs
  buffer_size: 1000000
  learning_starts: 50000
  hidden_size: 512
  n_step: 3
  gradient_steps: 16
  dueling: true

training:
  n_envs: 8  # Parallel environments
  train_freq: 4  # Train every N env steps
  device: cuda

gpu_optimization:
  use_amp: true
  compile_model: true
```

### CLI Options (Advanced Training)

```
--n_envs N              Number of parallel environments (default: 1)
--train_freq N          Train every N environment steps (default: 4)
--gradient_steps N      Gradient steps per training call (default: 1)
--n_step N              N-step returns, 1-5 recommended (default: 1)
--dueling               Use Dueling DQN architecture (default: True)
--hidden_size N         Hidden layer size (default: 512)
--profile               Enable detailed profiling
--log_dir DIR           TensorBoard log directory
```

---

## 📊 Monitoring & Profiling

### TensorBoard

View training metrics in real-time:

```bash
# Start TensorBoard
tensorboard --logdir runs/

# Then open: http://localhost:6006
```

**Metrics logged:**
- Training: return, loss, epsilon, buffer size
- Evaluation: return, episode length
- Performance: env FPS, env/learner time split
- GPU: utilization, memory usage (if pynvml available)

### GPU Utilization

Monitor GPU during training:

```bash
# In another terminal
watch -n 1 nvidia-smi

# Or use detailed monitoring
nvidia-smi dmon -s u
```

**Expected patterns:**
- During environment steps: Low GPU usage (10-30%)
- During training steps: High GPU usage (50-90%)
- Overall average: 30-60% (depending on train_freq)

---

## 🔬 Benchmarking

Compare different configurations:

```bash
# Baseline (small batch, single env)
python train_dqn_advanced.py \
  --batch_size 64 --n_envs 1 --gradient_steps 1 \
  --total_steps 500000 --profile

# High-throughput (large batch, multi-env)
python train_dqn_advanced.py \
  --batch_size 1024 --n_envs 8 --gradient_steps 16 \
  --total_steps 500000 --profile

# Ultra (maximum GPU utilization)
python train_dqn_advanced.py \
  --batch_size 2048 --n_envs 16 --gradient_steps 32 \
  --hidden_size 1024 --total_steps 500000 --profile
```

Compare wall-clock time and final performance.

---

## 🔗 Additional Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Gymnasium API](https://gymnasium.farama.org/)
- [DQN Paper](https://arxiv.org/abs/1312.5602) - Original Deep Q-Learning
- [Double DQN Paper](https://arxiv.org/abs/1509.06461) - Improved Q-Learning
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581) - Dueling Network Architectures
- [Rainbow Paper](https://arxiv.org/abs/1710.02298) - Combining Improvements in DQN
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)

---

**Happy Training! 🐍🎮🚀**
