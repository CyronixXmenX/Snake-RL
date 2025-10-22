# Quick Start Guide

Get up and running with Snake RL DQN training in 5 minutes!

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/CyronixXmenX/Snake-RL.git
cd Snake-RL

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Choose Your Training Method

We provide three training scripts for different use cases:

### 1. Simple Training (Original)

**Best for:** Learning the codebase, quick tests

```bash
# CPU training
python train_dqn.py --total_steps 500000

# GPU training
python train_dqn.py --config config_gpu.yaml
```

### 2. High-Throughput Training (Recommended)

**Best for:** Maximum GPU utilization, production training

```bash
# Quick start with high-throughput config
python train_dqn_advanced.py --config config_high_throughput.yaml

# Or custom settings
python train_dqn_advanced.py \
  --device cuda \
  --batch_size 1024 \
  --n_envs 8 \
  --gradient_steps 16 \
  --total_steps 2000000
```

**Features:**
- Vectorized environments
- GPU profiling and monitoring
- TensorBoard logging
- Dueling DQN + n-step returns

### 3. Stable-Baselines3 (Easiest)

**Best for:** Rapid prototyping, baseline comparisons

```bash
# DQN
python train_sb3_dqn.py --algo dqn --device cuda --n_envs 8

# QR-DQN (distributional, often better)
python train_sb3_dqn.py --algo qrdqn --device cuda --n_envs 8
```

**Features:**
- Battle-tested SB3 library
- QR-DQN support
- Built-in callbacks and monitoring

## 📊 Monitor Training

### TensorBoard

```bash
# Start TensorBoard (in another terminal)
tensorboard --logdir runs/

# Open browser to: http://localhost:6006
```

### GPU Monitoring

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Or detailed monitoring
nvidia-smi dmon -s u
```

## 🎯 Evaluate Your Agent

```bash
# Watch the agent play (with visualization)
python evaluate_dqn.py \
  --model checkpoints/dqn_snake_best.pth \
  --episodes 5 \
  --render True

# Quick performance test (no visualization)
python evaluate_dqn.py \
  --model checkpoints/dqn_snake_best.pth \
  --episodes 20 \
  --render False
```

## ⚙️ Common Configurations

### Fast Prototyping (100k steps, ~5 minutes)

```bash
python train_dqn_advanced.py \
  --total_steps 100000 \
  --batch_size 256 \
  --n_envs 4 \
  --gradient_steps 4 \
  --device cuda
```

### Balanced Training (2M steps, ~30-60 minutes)

```bash
python train_dqn_advanced.py \
  --config config_high_throughput.yaml
```

### Maximum Performance (High-end GPU)

```bash
python train_dqn_advanced.py \
  --batch_size 2048 \
  --n_envs 16 \
  --gradient_steps 32 \
  --hidden_size 1024 \
  --total_steps 2000000 \
  --use_amp \
  --compile_model \
  --device cuda
```

## 🔧 Hyperparameter Optimization

Find the best hyperparameters automatically:

```bash
# Run Optuna optimization (30 trials)
python hpo_optuna.py \
  --n_trials 30 \
  --trial_steps 200000 \
  --device cuda

# Advanced: optimize rewards too
python hpo_optuna.py \
  --n_trials 50 \
  --optimize_rewards \
  --device cuda
```

Results saved to `snake_dqn_hpo_best_params.txt` and `snake_dqn_hpo_best_config.yaml`.

## 📈 Expected Results

With default high-throughput config (24×20 grid):

| Steps | Mean Return | Episode Length | Notes |
|-------|-------------|----------------|-------|
| 0-100k | -5 to 0 | 10-20 | Random exploration |
| 100k-500k | 0 to 3 | 20-40 | Learning basics |
| 500k-1M | 3 to 5 | 40-60 | Improving efficiency |
| 1M-2M | 5+ | 50-80 | Near-optimal |

## 🐛 Troubleshooting

### CUDA Out of Memory

```bash
# Reduce batch size
--batch_size 512  # or 256

# Or use gradient accumulation
--batch_size 512 --gradient_accumulation_steps 2
```

### Training Too Slow

```bash
# Enable GPU optimizations
--use_amp --compile_model

# Increase batch size and gradient steps
--batch_size 1024 --gradient_steps 16
```

### Agent Not Learning

```bash
# Increase training duration
--total_steps 2000000

# Check epsilon decay
--eps_decay_steps 1000000

# Enable reward shaping (already enabled by default)
--distance_reward_scale 0.1
```

### torch.compile Warning (Windows)

This is expected - torch.compile requires Triton which isn't available on Windows. Either:
```bash
# Disable compile in config
compile_model: false

# Or ignore the warning (training still works)
```

## 📚 Next Steps

1. **Read the Performance Guide:** `PERFORMANCE_GUIDE.md`
   - Learn how to maximize GPU utilization
   - Understand profiling metrics
   - Tune hyperparameters for your GPU

2. **Experiment with Configurations:** `config_high_throughput.yaml`
   - Adjust batch sizes
   - Try different n-step values
   - Tune exploration schedule

3. **Run HPO:** Find optimal hyperparameters for your setup
   ```bash
   python hpo_optuna.py --n_trials 30 --device cuda
   ```

4. **Check the Full README:** `README.md`
   - Complete feature list
   - Detailed configuration options
   - Environment customization

## 🎓 Tips for Success

1. **Start small:** Test with 100k steps first
2. **Monitor training:** Use TensorBoard and GPU monitoring
3. **Enable profiling:** Use `--profile` to identify bottlenecks
4. **Use configs:** Easier to manage than long CLI commands
5. **Save checkpoints:** Training can be interrupted
6. **Compare methods:** Try all three training scripts
7. **Optimize GPU:** Follow the Performance Guide
8. **Use HPO:** Don't guess hyperparameters

## 💡 Example Workflow

```bash
# 1. Quick test (verify everything works)
python train_dqn_advanced.py --total_steps 10000 --device cuda

# 2. Short training (verify learning)
python train_dqn_advanced.py --total_steps 100000 --device cuda --profile

# 3. Full training (optimized)
python train_dqn_advanced.py --config config_high_throughput.yaml

# 4. Monitor in real-time
tensorboard --logdir runs/ &
watch -n 1 nvidia-smi

# 5. Evaluate best model
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --episodes 10

# 6. Optimize hyperparameters (optional)
python hpo_optuna.py --n_trials 30 --device cuda
```

---

**Questions? Check the full README or open an issue!**

Happy Training! 🐍🎮🚀
