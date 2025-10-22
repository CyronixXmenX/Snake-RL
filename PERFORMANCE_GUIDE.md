# High-Throughput DQN Performance Guide

This guide helps you maximize GPU utilization and training throughput for Snake RL.

## 🎯 Performance Targets

Based on the high-throughput configuration with CUDA GPU:

| Metric | Target | Hardware |
|--------|--------|----------|
| **GPU Utilization** | >50% during updates | Modern NVIDIA GPU |
| **Peak GPU Utilization** | 70-90% | During gradient steps |
| **Throughput** | 500-1000 steps/sec | RTX 3060+ |
| **Throughput** | 1000-2000 steps/sec | RTX 3090/4090 |
| **Mean Return** | >5.0 @ 2M steps | 24x20 grid |
| **Episode Length** | 50+ steps | After convergence |
| **Speedup vs CPU** | 20-50x | With AMP + compile |

## 🚀 Quick Start for High Performance

### Minimal Setup (Good Performance)

```bash
python train_dqn_advanced.py \
  --device cuda \
  --batch_size 1024 \
  --n_envs 4 \
  --gradient_steps 8 \
  --use_amp \
  --profile
```

### Recommended Setup (Best Performance)

```bash
python train_dqn_advanced.py \
  --config config_high_throughput.yaml \
  --profile
```

### Maximum Throughput (GPU-Heavy)

```bash
python train_dqn_advanced.py \
  --device cuda \
  --batch_size 2048 \
  --n_envs 16 \
  --gradient_steps 32 \
  --n_step 3 \
  --hidden_size 1024 \
  --use_amp \
  --compile_model \
  --profile
```

## 📊 Understanding GPU Utilization

### Expected Patterns

**Typical training profile:**
- Environment stepping: 10-30% GPU usage (mostly CPU work)
- Training/gradient steps: 50-90% GPU usage
- Overall average: 30-60% (depends on `train_freq` and `gradient_steps`)

**Why not 100% GPU utilization?**
- Environment simulation happens on CPU
- Python overhead between steps
- Data transfer CPU ↔ GPU (minimized with GPU buffer)
- This is **normal and expected** for RL

### What to Optimize

**Focus on GPU utilization during training steps:**
- This is where the actual learning happens
- Target: >50% during gradient computation
- Peaks of 70-90% are excellent

**Monitor with:**
```bash
# Basic monitoring
watch -n 1 nvidia-smi

# Detailed monitoring during training
nvidia-smi dmon -s u
```

## ⚙️ Tuning Hyperparameters for Performance

### Batch Size

**Impact:** Largest effect on GPU utilization

- **Small (64-256):** Low GPU usage, fast iteration
- **Medium (512-1024):** Good GPU usage, balanced
- **Large (2048-4096):** High GPU usage, slower iteration

**Recommendation:**
- Start with 1024
- Increase to 2048-4096 if VRAM allows
- Reduce to 512 if OOM errors

**Trade-offs:**
- Larger batches → Better GPU utilization
- Larger batches → May need higher learning rate
- Larger batches → More memory required

### Gradient Steps

**Impact:** Number of training batches per environment collection

- **Low (1-4):** More env stepping, less GPU work
- **Medium (8-16):** Balanced env/learner time
- **High (32-64):** GPU-heavy, env becomes bottleneck

**Recommendation:**
- Start with 16
- Increase to 32 if GPU utilization is low
- Decrease to 8 if training is slow

**Rule of thumb:**
```
gradient_steps * batch_size ≈ 8000-16000
```

### Number of Environments (n_envs)

**Impact:** Parallel environment simulation (CPU)

- **Single (1):** Simple, no parallelism overhead
- **Small (4-8):** Good CPU utilization
- **Large (16-32):** Maximum CPU throughput

**Recommendation:**
- Start with 8
- Scale to match CPU cores
- Use `SubprocVecEnv` for true parallelism (SB3)

**Note:** More envs mean more transitions per gradient step

### Train Frequency

**Impact:** How often to train relative to env steps

- **High (16):** More env stepping, less training
- **Medium (4-8):** Balanced
- **Low (1):** Train every step, GPU-heavy

**Recommendation:**
- Start with 4
- Decrease to 1-2 for maximum GPU utilization
- Increase to 8-16 if env is bottleneck

### Network Size (hidden_size)

**Impact:** Model capacity and compute requirements

- **Small (256):** Fast, may underfit
- **Medium (512):** Good capacity
- **Large (1024-2048):** High capacity, more GPU work

**Recommendation:**
- Start with 512
- Increase to 1024 if GPU underutilized
- Snake is relatively simple, 512 is usually enough

## 🔧 GPU Optimization Checklist

### Essential Optimizations

✅ **Large Batch Sizes**
```bash
--batch_size 1024  # Minimum for good GPU utilization
```

✅ **Multiple Gradient Steps**
```bash
--gradient_steps 16  # Keep GPU busy between env steps
```

✅ **Mixed Precision Training (AMP)**
```bash
--use_amp  # 2-3x speedup on modern GPUs
```

✅ **GPU Replay Buffer**
- Automatically enabled when device=cuda
- Zero-copy sampling for efficiency

### Advanced Optimizations

✅ **torch.compile (PyTorch 2.0+)**
```bash
--compile_model  # 20-30% speedup after warmup
```
- Not available on Windows (requires Triton)
- First run is slower (compilation overhead)

✅ **N-step Returns**
```bash
--n_step 3  # Better learning, slightly more computation
```

✅ **Dueling Architecture**
```bash
--dueling  # Better value estimation, same speed
```

### Memory Management

**If you get CUDA OOM errors:**

1. Reduce batch size:
   ```bash
   --batch_size 512  # or 256
   ```

2. Reduce buffer size:
   ```bash
   --buffer_size 500000  # Instead of 1M
   ```

3. Use gradient accumulation:
   ```bash
   --batch_size 512 --gradient_accumulation_steps 2
   # Effective batch size = 1024
   ```

4. Reduce network size:
   ```bash
   --hidden_size 256  # Instead of 512
   ```

## 📈 Benchmarking Your Setup

### Run Performance Test

```bash
# Baseline
python train_dqn_advanced.py \
  --total_steps 100000 \
  --batch_size 64 \
  --n_envs 1 \
  --gradient_steps 1 \
  --profile \
  --log_dir /tmp/benchmark_baseline

# Optimized
python train_dqn_advanced.py \
  --total_steps 100000 \
  --batch_size 1024 \
  --n_envs 8 \
  --gradient_steps 16 \
  --use_amp \
  --profile \
  --log_dir /tmp/benchmark_optimized
```

### What to Look For

In the profiling output:

```
Performance Profile:
  Environment steps: 100,000
  Training steps: 25,000
  Environment FPS: 850.5
  Environment time: 45.2s (38.1%)    ← Should be 30-50%
  Learner time: 62.8s (52.9%)        ← Should be 40-60%
  Final GPU utilization: 58.3%       ← Target >50% during training
```

**Good balance:**
- Env time: 30-50%
- Learner time: 40-60%
- GPU util during training: >50%

## 🐛 Troubleshooting

### Low GPU Utilization (<30%)

**Symptoms:**
- GPU mostly idle during training
- CPU at 100%
- Slow training despite GPU

**Solutions:**
1. Increase batch size (1024 → 2048)
2. Increase gradient steps (8 → 16 → 32)
3. Increase hidden size (512 → 1024)
4. Reduce train_freq (4 → 2 → 1)
5. Check if CPU is bottleneck (add more envs)

### High GPU Utilization but Slow Training

**Symptoms:**
- GPU at 80%+ but low steps/sec
- Training very slow

**Solutions:**
1. Reduce batch size (may be too large)
2. Reduce network size
3. Check if data transfer is slow
4. Enable AMP if not already
5. Consider enabling torch.compile

### Training Instability

**Symptoms:**
- Loss exploding or NaN
- Q-values diverging
- Agent performing worse over time

**Solutions:**
1. Reduce learning rate (1e-4 → 5e-5)
2. Reduce batch size (1024 → 512)
3. Increase learning_starts (50k → 100k)
4. Reduce n_step (3 → 1)
5. Check reward scaling

### Out of Memory (OOM)

**Symptoms:**
- CUDA out of memory error
- Training crashes

**Solutions:**
1. Reduce batch size (1024 → 512 → 256)
2. Reduce buffer size (1M → 500k)
3. Use gradient accumulation
4. Reduce network size (512 → 256)
5. Reduce n_envs (8 → 4)

## 📊 Configuration Templates

### For Different GPU Types

**Consumer GPU (GTX 1660, RTX 2060):**
```yaml
dqn:
  batch_size: 512
  gradient_steps: 8
  hidden_size: 256

training:
  n_envs: 4

gpu_optimization:
  use_amp: true
```

**Mid-range GPU (RTX 3060, RTX 3070):**
```yaml
dqn:
  batch_size: 1024
  gradient_steps: 16
  hidden_size: 512

training:
  n_envs: 8

gpu_optimization:
  use_amp: true
  compile_model: true
```

**High-end GPU (RTX 3090, RTX 4090):**
```yaml
dqn:
  batch_size: 2048
  gradient_steps: 32
  hidden_size: 1024

training:
  n_envs: 16

gpu_optimization:
  use_amp: true
  compile_model: true
```

### For Different Training Goals

**Fast Iteration (prototyping):**
```bash
--batch_size 256 \
--n_envs 4 \
--gradient_steps 4 \
--total_steps 500000
```

**Balanced (recommended):**
```bash
--batch_size 1024 \
--n_envs 8 \
--gradient_steps 16 \
--total_steps 2000000 \
--use_amp
```

**Maximum Performance (high-end GPU):**
```bash
--batch_size 2048 \
--n_envs 16 \
--gradient_steps 32 \
--hidden_size 1024 \
--total_steps 2000000 \
--use_amp \
--compile_model
```

## 🔍 Profiling and Monitoring

### Enable Profiling

```bash
python train_dqn_advanced.py \
  --config config_high_throughput.yaml \
  --profile \
  --log_dir runs/profiled_run
```

### TensorBoard Metrics

Start TensorBoard:
```bash
tensorboard --logdir runs/
```

**Key metrics to monitor:**
- `perf/env_fps`: Environment steps per second
- `perf/env_time_pct`: % time in env stepping
- `perf/learner_time_pct`: % time in training
- `perf/gpu_utilization`: GPU usage during training
- `train/loss`: Training loss (should decrease)
- `train/return_mean`: Episode returns (should increase)

### Real-time GPU Monitoring

```bash
# Basic GPU stats (1 second refresh)
watch -n 1 nvidia-smi

# Detailed utilization monitoring
nvidia-smi dmon -s u -d 1

# GPU memory monitoring
nvidia-smi dmon -s m -d 1

# Combined
nvidia-smi dmon -s mu -d 1
```

## 📚 Further Reading

- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Double DQN Paper](https://arxiv.org/abs/1509.06461)
- [Dueling DQN Paper](https://arxiv.org/abs/1511.06581)
- [Rainbow DQN Paper](https://arxiv.org/abs/1710.02298)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## 🎓 Tips from Experience

1. **Start simple, scale up:** Begin with default settings, then optimize
2. **Profile first, optimize second:** Use `--profile` to identify bottlenecks
3. **GPU utilization ≠ speed:** Focus on steps/sec and wall-clock time
4. **Balance is key:** Env time and learner time should be roughly equal
5. **Batch size matters most:** This is your biggest lever for GPU utilization
6. **More isn't always better:** Huge batches can hurt learning stability
7. **Test on CPU first:** Ensure correctness before scaling to GPU
8. **Monitor training curves:** Learning should be smooth and monotonic
9. **Save checkpoints often:** GPU crashes can lose hours of training
10. **Use HPO for fine-tuning:** Optuna can find better configs than manual search

---

**Happy Optimizing! 🚀**
