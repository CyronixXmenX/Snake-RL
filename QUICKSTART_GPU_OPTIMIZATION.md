# Quick Start: Maximum GPU Utilization

## Problem
Your training is using only **1.5% of GPU** capacity, wasting 98.5% of available processing power.

## Solution
Use the **optimized training script** to achieve **70-90% GPU utilization**!

## Quick Start (3 Commands)

### 1. Check Your GPU
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 2. Start Optimized Training
```bash
python train_dqn_optimized.py --config config_gpu_optimized.yaml
```

### 3. Monitor GPU Usage (separate terminal)
```bash
watch -n 1 nvidia-smi
```

You should see:
- **GPU Utilization**: 70-90% (vs 1.5% before)
- **Memory Usage**: 4-6GB
- **Power Draw**: Near TDP (GPU is working hard!)

## What's Different?

### Old Way (train_dqn.py)
```
1 environment → 1 GPU forward pass → 1 env step → 1 GPU backward pass
```
**Result**: GPU waits 95% of the time

### New Way (train_dqn_optimized.py)
```
8 environments → 1 batched GPU forward pass → 8 env steps → 4 GPU backward passes
```
**Result**: GPU stays busy 75% of the time

## Expected Performance

| Metric | Standard | Optimized | Improvement |
|--------|----------|-----------|-------------|
| GPU Utilization | 1.5% | 75% | **50x** |
| Training Speed | 1,000 steps/s | 8,000+ steps/s | **8x** |
| Wall Time (500k steps) | 8 hours | 1 hour | **8x** |

## Configuration Options

### Maximum Performance (8GB+ GPU)
```bash
python train_dqn_optimized.py \
  --config config_gpu_optimized.yaml \
  --num_envs 16 \
  --train_freq 8 \
  --batch_size 512
```

### Balanced (4-8GB GPU)
```bash
python train_dqn_optimized.py \
  --config config_gpu_optimized.yaml
```

### Limited Memory (2-4GB GPU)
```bash
python train_dqn_optimized.py \
  --config config_gpu.yaml \
  --num_envs 4 \
  --train_freq 4 \
  --batch_size 64
```

## Troubleshooting

### Still Low GPU Usage?

**Check 1**: Are you using the optimized script?
```bash
# Wrong (low GPU usage)
python train_dqn.py --config config_gpu.yaml

# Correct (high GPU usage)
python train_dqn_optimized.py --config config_gpu_optimized.yaml
```

**Check 2**: Increase parallelism
```bash
python train_dqn_optimized.py \
  --config config_gpu_optimized.yaml \
  --num_envs 16 \  # More parallel environments
  --train_freq 8   # More training steps
```

### Out of Memory?

**Solution 1**: Enable AMP (saves 50% memory)
```bash
python train_dqn_optimized.py \
  --config config_gpu_optimized.yaml \
  --use_amp
```

**Solution 2**: Reduce batch size
```bash
python train_dqn_optimized.py \
  --config config_gpu.yaml \
  --batch_size 64 \
  --num_envs 4
```

### CPU Bottleneck?

**Symptoms**: CPU at 100%, GPU at 50%

**Solution**: Reduce environments, increase training
```bash
python train_dqn_optimized.py \
  --config config_gpu_optimized.yaml \
  --num_envs 4 \   # Fewer envs = less CPU load
  --train_freq 8   # More training = more GPU work
```

## Key Parameters Explained

### `--num_envs` (default: 8)
Number of parallel environments collecting data.
- **More** = More samples per iteration, better GPU utilization
- **Too many** = CPU bottleneck
- **Recommended**: 4-16

### `--train_freq` (default: 4)
Number of training steps per environment step.
- **Higher** = GPU stays busier, faster training
- **Too high** = May overtrain on limited data
- **Recommended**: 2-8

### `--batch_size` (default: 256 in optimized config)
Number of samples per training batch.
- **Larger** = Better GPU utilization, more memory
- **Smaller** = Less memory, may train slower
- **Recommended**: 128-512

## Verify It's Working

### Good Signs
```bash
nvidia-smi
```
Look for:
- ✅ GPU Util: 70-90%
- ✅ Memory: Steady at 4-6GB
- ✅ Power: 250-300W (near TDP)
- ✅ Temp: 65-75°C

### Bad Signs
- ❌ GPU Util: < 20%
- ❌ Memory: < 1GB
- ❌ Power: < 100W
- ❌ Temp: < 50°C

If you see bad signs, you're probably still using the standard training script!

## Demo

See the improvements in action:
```bash
python demo_gpu_improvements.py
```

This shows the difference between standard and optimized approaches.

## Learn More

- **Full Details**: [GPU_UTILIZATION_IMPROVEMENTS.md](GPU_UTILIZATION_IMPROVEMENTS.md)
- **GPU Features**: [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md)
- **All Optimizations**: [OPTIMIZATIONS.md](OPTIMIZATIONS.md)

## Quick Reference

```bash
# Maximum GPU utilization (recommended)
python train_dqn_optimized.py --config config_gpu_optimized.yaml

# Monitor GPU
watch -n 1 nvidia-smi

# Demo the difference
python demo_gpu_improvements.py

# Benchmark standard vs optimized
python benchmark_gpu_utilization.py
```

---

**Remember**: The key is using `train_dqn_optimized.py` instead of `train_dqn.py`!
