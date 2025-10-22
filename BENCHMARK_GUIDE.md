# Hyperparameter Benchmark Guide

This guide explains how to use the hyperparameter benchmark tool to find optimal training settings for Snake RL.

## Overview

The `benchmark_hyperparameters.py` script performs a brute-force grid search over various hyperparameter combinations to identify the best settings for training. It runs short training sessions with different configurations and reports the top-performing settings.

## Quick Start

### Basic Usage

Run a quick benchmark with default parameters:

```bash
python benchmark_hyperparameters.py --benchmark_steps 20000
```

This will test:
- Learning rates: 0.0001, 0.0002, 0.0005
- Batch sizes: 32, 64, 128
- Gamma values: 0.95, 0.99
- Epsilon decay steps: 10000, 20000, 30000
- Distance reward scales: 0.0, 0.05, 0.1, 0.2

Total: **144 configurations** (3 × 3 × 2 × 3 × 4)

### Comprehensive Search

For more thorough results, run with more steps and multiple runs per configuration:

```bash
python benchmark_hyperparameters.py --benchmark_steps 100000 --n_runs 3
```

This will run each configuration 3 times with different seeds to get more stable results.

### Custom Parameter Ranges

Test specific hyperparameters:

```bash
python benchmark_hyperparameters.py \
  --lr 0.0001 0.0002 \
  --batch_size 32 64 128 \
  --gamma 0.99 \
  --epsilon_decay 20000 \
  --distance_reward 0.0 0.1 0.2 \
  --benchmark_steps 50000
```

This tests only 18 configurations (2 × 3 × 1 × 1 × 3).

## Command-Line Arguments

### Benchmark Settings

- `--benchmark_steps`: Training steps per configuration (default: 50000)
  - Increase for more accurate results
  - Decrease for faster exploration
  
- `--eval_interval`: Steps between evaluations (default: 10000)
  - Lower values give more frequent feedback
  
- `--eval_episodes`: Episodes per evaluation (default: 5)
  - Higher values give more stable evaluation metrics
  
- `--n_runs`: Number of runs per configuration (default: 1)
  - Increase to 3-5 for more reliable results
  - Each run uses a different random seed

### Grid Search Ranges

- `--lr`: Learning rates to test
  - Example: `--lr 0.0001 0.0002 0.0005`
  - Default: 0.0001, 0.0002, 0.0005
  
- `--batch_size`: Batch sizes to test
  - Example: `--batch_size 32 64 128`
  - Default: 32, 64, 128
  
- `--gamma`: Discount factors to test
  - Example: `--gamma 0.95 0.99`
  - Default: 0.95, 0.99
  
- `--epsilon_decay`: Epsilon decay steps to test
  - Example: `--epsilon_decay 10000 20000 30000`
  - Default: 10000, 20000, 30000
  
- `--distance_reward`: Distance reward scales to test
  - Example: `--distance_reward 0.0 0.05 0.1 0.2`
  - Default: 0.0, 0.05, 0.1, 0.2
  - 0.0 means no reward shaping (sparse rewards only)

### Environment Settings

- `--grid_w`: Grid width (default: 24)
- `--grid_h`: Grid height (default: 20)
- `--device`: Device to use - auto, cpu, or cuda (default: auto)
- `--seed`: Base random seed (default: 42)

### Output Settings

- `--output_dir`: Directory to save results (default: benchmark_results)
- `--quiet`: Suppress progress bars
- `--top_n`: Number of top configurations to display (default: 5)

## Understanding the Results

### During Execution

The script will show:
- Total configurations to test
- Progress for each configuration
- Average return and evaluation scores

Example output:
```
[1/8] Testing configuration:
  lr: 0.0001
  batch_size: 32
  gamma: 0.99
  epsilon_decay_steps: 2000
  distance_reward_scale: 0.0
  buffer_size: 50000
  target_update: 1000
  Results: avg_return=-1.083±0.000, best_eval=-1.100
```

### Final Summary

After all configurations are tested, you'll see:

1. **Top N Configurations**: Ranked by average return
   - Average Return: Higher is better (closer to positive)
   - Best Eval Return: Best evaluation performance
   - Training Time: Time taken for training
   - Parameters: Full hyperparameter configuration

2. **Best Single Run**: The single run with highest return
   - Useful if you want to know the absolute best result
   - May be less stable than averaged results

### Saved Results

Results are automatically saved to JSON files:
```
benchmark_results/benchmark_results_YYYYMMDD_HHMMSS.json
```

The JSON file contains:
- Configuration for each run
- Final average return
- Best evaluation return
- Training time
- Average episode length
- Total episodes completed
- Random seed used

You can analyze these results further with custom scripts.

## Usage Examples

### Example 1: Quick Test

Test a small set of parameters quickly:

```bash
python benchmark_hyperparameters.py \
  --benchmark_steps 10000 \
  --lr 0.0001 0.0002 \
  --batch_size 64 \
  --gamma 0.99 \
  --epsilon_decay 10000 \
  --distance_reward 0.0 0.1
```

- **Total configurations**: 4 (2 × 1 × 1 × 1 × 2)
- **Estimated time**: ~1-2 minutes

### Example 2: Thorough Search

Run a comprehensive search with multiple runs:

```bash
python benchmark_hyperparameters.py \
  --benchmark_steps 100000 \
  --n_runs 3 \
  --eval_interval 20000
```

- **Total runs**: 432 (144 configs × 3 runs)
- **Estimated time**: Several hours

### Example 3: Focus on Learning Rate

Compare different learning rates with fixed other parameters:

```bash
python benchmark_hyperparameters.py \
  --benchmark_steps 50000 \
  --lr 0.00005 0.0001 0.0002 0.0003 0.0004 0.0005 \
  --batch_size 64 \
  --gamma 0.99 \
  --epsilon_decay 20000 \
  --distance_reward 0.1 \
  --n_runs 3
```

- **Total runs**: 18 (6 learning rates × 3 runs)
- **Estimated time**: ~15-20 minutes

### Example 4: GPU Acceleration

Use GPU for faster benchmarking:

```bash
python benchmark_hyperparameters.py \
  --benchmark_steps 100000 \
  --device cuda \
  --batch_size 128 256
```

GPU can speed up training significantly (5-20x faster).

## Interpreting Results

### What to Look For

1. **Average Return**: 
   - Higher is better
   - Values > 0 indicate the agent is consistently eating food
   - Values < -1 suggest the agent is dying quickly

2. **Best Eval Return**:
   - Shows peak performance during training
   - May be higher than average if there's variance

3. **Training Time**:
   - Faster configurations allow more experimentation
   - Consider time/performance tradeoff

4. **Stability**:
   - When using `--n_runs > 1`, look at standard deviation
   - Lower std indicates more stable learning

### Common Patterns

- **Higher learning rate**: Faster initial learning, but may be unstable
- **Larger batch size**: More stable gradients, but slower per-step updates
- **Higher gamma**: Values future rewards more, may lead to better long-term play
- **Longer epsilon decay**: More exploration, may find better strategies
- **Distance reward > 0**: Dense rewards help early learning but may bias behavior

## Tips and Best Practices

### 1. Start Small

Begin with a quick benchmark to get a sense of the landscape:
```bash
python benchmark_hyperparameters.py --benchmark_steps 10000 --quiet
```

### 2. Refine Iteratively

Use results from quick tests to narrow down good ranges, then test more thoroughly:
```bash
# First pass: wide range
python benchmark_hyperparameters.py --lr 0.00001 0.0001 0.001 --benchmark_steps 20000

# Second pass: refine around best value
python benchmark_hyperparameters.py --lr 0.00005 0.0001 0.00015 0.0002 --benchmark_steps 50000 --n_runs 3
```

### 3. Use Multiple Runs for Final Selection

Always use `--n_runs 3` or higher when selecting final hyperparameters:
```bash
python benchmark_hyperparameters.py --n_runs 5 --benchmark_steps 100000
```

### 4. Consider Your Hardware

- **CPU**: Use smaller batch sizes (32-64) and fewer steps
- **GPU**: Use larger batch sizes (128-256) and more steps

### 5. Balance Search Space Size

Too many parameters = very long benchmarks. Focus on the most important ones:
- Learning rate (most important)
- Distance reward scale (affects learning dynamics)
- Epsilon decay (affects exploration)

### 6. Validate Best Configuration

After finding the best hyperparameters, run a full training session:
```bash
python train_dqn.py \
  --lr 0.0001 \
  --batch_size 64 \
  --gamma 0.99 \
  --eps_decay_steps 20000 \
  --distance_reward_scale 0.1 \
  --total_steps 500000
```

## Advanced Usage

### Custom Analysis

Load and analyze results programmatically:

```python
import json
import numpy as np

# Load results
with open('benchmark_results/benchmark_results_20231022_104001.json', 'r') as f:
    results = json.load(f)

# Find configurations with return > threshold
good_configs = [r for r in results if r['final_avg_return'] > -0.5]

# Analyze learning rate impact
lr_results = {}
for r in results:
    lr = r['config']['lr']
    if lr not in lr_results:
        lr_results[lr] = []
    lr_results[lr].append(r['final_avg_return'])

for lr, returns in lr_results.items():
    print(f"LR {lr}: avg={np.mean(returns):.3f}, std={np.std(returns):.3f}")
```

### Parallel Execution

For very large searches, you can split the work:

```bash
# Terminal 1: Test learning rates 0.0001, 0.0002
python benchmark_hyperparameters.py --lr 0.0001 0.0002 --output_dir results_part1

# Terminal 2: Test learning rates 0.0003, 0.0004
python benchmark_hyperparameters.py --lr 0.0003 0.0004 --output_dir results_part2
```

Then combine the results manually.

## Troubleshooting

### Benchmark Takes Too Long

- Reduce `--benchmark_steps` (try 10000-20000 for quick tests)
- Reduce parameter ranges (fewer values to test)
- Set `--n_runs 1` instead of multiple runs
- Use `--quiet` to disable progress bars (slightly faster)

### Out of Memory

- Reduce `--batch_size` values
- Reduce buffer size (currently fixed at 50000)
- Use CPU instead of GPU if GPU memory is limited

### Poor Results Across All Configurations

- Increase `--benchmark_steps` (may need 50000+ for meaningful learning)
- Check if environment is too difficult (large grid, sparse rewards)
- Try enabling distance rewards: `--distance_reward 0.1 0.2`

### Results Not Saved

- Check that `--output_dir` is writable
- Results are saved even if script crashes (after each config)
- Look in default `benchmark_results/` directory

## Related Tools

- `train_dqn.py`: Full training with best hyperparameters found
- `evaluate_dqn.py`: Evaluate trained models
- `benchmark_gpu.py`: GPU performance benchmarking (not hyperparameters)

## Questions?

If you find good hyperparameters or have suggestions for improving the benchmark, please open an issue or PR!
