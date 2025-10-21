# Quick Start Guide

Get started with Snake RL in minutes!

## Installation

```bash
# Clone the repository
git clone https://github.com/CyronixXmenX/snake-ml.git
cd snake-ml

# Install dependencies
pip install -r requirements.txt
```

## Quick Training

### Option 1: Use Default Settings
```bash
python train_dqn.py
```

### Option 2: Use Configuration File (Recommended)
```bash
python train_dqn.py --config config.yaml
```

### Option 3: Custom Parameters
```bash
python train_dqn.py \
  --grid_w 24 \
  --grid_h 20 \
  --total_steps 100000 \
  --lr 0.0001 \
  --batch_size 64
```

## Monitor Training

Training logs are automatically saved to `checkpoints/training.log`. You can watch progress in real-time:

```bash
# In a separate terminal
tail -f checkpoints/training.log
```

## Evaluate Trained Agent

### Headless Evaluation
```bash
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --episodes 10
```

### Visual Evaluation (with Pygame window)
```bash
python evaluate_dqn.py \
  --model checkpoints/dqn_snake_best.pth \
  --episodes 5 \
  --render True \
  --step_delay 0.1
```

## Play Manually

Want to play Snake yourself?

```bash
python main.py
```

**Controls:**
- Arrow keys or WASD: Move
- P: Pause
- R: Restart (when game over)
- ESC or Q: Quit

## Benchmark Performance

Test the performance of your setup:

```bash
python benchmark.py
```

This will show you:
- Steps per second
- Average step/reset time
- Inference speed
- Training speed

## Configuration Tips

### For Fast Training (Testing)
```yaml
training:
  total_steps: 50000
  eval_interval: 5000

exploration:
  epsilon_decay_steps: 20000
```

### For Better Performance
```yaml
dqn:
  buffer_size: 200000
  batch_size: 128
  train_start: 20000
```

### For Smaller Grid (Faster Learning)
```yaml
environment:
  grid_width: 12
  grid_height: 10
```

## Common Issues

### CUDA Out of Memory
```bash
python train_dqn.py --device cpu
```

### Pygame Display Issues
If you're on a headless server, disable rendering:
```bash
python evaluate_dqn.py --model checkpoints/dqn_snake_best.pth --render False
```

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## File Structure

```
snake-ml/
├── train_dqn.py          # Main training script
├── evaluate_dqn.py       # Evaluation script
├── main.py               # Manual play game
├── snake_env.py          # Environment implementation
├── dqn_agent.py          # DQN agent implementation
├── config.yaml           # Configuration file
├── config_utils.py       # Config utilities
├── logger_utils.py       # Logging utilities
├── benchmark.py          # Performance benchmarks
├── requirements.txt      # Dependencies
├── checkpoints/          # Saved models (auto-created)
└── logs/                 # Training logs (auto-created)
```

## What's Next?

1. **Experiment with hyperparameters** - Edit `config.yaml`
2. **Try different reward schemes** - Adjust penalties and rewards
3. **Change grid size** - Smaller = faster learning, larger = harder
4. **Monitor with TensorBoard** - Add your own logging
5. **Create variations** - Obstacles, multiple foods, speed changes

## Resources

- `README.md` - Detailed documentation
- `OPTIMIZATIONS.md` - Performance improvements explained
- `CHANGELOG.md` - List of all changes

## Getting Help

If you encounter issues:
1. Check the logs in `checkpoints/training.log`
2. Run benchmarks to verify your setup
3. Try with smaller parameters first
4. Check GPU/CUDA availability with `torch.cuda.is_available()`

Happy training! 🐍🎮
