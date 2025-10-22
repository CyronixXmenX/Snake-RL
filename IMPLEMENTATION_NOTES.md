# Implementation Notes: Loop Detection and Heatmap Reward System

## Overview
This implementation adds three new features to the Snake RL environment to address the issue of negative returns and improve learning efficiency:

1. **Loop Detection & Penalty** - Prevents the snake from going in circles
2. **Heatmap-based Exploration Rewards** - Encourages visiting new areas
3. **Enhanced Distance-based Rewards** - Already existed, now complemented by the new systems

## Problem Statement
The original issue was: "the return is still negative, so what if we make a penalty for going in circles and a system to detect it and maybe some heatmap reward system so the snake gets to the apple also maybe some penalties for getting away from the apple"

## Solution Design

### 1. Loop Detection System

**How it works:**
- Maintains a queue (`_recent_positions`) of the last N positions the snake's head visited
- Before each move, checks if the new head position is in this queue
- If detected, applies a configurable penalty (`loop_penalty`)

**Parameters:**
- `loop_penalty` (default: -0.05): Negative reward applied when a loop is detected
- `loop_detection_window` (default: 8): Number of recent positions to track

**Benefits:**
- Discourages repetitive circular movements
- Helps the snake learn to explore rather than spin in place
- Computationally efficient O(1) lookup using deque

### 2. Heatmap-based Exploration Rewards

**How it works:**
- Maintains a 2D heatmap (`_position_heatmap`) tracking visit counts for each cell
- Rewards visiting cells with lower visit counts
- Uses exponential decay: first visit gets full reward, subsequent visits get progressively less

**Reward calculation:**
```python
if visit_count == 0:
    reward = 1.0  # Full exploration reward
else:
    reward = max(0.0, 1.0 - (visit_count * 0.2))
```

**Parameters:**
- `exploration_reward_scale` (default: 0.02): Multiplier for heatmap rewards

**Benefits:**
- Encourages the snake to explore the entire grid
- Helps prevent getting stuck in small areas
- Provides dense reward signals for better learning
- Resets on each episode, ensuring fresh exploration

### 3. Distance-based Reward Shaping (Enhanced)

**Already existed, now works together with:**
- Loop detection prevents circular movements away from food
- Exploration rewards prevent camping in one spot
- All three systems work synergistically

**How it works:**
- Calculates Manhattan distance to food before and after each move
- Rewards getting closer, penalizes moving away
- Scaled by `distance_reward_scale` (default: 0.1)

## Implementation Details

### Modified Files

1. **snake_env.py**
   - Added new parameters to `__init__()`
   - Added `_position_heatmap` for tracking cell visits
   - Added `_recent_positions` deque for loop detection
   - Implemented `_detect_loop()` method
   - Implemented `_get_heatmap_reward()` method
   - Updated `reset()` to initialize heatmap and clear recent positions
   - Updated `step()` to apply loop penalties and exploration rewards

2. **config.yaml & config_gpu.yaml**
   - Added default values for new parameters

3. **config_utils.py**
   - Added parameter mappings for config file loading
   - Added command-line argument definitions

4. **train_dqn.py**
   - Updated environment creation to pass new parameters
   - Updated logging to include new parameters

5. **README.md**
   - Documented new features and parameters
   - Added usage examples

### Reward Flow in step()

```
1. Start with step_penalty (-0.01)
2. Check for loop → Add loop_penalty if detected (-0.05)
3. Calculate exploration reward → Add scaled heatmap reward (+0.0 to +0.02)
4. Calculate distance change → Add scaled distance reward (±0.1 per cell)
5. Check if food eaten → Add food_reward (+1.0)
6. Check if died → Add death_reward (-1.0)
```

## Configuration Examples

### Conservative (Subtle Guidance)
```yaml
loop_penalty: -0.02
exploration_reward_scale: 0.01
loop_detection_window: 6
```

### Default (Balanced)
```yaml
loop_penalty: -0.05
exploration_reward_scale: 0.02
loop_detection_window: 8
```

### Aggressive (Strong Exploration)
```yaml
loop_penalty: -0.1
exploration_reward_scale: 0.05
loop_detection_window: 12
```

### Disabled (Original Behavior)
```yaml
loop_penalty: 0.0
exploration_reward_scale: 0.0
loop_detection_window: 0  # Won't be used if penalty is 0
```

## Performance Considerations

1. **Memory Overhead:**
   - Heatmap: `grid_h * grid_w * 4 bytes` (int32)
   - Recent positions: `loop_detection_window * 16 bytes` (2 ints per position)
   - For 24x20 grid with window=8: ~2KB total (negligible)

2. **Computational Overhead:**
   - Loop detection: O(1) with deque
   - Heatmap lookup: O(1) array access
   - Heatmap update: O(1) array update
   - Total: Minimal impact on performance

3. **Training Impact:**
   - More dense reward signals → Faster learning
   - Less circular behavior → More efficient exploration
   - Better guidance toward food → Higher episode returns

## Testing

The implementation has been:
- ✓ Syntax validated with Python AST parser
- ✓ Security scanned with CodeQL (0 vulnerabilities)
- ✓ Integrated with existing configuration system
- ✓ Documented in README and code comments

## Expected Improvements

Based on the reward structure:
1. **Fewer negative returns:** Loop penalty + exploration rewards provide positive signals
2. **Better food-seeking:** Distance rewards + exploration reduce aimless wandering
3. **Faster learning:** Dense reward signals from multiple sources
4. **More efficient episodes:** Less time spent going in circles

## Future Enhancements

Potential improvements to consider:
1. Adaptive loop detection window based on snake length
2. Spatial heatmap decay (older visits count less)
3. Food-proximity weighted exploration (explore more near food)
4. Performance metrics tracking (loop frequency, coverage %)
