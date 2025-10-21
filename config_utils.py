"""
Configuration utilities for loading and managing training settings.

Provides easy loading of configuration from YAML files with defaults
and command-line argument override support.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file. If None, returns empty dict.
        
    Returns:
        Dictionary with configuration settings
        
    Raises:
        FileNotFoundError: If config_path does not exist
        ImportError: If yaml is not installed
    """
    if config_path is None:
        return {}
    
    if yaml is None:
        raise ImportError("PyYAML is required for config file support. Install with: pip install pyyaml")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config or {}


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge configuration file settings with command-line arguments.
    
    Command-line arguments take precedence over config file settings.
    
    Args:
        config: Configuration dictionary from YAML
        args: Parsed command-line arguments
        
    Returns:
        Updated Namespace with merged settings
    """
    # Mapping from config structure to argument names
    config_to_args = {
        ('environment', 'grid_width'): 'grid_w',
        ('environment', 'grid_height'): 'grid_h',
        ('environment', 'step_penalty'): 'step_penalty',
        ('environment', 'food_reward'): 'food_reward',
        ('environment', 'death_reward'): 'death_reward',
        ('dqn', 'learning_rate'): 'lr',
        ('dqn', 'gamma'): 'gamma',
        ('dqn', 'batch_size'): 'batch_size',
        ('dqn', 'target_update'): 'target_update',
        ('dqn', 'buffer_size'): 'buffer_size',
        ('dqn', 'train_start'): 'train_start',
        ('training', 'total_steps'): 'total_steps',
        ('training', 'seed'): 'seed',
        ('training', 'device'): 'device',
        ('training', 'checkpoint_dir'): 'checkpoint_dir',
        ('training', 'eval_interval'): 'eval_interval',
        ('training', 'eval_episodes'): 'eval_episodes',
        ('exploration', 'epsilon_start'): 'eps_start',
        ('exploration', 'epsilon_end'): 'eps_end',
        ('exploration', 'epsilon_decay_steps'): 'eps_decay_steps',
    }
    
    # Only update args that weren't explicitly set on command line
    # (i.e., still have default values)
    parser = argparse.ArgumentParser()
    add_training_arguments(parser)
    defaults = vars(parser.parse_args([]))
    
    for (section, key), arg_name in config_to_args.items():
        if section in config and key in config[section]:
            # Only override if arg still has default value
            if hasattr(args, arg_name) and getattr(args, arg_name) == defaults.get(arg_name):
                setattr(args, arg_name, config[section][key])
    
    return args


def add_training_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add all training-related command-line arguments to parser.
    
    Args:
        parser: ArgumentParser instance to add arguments to
    """
    parser.add_argument("--config", type=str, default=None, 
                        help="Path to YAML config file")
    parser.add_argument("--grid_w", type=int, default=24, help="Grid width")
    parser.add_argument("--grid_h", type=int, default=20, help="Grid height")
    parser.add_argument("--total_steps", type=int, default=500_000, help="Total training steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--buffer_size", type=int, default=100_000, help="Replay buffer size")
    parser.add_argument("--train_start", type=int, default=10_000, help="Steps before training starts")
    parser.add_argument("--target_update", type=int, default=1000, help="Steps between target network updates")
    parser.add_argument("--eps_start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--eps_end", type=float, default=0.05, help="Final epsilon")
    parser.add_argument("--eps_decay_steps", type=int, default=200_000, help="Epsilon decay steps")
    parser.add_argument("--step_penalty", type=float, default=-0.01, help="Penalty for each step")
    parser.add_argument("--food_reward", type=float, default=1.0, help="Reward for eating food")
    parser.add_argument("--death_reward", type=float, default=-1.0, help="Penalty for dying")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], 
                        help="Compute device")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="Directory for saving checkpoints")
    parser.add_argument("--eval_interval", type=int, default=10_000, 
                        help="Steps between evaluations")
    parser.add_argument("--eval_episodes", type=int, default=5, 
                        help="Episodes per evaluation")
