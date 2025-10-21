"""
Logging utilities for training and evaluation.

Provides structured logging with file and console output.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "snake_rl",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console output.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        console: Whether to also log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    
    # Console handler
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    
    return logger


class TrainingLogger:
    """
    Logger specifically for training metrics and progress.
    
    Provides convenient methods for logging training events.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_config(self, config: dict) -> None:
        """Log training configuration."""
        self.logger.info("=" * 60)
        self.logger.info("Training Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.info(f"  {key}:")
                for k, v in value.items():
                    self.logger.info(f"    {k}: {v}")
            else:
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)
    
    def log_episode(
        self, 
        episode: int, 
        return_: float, 
        length: int, 
        epsilon: float
    ) -> None:
        """Log episode completion."""
        self.logger.info(
            f"Episode {episode}: return={return_:.2f}, length={length}, eps={epsilon:.3f}"
        )
    
    def log_eval(
        self, 
        step: int, 
        avg_return: float, 
        episodes: int
    ) -> None:
        """Log evaluation results."""
        self.logger.info(
            f"Eval @ step {step}: avg_return={avg_return:.2f} ({episodes} episodes)"
        )
    
    def log_checkpoint(self, path: str, is_best: bool = False) -> None:
        """Log checkpoint save."""
        prefix = "Best checkpoint" if is_best else "Checkpoint"
        self.logger.info(f"{prefix} saved: {path}")
    
    def log_training_start(self, total_steps: int, device: str) -> None:
        """Log training start."""
        self.logger.info("=" * 60)
        self.logger.info(f"Starting training for {total_steps} steps")
        self.logger.info(f"Device: {device}")
        self.logger.info("=" * 60)
    
    def log_training_end(self, duration_min: float, best_return: float) -> None:
        """Log training completion."""
        self.logger.info("=" * 60)
        self.logger.info(f"Training completed in {duration_min:.1f} minutes")
        self.logger.info(f"Best evaluation return: {best_return:.2f}")
        self.logger.info("=" * 60)
