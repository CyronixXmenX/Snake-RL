"""
Vectorized environment wrapper for parallel data collection.

This module implements vectorized environments to collect multiple experiences
in parallel, significantly improving GPU utilization by batching operations.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np
from snake_env import SnakeEnv


class VectorizedSnakeEnv:
    """
    Vectorized Snake environment for parallel experience collection.
    
    Runs multiple Snake environments in parallel to increase data collection
    throughput and reduce CPU-GPU synchronization overhead.
    
    Args:
        num_envs: Number of parallel environments
        grid_w: Width of each game grid
        grid_h: Height of each game grid
        step_penalty: Negative reward per step
        food_reward: Reward for eating food
        death_reward: Penalty for dying
        max_steps_multiplier: Max episode length multiplier
    """
    
    def __init__(
        self,
        num_envs: int,
        grid_w: int = 24,
        grid_h: int = 20,
        step_penalty: float = -0.01,
        food_reward: float = 1.0,
        death_reward: float = -1.0,
        max_steps_multiplier: float = 4.0,
    ) -> None:
        self.num_envs = num_envs
        self.envs = [
            SnakeEnv(
                grid_w=grid_w,
                grid_h=grid_h,
                step_penalty=step_penalty,
                food_reward=food_reward,
                death_reward=death_reward,
                max_steps_multiplier=max_steps_multiplier,
                render_mode="none",
            )
            for _ in range(num_envs)
        ]
        
        # Cache observation shape
        self.obs_shape = self.envs[0].observation_space.shape
        self.num_actions = self.envs[0].action_space.n
        
    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """
        Reset all environments.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Stacked observations from all environments, shape (num_envs, C, H, W)
        """
        observations = []
        for i, env in enumerate(self.envs):
            env_seed = seed + i if seed is not None else None
            obs, _ = env.reset(seed=env_seed)
            observations.append(obs)
        return np.stack(observations, axis=0)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List]:
        """
        Step all environments with given actions.
        
        Args:
            actions: Array of actions for each environment, shape (num_envs,)
            
        Returns:
            Tuple of (observations, rewards, terminateds, truncateds, infos)
            - observations: shape (num_envs, C, H, W)
            - rewards: shape (num_envs,)
            - terminateds: shape (num_envs,)
            - truncateds: shape (num_envs,)
            - infos: List of info dicts
        """
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            obs, reward, terminated, truncated, info = env.step(int(action))
            
            # Auto-reset on episode end
            if terminated or truncated:
                obs, _ = env.reset()
                
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)
        
        return (
            np.stack(observations, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(terminateds, dtype=np.float32),
            np.array(truncateds, dtype=np.float32),
            infos
        )
    
    def close(self) -> None:
        """Close all environments."""
        for env in self.envs:
            env.close()
