"""
Snake Gymnasium Environment for Reinforcement Learning.

This module implements a Snake game environment compatible with Gymnasium API,
designed for training Deep Q-Networks and other RL algorithms.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Optional/lazy pygame import for render()
try:
    import pygame
except ImportError:
    pygame = None


Direction = Tuple[int, int]
Pos = Tuple[int, int]


def _is_opposite(a: Direction, b: Direction) -> bool:
    """Check if two directions are opposite to each other."""
    return a[0] == -b[0] and a[1] == -b[1]


class SnakeEnv(gym.Env):
    """
    A Gymnasium environment for the Snake game.
    
    The environment features:
    - 3-channel observation: [head, body, food] on a H×W grid
    - Discrete actions: 0=Up, 1=Down, 2=Left, 3=Right
    - Reverse-direction input is ignored (like the Pygame game)
    - Rewards: configurable for eating food, dying, and time penalty
    
    Args:
        grid_w: Width of the game grid in cells
        grid_h: Height of the game grid in cells
        step_penalty: Negative reward given at each step (time penalty)
        food_reward: Positive reward for eating food
        death_reward: Negative reward for collision/death
        max_steps_multiplier: Maximum episode steps as multiple of grid size
        render_mode: Either "human" for pygame rendering or "none"
        cell_size: Pixel size of each cell for rendering
    """
    
    metadata = {"render_modes": ["human", "none"], "render_fps": 12}
    
    # Action directions mapping
    ACTION_DIRS: Tuple[Direction, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def __init__(
        self,
        grid_w: int = 24,
        grid_h: int = 20,
        step_penalty: float = -0.01,
        food_reward: float = 1.0,
        death_reward: float = -1.0,
        max_steps_multiplier: float = 4.0,
        render_mode: str = "none",
        cell_size: int = 24,
    ) -> None:
        super().__init__()
        
        if render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")
        
        self.grid_w = int(grid_w)
        self.grid_h = int(grid_h)
        self.step_penalty = float(step_penalty)
        self.food_reward = float(food_reward)
        self.death_reward = float(death_reward)
        self.max_steps_multiplier = float(max_steps_multiplier)
        self.render_mode = render_mode
        self.cell_size = int(cell_size)

        # Observation: 3xH x W uint8 in [0, 255] -> normalized to [0,1] by agent
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, self.grid_h, self.grid_w), dtype=np.uint8
        )
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)

        # Internal state
        self.snake: List[Pos] = []
        self.direction: Direction = (1, 0)
        self.food: Optional[Pos] = None
        self.steps = 0
        self.max_steps = int(self.grid_w * self.grid_h * self.max_steps_multiplier)
        self._rng = np.random.RandomState()  # set in reset by gymnasium seeding
        self._snake_set: set[Pos] = set()  # Cache for O(1) collision checks

        # Render members
        self._screen = None
        self._clock = None

    def seed(self, seed: Optional[int] = None) -> None:
        """Set the random seed for reproducibility."""
        self._rng = np.random.RandomState(seed)

    def _spawn_food(self) -> None:
        """
        Spawn food at a random empty location.
        
        If the grid is full (win condition), food is set to None.
        """
        # If full, no place to spawn -> win condition
        if len(self.snake) >= self.grid_w * self.grid_h:
            self.food = None
            return
        
        # Use cached snake set for faster collision detection
        while True:
            p = (int(self._rng.randint(0, self.grid_w)), int(self._rng.randint(0, self.grid_h)))
            if p not in self._snake_set:
                self.food = p
                return

    def _reset_snake(self) -> None:
        """Initialize the snake at the center of the grid."""
        cx, cy = self.grid_w // 2, self.grid_h // 2
        length = 4
        # body from left to right; head at index 0 moving right
        self.snake = [(cx - i, cy) for i in range(length)]
        self.direction = (1, 0)
        self._snake_set = set(self.snake)  # Update cached set

    def _get_obs(self) -> np.ndarray:
        """
        Generate the observation as a 3-channel image.
        
        Returns:
            np.ndarray: Shape (3, H, W) with channels [head, body, food]
        """
        # Channels: [head, body, food]
        obs = np.zeros((3, self.grid_h, self.grid_w), dtype=np.uint8)
        if self.snake:
            hx, hy = self.snake[0]
            if 0 <= hx < self.grid_w and 0 <= hy < self.grid_h:
                obs[0, hy, hx] = 255
            for (x, y) in self.snake[1:]:
                if 0 <= x < self.grid_w and 0 <= y < self.grid_h:
                    obs[1, y, x] = 255
        if self.food:
            fx, fy = self.food
            obs[2, fy, fx] = 255
        return obs

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observation, info dict)
        """
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self._reset_snake()
        self._spawn_food()
        obs = self._get_obs()
        info: Dict[str, Any] = {}
        if self.render_mode == "human":
            self._ensure_renderer()
            self._render_pygame()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Integer action (0=Up, 1=Down, 2=Left, 3=Right)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.steps += 1

        # Determine movement direction (prevent reverse)
        intended = self.ACTION_DIRS[action]
        # Ignore reversing directly into itself, like the Pygame version
        if len(self.snake) > 1 and _is_opposite(self.direction, intended):
            move_dir = self.direction
        else:
            move_dir = intended
        self.direction = move_dir

        # Calculate new head position
        hx, hy = self.snake[0]
        nx, ny = hx + move_dir[0], hy + move_dir[1]
        new_head = (nx, ny)

        reward = self.step_penalty
        terminated = False
        truncated = False

        # Check wall collision
        if not (0 <= nx < self.grid_w and 0 <= ny < self.grid_h):
            reward += self.death_reward
            terminated = True
        else:
            # Check self collision using cached set
            if new_head in self._snake_set:
                reward += self.death_reward
                terminated = True
            else:
                # Proceed move
                self.snake.insert(0, new_head)
                self._snake_set.add(new_head)
                
                ate = (self.food is not None and new_head == self.food)
                if ate:
                    reward += self.food_reward
                    self._spawn_food()
                else:
                    # Remove tail
                    tail = self.snake.pop()
                    self._snake_set.discard(tail)

                # If food cannot spawn (board full), consider it a win
                if self.food is None and len(self.snake) == self.grid_w * self.grid_h:
                    reward += 1.0
                    terminated = True

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info: Dict[str, Any] = {"length": len(self.snake), "steps": self.steps}
        if self.render_mode == "human":
            self._render_pygame()
        return obs, reward, terminated, truncated, info

    # ---------------------------
    # Rendering (pygame)
    # ---------------------------
    def _ensure_renderer(self) -> None:
        """Initialize pygame renderer if not already initialized."""
        if pygame is None:
            raise RuntimeError("pygame is not installed but render_mode='human' was requested.")
        if self._screen is None:
            pygame.init()
            w = self.grid_w * self.cell_size
            h = self.grid_h * self.cell_size
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Snake RL (DQN)")
            self._clock = pygame.time.Clock()

    def _render_pygame(self) -> None:
        """Render the current game state using pygame."""
        if self._screen is None:
            return
        
        # Handle quit events to avoid freezing the window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        COLOR_BG = (22, 24, 29)
        COLOR_GRID = (36, 39, 46)
        COLOR_SNAKE = (0, 200, 140)
        COLOR_HEAD = (0, 230, 170)
        COLOR_FOOD = (240, 80, 80)

        self._screen.fill(COLOR_BG)
        
        # Draw grid lines
        for x in range(self.grid_w + 1):
            px = x * self.cell_size
            pygame.draw.line(self._screen, COLOR_GRID, (px, 0), (px, self.grid_h * self.cell_size), 1)
        for y in range(self.grid_h + 1):
            py = y * self.cell_size
            pygame.draw.line(self._screen, COLOR_GRID, (0, py), (self.grid_w * self.cell_size, py), 1)

        # Draw food
        if self.food:
            fx, fy = self.food
            rect = pygame.Rect(fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size)
            center = rect.center
            radius = self.cell_size // 2 - 2
            pygame.draw.circle(self._screen, COLOR_FOOD, center, radius)

        # Draw snake
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self._screen, COLOR_HEAD if i == 0 else COLOR_SNAKE, rect)

        pygame.display.flip()
        if self._clock:
            self._clock.tick(self.metadata["render_fps"])

    def close(self) -> None:
        """Clean up resources."""
        if self._screen is not None and pygame is not None:
            pygame.quit()
            self._screen = None
            self._clock = None