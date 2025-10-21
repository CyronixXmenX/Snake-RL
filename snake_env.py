from __future__ import annotations

import math
from typing import Optional, Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Optional/lazy pygame import for render()
try:
    import pygame
except Exception:
    pygame = None


Direction = Tuple[int, int]
Pos = Tuple[int, int]


def _is_opposite(a: Direction, b: Direction) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]


class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "none"], "render_fps": 12}

    def __init__(
        self,
        grid_w: int = 24,
        grid_h: int = 20,
        step_penalty: float = -0.01,
        food_reward: float = 1.0,
        death_reward: float = -1.0,
        max_steps_multiplier: float = 4.0,
        render_mode: str = "none",
        cell_size: int = 24,  # for pygame render
    ):
        super().__init__()
        assert render_mode in ("human", "none")
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

        # Render members
        self._screen = None
        self._clock = None

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)

    def _spawn_food(self):
        # If full, no place to spawn -> win condition
        if len(self.snake) >= self.grid_w * self.grid_h:
            self.food = None
            return
        snake_set = set(self.snake)
        while True:
            p = (int(self._rng.randint(0, self.grid_w)), int(self._rng.randint(0, self.grid_h)))
            if p not in snake_set:
                self.food = p
                return

    def _reset_snake(self):
        cx, cy = self.grid_w // 2, self.grid_h // 2
        length = 4
        # body from left to right; head at index 0 moving right
        self.snake = [(cx - i, cy) for i in range(length)]
        self.direction = (1, 0)

    def _get_obs(self) -> np.ndarray:
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

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self.seed(seed)
        self.steps = 0
        self._reset_snake()
        self._spawn_food()
        obs = self._get_obs()
        info = {}
        if self.render_mode == "human":
            self._ensure_renderer()
            self._render_pygame()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action)
        self.steps += 1

        action_dirs: Tuple[Direction, ...] = ((0, -1), (0, 1), (-1, 0), (1, 0))
        intended = action_dirs[action]
        # Ignore reversing directly into itself, like the Pygame version
        if len(self.snake) > 1 and _is_opposite(self.direction, intended):
            move_dir = self.direction
        else:
            move_dir = intended
        self.direction = move_dir

        # Move
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
            # Check self collision
            if new_head in self.snake:
                reward += self.death_reward
                terminated = True
            else:
                # Proceed move
                self.snake.insert(0, new_head)
                ate = (self.food is not None and new_head == self.food)
                if ate:
                    reward += self.food_reward
                    self._spawn_food()
                else:
                    # remove tail
                    self.snake.pop()

                # If food cannot spawn (board full), consider it a win and terminate with a small bonus
                if self.food is None and len(self.snake) == self.grid_w * self.grid_h:
                    reward += 1.0
                    terminated = True

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = {"length": len(self.snake), "steps": self.steps}
        if self.render_mode == "human":
            self._render_pygame()
        return obs, reward, terminated, truncated, info

    # ---------------------------
    # Rendering (pygame)
    # ---------------------------
    def _ensure_renderer(self):
        if pygame is None:
            raise RuntimeError("pygame is not installed but render_mode='human' was requested.")
        if self._screen is None:
            pygame.init()
            w = self.grid_w * self.cell_size
            h = self.grid_h * self.cell_size
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Snake RL (DQN)")
            self._clock = pygame.time.Clock()

    def _render_pygame(self):
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
        # grid
        for x in range(self.grid_w + 1):
            px = x * self.cell_size
            pygame.draw.line(self._screen, COLOR_GRID, (px, 0), (px, self.grid_h * self.cell_size), 1)
        for y in range(self.grid_h + 1):
            py = y * self.cell_size
            pygame.draw.line(self._screen, COLOR_GRID, (0, py), (self.grid_w * self.cell_size, py), 1)

        # food
        if self.food:
            fx, fy = self.food
            rect = pygame.Rect(fx * self.cell_size, fy * self.cell_size, self.cell_size, self.cell_size)
            center = rect.center
            radius = self.cell_size // 2 - 2
            pygame.draw.circle(self._screen, COLOR_FOOD, center, radius)

        # snake
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self._screen, COLOR_HEAD if i == 0 else COLOR_SNAKE, rect)

        pygame.display.flip()
        if self._clock:
            self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None and pygame is not None:
            pygame.quit()
            self._screen = None
            self._clock = None