import sys
import random
from pathlib import Path

import pygame

# =========================
# CONFIGURATION
# =========================
CELL_SIZE = 24         # Pixel size of one grid cell
GRID_WIDTH = 24        # Number of cells horizontally
GRID_HEIGHT = 20       # Number of cells vertically
FPS = 60               # Render frames per second

# Speed control: lower delay means faster movement
BASE_MOVE_DELAY_MS = 150   # Starting delay between moves in ms
MIN_MOVE_DELAY_MS = 60     # Minimum delay (cap)
SPEEDUP_PER_FOOD_MS = 3    # Speed up per food eaten

# Colors
COLOR_BG = (22, 24, 29)         # Background
COLOR_GRID = (36, 39, 46)       # Grid lines
COLOR_SNAKE = (0, 200, 140)     # Snake body
COLOR_SNAKE_HEAD = (0, 230, 170)
COLOR_FOOD = (240, 80, 80)
COLOR_TEXT = (230, 235, 240)
COLOR_TEXT_DIM = (170, 178, 189)
COLOR_GAME_OVER = (255, 90, 90)

# Font sizes
FONT_MAIN_SIZE = 22
FONT_LARGE_SIZE = 38

# High score file
HIGHSCORE_FILE = Path("highscore.txt")

# =========================
# GAME OBJECTS
# =========================
Vec2 = tuple[int, int]


def add_vec(a: Vec2, b: Vec2) -> Vec2:
    return a[0] + b[0], a[1] + b[1]


def opposite(a: Vec2, b: Vec2) -> bool:
    return a[0] == -b[0] and a[1] == -b[1]


class Snake:
    def __init__(self, start: Vec2, length: int = 3):
        self.body: list[Vec2] = [(start[0] - i, start[1]) for i in range(length)]
        self.direction: Vec2 = (1, 0)  # moving right
        self.grow_pending: int = 0

    @property
    def head(self) -> Vec2:
        return self.body[0]

    def set_direction(self, new_dir: Vec2):
        # Prevent reversing directly into itself
        if len(self.body) > 1 and opposite(self.direction, new_dir):
            return
        self.direction = new_dir

    def step(self) -> Vec2:
        """Advance snake by one cell. Returns new head position."""
        new_head = add_vec(self.head, self.direction)
        self.body.insert(0, new_head)
        if self.grow_pending > 0:
            self.grow_pending -= 1
        else:
            self.body.pop()
        return new_head

    def grow(self, amount: int = 1):
        self.grow_pending += amount

    def collides_with_self(self) -> bool:
        return self.head in self.body[1:]

    def occupies(self, pos: Vec2) -> bool:
        return pos in self.body


class Food:
    def __init__(self, grid_w: int, grid_h: int):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.pos: Vec2 | None = None

    def respawn(self, forbidden: set[Vec2]):
        while True:
            p = (random.randint(0, self.grid_w - 1), random.randint(0, self.grid_h - 1))
            if p not in forbidden:
                self.pos = p
                break


class HighScore:
    def __init__(self, file_path: Path):
        self.file = file_path
        self.value = 0
        self._load()

    def _load(self):
        try:
            if self.file.exists():
                self.value = int(self.file.read_text().strip() or "0")
        except Exception:
            self.value = 0

    def try_set(self, score: int):
        if score > self.value:
            self.value = score
            try:
                self.file.write_text(str(self.value))
            except Exception:
                pass


# =========================
# GAME
# =========================
class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Snake")
        self.screen_w = CELL_SIZE * GRID_WIDTH
        self.screen_h = CELL_SIZE * GRID_HEIGHT
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self.clock = pygame.time.Clock()

        self.font_main = pygame.font.SysFont("consolas,menlo,monaco,dejavusansmono", FONT_MAIN_SIZE, bold=False)
        self.font_large = pygame.font.SysFont("consolas,menlo,monaco,dejavusansmono", FONT_LARGE_SIZE, bold=True)

        self.reset()
        self.highscore = HighScore(HIGHSCORE_FILE)

    def reset(self):
        start = (GRID_WIDTH // 2, GRID_HEIGHT // 2)
        self.snake = Snake(start, length=4)
        self.food = Food(GRID_WIDTH, GRID_HEIGHT)
        self.food.respawn(forbidden=set(self.snake.body))
        self.score = 0
        self.paused = False
        self.game_over = False
        self.move_delay_ms = BASE_MOVE_DELAY_MS
        self.last_move_ms = pygame.time.get_ticks()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.quit_game()

            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.quit_game()

                if event.key == pygame.K_p:
                    if not self.game_over:
                        self.paused = not self.paused

                if self.game_over:
                    if event.key == pygame.K_r:
                        self.reset()
                    continue

                if event.key in (pygame.K_UP, pygame.K_w):
                    self.snake.set_direction((0, -1))
                elif event.key in (pygame.K_DOWN, pygame.K_s):
                    self.snake.set_direction((0, 1))
                elif event.key in (pygame.K_LEFT, pygame.K_a):
                    self.snake.set_direction((-1, 0))
                elif event.key in (pygame.K_RIGHT, pygame.K_d):
                    self.snake.set_direction((1, 0))

    def update(self):
        if self.paused or self.game_over:
            return

        now = pygame.time.get_ticks()
        if now - self.last_move_ms < self.move_delay_ms:
            return
        self.last_move_ms = now

        new_head = self.snake.step()

        # Wall collision
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            self.trigger_game_over()
            return

        # Self collision
        if self.snake.collides_with_self():
            self.trigger_game_over()
            return

        # Food eaten
        if self.food.pos and new_head == self.food.pos:
            self.snake.grow(1)
            self.score += 1
            # Increase speed
            self.move_delay_ms = max(MIN_MOVE_DELAY_MS, self.move_delay_ms - SPEEDUP_PER_FOOD_MS)
            # Respawn food not on snake
            self.food.respawn(forbidden=set(self.snake.body))

    def trigger_game_over(self):
        self.game_over = True
        self.highscore.try_set(self.score)

    def draw_grid(self):
        # Subtle grid lines
        for x in range(GRID_WIDTH + 1):
            px = x * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_GRID, (px, 0), (px, self.screen_h), 1)
        for y in range(GRID_HEIGHT + 1):
            py = y * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_GRID, (0, py), (self.screen_w, py), 1)

    def draw_snake(self):
        # Draw body
        for i, (x, y) in enumerate(self.snake.body):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if i == 0:
                pygame.draw.rect(self.screen, COLOR_SNAKE_HEAD, rect)
                # eyes
                cx, cy = rect.center
                eye_r = max(2, CELL_SIZE // 9)
                dx, dy = self.snake.direction
                offset = CELL_SIZE // 6
                eye1 = (cx - dy * offset - dx * offset, cy + dx * offset - dy * offset)
                eye2 = (cx + dy * offset - dx * offset, cy - dx * offset - dy * offset)
                pygame.draw.circle(self.screen, (20, 20, 20), eye1, eye_r)
                pygame.draw.circle(self.screen, (20, 20, 20), eye2, eye_r)
            else:
                pygame.draw.rect(self.screen, COLOR_SNAKE, rect)

    def draw_food(self):
        if not self.food.pos:
            return
        x, y = self.food.pos
        rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        # Draw as a circle with a little leaf
        center = rect.center
        radius = CELL_SIZE // 2 - 2
        pygame.draw.circle(self.screen, COLOR_FOOD, center, radius)
        # leaf
        leaf_pos = (center[0] - radius // 2, center[1] - radius - 2)
        pygame.draw.circle(self.screen, (60, 190, 90), leaf_pos, max(2, radius // 4))

    def draw_hud(self):
        # Score and High Score
        score_surf = self.font_main.render(f"Score: {self.score}", True, COLOR_TEXT)
        hs_surf = self.font_main.render(f"High: {self.highscore.value}", True, COLOR_TEXT_DIM)
        self.screen.blit(score_surf, (8, 6))
        self.screen.blit(hs_surf, (8 + score_surf.get_width() + 16, 6))

        if self.paused and not self.game_over:
            overlay = self.font_large.render("PAUSED", True, COLOR_TEXT)
            self.screen.blit(overlay, overlay.get_rect(center=(self.screen_w // 2, 30 + overlay.get_height() // 2)))

        if self.game_over:
            title = self.font_large.render("GAME OVER", True, COLOR_GAME_OVER)
            tip = self.font_main.render("Press R to restart • Esc/Q to quit", True, COLOR_TEXT_DIM)
            self.screen.blit(title, title.get_rect(center=(self.screen_w // 2, self.screen_h // 2 - 20)))
            self.screen.blit(tip, tip.get_rect(center=(self.screen_w // 2, self.screen_h // 2 + 20)))

    def draw(self):
        self.screen.fill(COLOR_BG)
        self.draw_grid()
        self.draw_food()
        self.draw_snake()
        self.draw_hud()
        pygame.display.flip()

    def quit_game(self):
        pygame.quit()
        sys.exit(0)

    def run(self):
        while True:
            self.handle_input()
            self.update()
            self.draw()
            self.clock.tick(FPS)


if __name__ == "__main__":
    Game().run()