from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
from tqdm import trange

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    if decay_steps <= 0:
        return end
    t = min(step / decay_steps, 1.0)
    return start + (end - start) * t


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid_w", type=int, default=24)
    parser.add_argument("--grid_h", type=int, default=20)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--buffer_size", type=int, default=100_000)
    parser.add_argument("--train_start", type=int, default=10_000)
    parser.add_argument("--target_update", type=int, default=1000)
    parser.add_argument("--eps_start", type=float, default=1.0)
    parser.add_argument("--eps_end", type=float, default=0.05)
    parser.add_argument("--eps_decay_steps", type=int, default=200_000)
    parser.add_argument("--step_penalty", type=float, default=-0.01)
    parser.add_argument("--food_reward", type=float, default=1.0)
    parser.add_argument("--death_reward", type=float, default=-1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--eval_interval", type=int, default=10_000)  # steps
    parser.add_argument("--eval_episodes", type=int, default=5)
    args = parser.parse_args()

    env = SnakeEnv(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        step_penalty=args.step_penalty,
        food_reward=args.food_reward,
        death_reward=args.death_reward,
        render_mode="none",
    )
    env.reset(seed=args.seed)

    cfg = DQNConfig(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        target_update=args.target_update,
        buffer_size=args.buffer_size,
        train_start=args.train_start,
        device=args.device,
    )
    agent = DQNAgent(cfg)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    latest_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_latest.pth")
    best_ckpt = os.path.join(args.checkpoint_dir, "dqn_snake_best.pth")

    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    ep_len = 0
    returns = deque(maxlen=100)
    lengths = deque(maxlen=100)
    best_avg_return = -1e9

    t0 = time.time()
    pbar = trange(args.total_steps, desc="Training", unit="step")
    for step in pbar:
        epsilon = linear_epsilon(step, args.eps_start, args.eps_end, args.eps_decay_steps)
        action = agent.act(obs, epsilon=epsilon)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.push(obs, action, reward, next_obs, done)

        loss = agent.train_step()

        obs = next_obs
        ep_return += reward
        ep_len += 1

        if done:
            returns.append(ep_return)
            lengths.append(ep_len)
            obs, _ = env.reset()
            ep_return = 0.0
            ep_len = 0

        # UI update
        avg_ret = np.mean(returns) if returns else 0.0
        avg_len = int(np.mean(lengths)) if lengths else 0
        pbar.set_postfix(eps=f"{epsilon:.3f}", avg_return=f"{avg_ret:.2f}", avg_len=avg_len, loss=f"{(loss or 0):.4f}")

        # Periodic evaluation (headless)
        if (step + 1) % args.eval_interval == 0:
            eval_ret = evaluate(agent, env, episodes=args.eval_episodes)
            if eval_ret > best_avg_return:
                best_avg_return = eval_ret
                agent.save(best_ckpt)
            agent.save(latest_ckpt)

    agent.save(latest_ckpt)
    env.close()
    dt = time.time() - t0
    print(f"Training complete in {dt/60:.1f} min. Best eval avg return: {best_avg_return:.2f}")


def evaluate(agent: DQNAgent, env: SnakeEnv, episodes: int = 5) -> float:
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += reward
            done = terminated or truncated
        total += ret
    return total / episodes


if __name__ == "__main__":
    main()