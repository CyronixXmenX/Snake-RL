from __future__ import annotations

import argparse
import time

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--grid_w", type=int, default=24)
    parser.add_argument("--grid_h", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--step_delay", type=float, default=0.05, help="Seconds between steps when rendering")
    args = parser.parse_args()

    env = SnakeEnv(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        render_mode="human" if args.render else "none",
    )
    cfg = DQNConfig(grid_w=args.grid_w, grid_h=args.grid_h)
    agent = DQNAgent(cfg)
    agent.load(args.model, strict=False)

    scores = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            done = terminated or truncated
            if args.render:
                time.sleep(args.step_delay)
        scores.append(ret)
        print(f"Episode {ep+1}: return={ret:.2f}, length={info.get('length', 0)}")

    avg = sum(scores) / len(scores) if scores else 0.0
    print(f"Average return over {len(scores)} episodes: {avg:.2f}")
    env.close()


if __name__ == "__main__":
    main()