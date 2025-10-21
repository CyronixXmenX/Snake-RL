"""
Evaluation script for trained DQN agent.

Loads a trained agent and evaluates its performance on the Snake environment,
with optional visualization.
"""

from __future__ import annotations

import argparse
import time
from typing import List

from snake_env import SnakeEnv
from dqn_agent import DQNAgent, DQNConfig


def main() -> None:
    """Main evaluation loop."""
    parser = argparse.ArgumentParser(description="Evaluate trained DQN agent on Snake")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--grid_w", type=int, default=24, help="Grid width")
    parser.add_argument("--grid_h", type=int, default=20, help="Grid height")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--render", type=lambda x: x.lower() == 'true', default=False,
                        help="Enable pygame rendering (True/False)")
    parser.add_argument("--step_delay", type=float, default=0.05, 
                        help="Seconds between steps when rendering")
    args = parser.parse_args()

    # Create environment
    env = SnakeEnv(
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        render_mode="human" if args.render else "none",
    )
    
    # Create and load agent
    cfg = DQNConfig(grid_w=args.grid_w, grid_h=args.grid_h)
    agent = DQNAgent(cfg)
    agent.load(args.model, strict=False)

    # Run evaluation episodes
    scores: List[float] = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ret = 0.0
        steps = 0
        
        while not done:
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += reward
            done = terminated or truncated
            steps += 1
            
            if args.render:
                time.sleep(args.step_delay)
        
        length = info.get('length', 0)
        scores.append(ret)
        print(f"Episode {ep+1}/{args.episodes}: return={ret:.2f}, length={length}, steps={steps}")

    # Print statistics
    if scores:
        avg = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({len(scores)} episodes):")
        print(f"  Average return: {avg:.2f}")
        print(f"  Min return: {min_score:.2f}")
        print(f"  Max return: {max_score:.2f}")
        print(f"{'='*50}")
    
    env.close()


if __name__ == "__main__":
    main()