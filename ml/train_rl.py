#!/usr/bin/env python3
"""
Reinforcement learning training for Scoundrel using Policy Gradient (REINFORCE).

The agent learns by playing games and receiving rewards:
- Win: +1 (plus bonus for remaining health)
- Lose: -1
"""

import argparse
import numpy as np
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from scoundrel_env import ScoundrelEnv


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs action probabilities.
    """

    def __init__(self, input_size=26, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 5),  # 5 actions: skip, leave 0-3
        )

    def forward(self, x):
        return self.network(x)

    def get_action(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, torch.Tensor]:
        """Sample an action from the policy."""
        state_t = torch.from_numpy(state).unsqueeze(0)
        logits = self.forward(state_t).squeeze(0)

        # Mask invalid actions
        mask = torch.full((5,), float('-inf'))
        for a in valid_actions:
            mask[a] = 0
        masked_logits = logits + mask

        probs = torch.softmax(masked_logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action)


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """Compute discounted returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return returns


def train_episode(env: ScoundrelEnv, policy: PolicyNetwork,
                  optimizer: optim.Optimizer, gamma: float = 0.99) -> Tuple[float, bool]:
    """Train on a single episode."""
    log_probs = []
    rewards = []

    state = env.reset()

    while not env.done:
        valid_actions = env.get_valid_actions()
        action, log_prob = policy.get_action(state, valid_actions)

        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        rewards.append(reward)

        state = next_state

    # Compute returns
    returns = compute_returns(rewards, gamma)
    returns = torch.tensor(returns)

    # Normalize returns
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute policy loss
    policy_loss = []
    for log_prob, G in zip(log_probs, returns):
        policy_loss.append(-log_prob * G)

    if policy_loss:
        optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        optimizer.step()

    return sum(rewards), env.won


def evaluate(policy: PolicyNetwork, num_episodes: int = 1000) -> Tuple[float, float]:
    """Evaluate the policy."""
    wins = 0
    total_reward = 0

    for seed in range(num_episodes):
        env = ScoundrelEnv(seed=seed)
        state = env.reset()

        episode_reward = 0
        while not env.done:
            valid_actions = env.get_valid_actions()

            # Greedy action selection
            state_t = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                logits = policy(state_t).squeeze(0)

            mask = torch.full((5,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0
            masked_logits = logits + mask

            action = masked_logits.argmax().item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward

        if env.won:
            wins += 1
        total_reward += episode_reward

    return wins / num_episodes, total_reward / num_episodes


def main():
    parser = argparse.ArgumentParser(description='Train Scoundrel RL Agent')
    parser.add_argument('--episodes', type=int, default=100000, help='Training episodes')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--eval-interval', type=int, default=5000, help='Evaluation interval')
    parser.add_argument('--output', type=str, default='ml/rl_policy.pt', help='Output path')
    args = parser.parse_args()

    print("Training Scoundrel RL Agent")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    print()

    policy = PolicyNetwork()
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    # Track metrics
    recent_rewards = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)

    best_win_rate = 0
    best_model_state = None

    for episode in range(args.episodes):
        env = ScoundrelEnv()
        reward, won = train_episode(env, policy, optimizer, args.gamma)

        recent_rewards.append(reward)
        recent_wins.append(1 if won else 0)

        if (episode + 1) % args.eval_interval == 0:
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)

            # Proper evaluation
            eval_win_rate, eval_reward = evaluate(policy, num_episodes=1000)

            print(f"Episode {episode + 1}:")
            print(f"  Training: win_rate={win_rate:.3f}, avg_reward={avg_reward:.3f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_reward={eval_reward:.3f}")

            if eval_win_rate > best_win_rate:
                best_win_rate = eval_win_rate
                best_model_state = policy.state_dict().copy()
                print(f"  New best! win_rate={eval_win_rate:.4f}")

    # Load best model
    if best_model_state:
        policy.load_state_dict(best_model_state)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_win_rate, final_reward = evaluate(policy, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg reward: {final_reward:.3f}")

    # Save model
    torch.save({
        'state_dict': policy.state_dict(),
        'win_rate': final_win_rate,
    }, args.output)
    print(f"\nSaved model to {args.output}")


if __name__ == '__main__':
    main()
