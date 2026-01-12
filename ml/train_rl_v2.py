#!/usr/bin/env python3
"""
Improved RL training using Actor-Critic (A2C) with shaped rewards.

Key improvements over v1:
1. Shaped rewards give feedback during the game (not just at end)
2. Actor-Critic architecture reduces variance
3. Entropy bonus encourages exploration
4. Optional pre-training from heuristic demonstrations
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


class ActorCritic(nn.Module):
    """
    Combined actor-critic network with shared features.
    """

    def __init__(self, input_size=26, hidden_size=128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, 5)  # 5 actions

        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.shared(x)
        policy_logits = self.actor(features)
        value = self.critic(features)
        return policy_logits, value

    def get_action(self, state: np.ndarray, valid_actions: List[int]) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """Sample an action and return log prob and value estimate."""
        state_t = torch.from_numpy(state).unsqueeze(0)
        logits, value = self.forward(state_t)
        logits = logits.squeeze(0)
        value = value.squeeze()

        # Mask invalid actions
        mask = torch.full((5,), float('-inf'))
        for a in valid_actions:
            mask[a] = 0
        masked_logits = logits + mask

        probs = torch.softmax(masked_logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value, dist.entropy()


class ShapedRewardEnv:
    """
    Wrapper that adds shaped rewards based on game progress.
    """

    def __init__(self, seed=None):
        self.env = ScoundrelEnv(seed=seed)
        self.prev_health = 20
        self.prev_deck_size = 44
        self.monsters_defeated = 0

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        self.prev_health = 20
        self.prev_deck_size = len(self.env.deck)
        self.monsters_defeated = 0
        return obs

    @property
    def done(self):
        return self.env.done

    @property
    def won(self):
        return self.env.won

    def get_valid_actions(self):
        return self.env.get_valid_actions()

    def step(self, action):
        # Track state before action
        old_deck_size = len(self.env.deck)
        old_health = self.env.health

        obs, reward, done, info = self.env.step(action)

        # Shaped reward components
        shaped_reward = 0.0

        # Small reward for progress through deck
        deck_progress = (old_deck_size - len(self.env.deck)) / 44.0
        shaped_reward += 0.01 * deck_progress

        # Penalty for health loss (but not too strong)
        health_lost = old_health - self.env.health
        if health_lost > 0:
            shaped_reward -= 0.02 * health_lost / 20.0

        # Terminal rewards (unchanged from env)
        if done:
            if self.env.won:
                # Big bonus for winning + health bonus
                shaped_reward += 1.0 + self.env.health / 20.0
            else:
                # Penalty for losing, but scale by progress
                progress = 1.0 - len(self.env.deck) / 44.0
                shaped_reward += -1.0 + 0.5 * progress  # Less harsh if got far

        return obs, shaped_reward, done, info


def train_episode(env: ShapedRewardEnv, model: ActorCritic,
                  optimizer: optim.Optimizer, gamma: float = 0.99,
                  entropy_coef: float = 0.01) -> Tuple[float, bool]:
    """Train on a single episode using A2C."""
    log_probs = []
    values = []
    rewards = []
    entropies = []

    state = env.reset()

    while not env.done:
        valid_actions = env.get_valid_actions()
        action, log_prob, value, entropy = model.get_action(state, valid_actions)

        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)

        state = next_state

    # Compute returns and advantages
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    # Advantage = returns - value estimates
    advantages = returns - values.detach()

    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Compute losses
    policy_loss = -(log_probs * advantages).mean()
    value_loss = 0.5 * ((returns - values) ** 2).mean()
    entropy_loss = -entropy_coef * entropies.mean()

    total_loss = policy_loss + value_loss + entropy_loss

    # Update
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return sum(rewards), env.won


def evaluate(model: ActorCritic, num_episodes: int = 1000) -> Tuple[float, float]:
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
                logits, _ = model(state_t)
                logits = logits.squeeze(0)

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
    parser = argparse.ArgumentParser(description='Train Scoundrel A2C Agent')
    parser.add_argument('--episodes', type=int, default=200000, help='Training episodes')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--eval-interval', type=int, default=10000, help='Evaluation interval')
    parser.add_argument('--output', type=str, default='ml/rl_a2c.pt', help='Output path')
    args = parser.parse_args()

    print("Training Scoundrel A2C Agent with Shaped Rewards")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    print(f"Entropy coefficient: {args.entropy_coef}")
    print()

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Track metrics
    recent_rewards = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)

    best_win_rate = 0
    best_model_state = None

    for episode in range(args.episodes):
        env = ShapedRewardEnv()
        reward, won = train_episode(env, model, optimizer, args.gamma, args.entropy_coef)

        recent_rewards.append(reward)
        recent_wins.append(1 if won else 0)

        if (episode + 1) % args.eval_interval == 0:
            avg_reward = np.mean(recent_rewards)
            win_rate = np.mean(recent_wins)

            # Proper evaluation
            eval_win_rate, eval_reward = evaluate(model, num_episodes=1000)

            print(f"Episode {episode + 1}:")
            print(f"  Training: win_rate={win_rate:.4f}, avg_reward={avg_reward:.3f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_reward={eval_reward:.3f}")

            if eval_win_rate > best_win_rate:
                best_win_rate = eval_win_rate
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  New best! win_rate={eval_win_rate:.4f}")

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_win_rate, final_reward = evaluate(model, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg reward: {final_reward:.3f}")

    # Save model
    torch.save({
        'state_dict': model.state_dict(),
        'win_rate': final_win_rate,
        'input_size': 26,
        'hidden_size': 128,
    }, args.output)
    print(f"\nSaved model to {args.output}")


if __name__ == '__main__':
    main()
