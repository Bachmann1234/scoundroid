#!/usr/bin/env python3
"""
RL training with DENSE step-by-step rewards.

Instead of just score at the end, give feedback at each step:
- Progress reward for clearing cards from deck
- Health change penalty/bonus
- Monster defeat bonus
- Final score bonus
"""

import argparse
import numpy as np
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from scoundrel_env import ScoundrelEnv, CardType


class ActorCritic(nn.Module):
    def __init__(self, input_size=26, hidden_size=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size, 5)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state: np.ndarray, valid_actions: List[int]):
        state_t = torch.from_numpy(state).unsqueeze(0)
        logits, value = self.forward(state_t)
        logits = logits.squeeze(0)
        value = value.squeeze()

        mask = torch.full((5,), float('-inf'))
        for a in valid_actions:
            mask[a] = 0
        masked_logits = logits + mask

        probs = torch.softmax(masked_logits, dim=0)
        dist = Categorical(probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value, dist.entropy()


class DenseRewardEnv:
    """Wrapper that provides dense step-by-step rewards."""

    def __init__(self, seed=None):
        self.env = ScoundrelEnv(seed=seed)
        self.prev_health = 20
        self.prev_deck_size = 44

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        self.prev_health = 20
        self.prev_deck_size = len(self.env.deck) + len(self.env.room)
        return obs

    @property
    def done(self):
        return self.env.done

    @property
    def won(self):
        return self.env.won

    def get_valid_actions(self):
        return self.env.get_valid_actions()

    def calculate_score(self):
        if self.env.won:
            return float(self.env.health)
        else:
            remaining = sum(c.value for c in self.env.deck if c.card_type == CardType.MONSTER)
            remaining += sum(c.value for c in self.env.room if c.card_type == CardType.MONSTER)
            return float(self.env.health - remaining)

    def step(self, action):
        old_health = self.env.health
        old_deck_size = len(self.env.deck) + len(self.env.room)

        obs, _, done, info = self.env.step(action)

        # Dense rewards
        reward = 0.0

        # 1. Progress through deck (small positive)
        cards_cleared = old_deck_size - (len(self.env.deck) + len(self.env.room))
        reward += 0.02 * cards_cleared

        # 2. Health change (scale by importance)
        health_change = self.env.health - old_health
        if health_change < 0:
            # Losing health is bad, but scale by current health
            # Losing health when low is worse
            health_fraction = old_health / 20.0
            reward += 0.05 * health_change * (2 - health_fraction)
        else:
            # Gaining health (potion) is good
            reward += 0.02 * health_change

        # 3. Terminal bonus based on score
        if done:
            score = self.calculate_score()
            # Map score to reward: -150 to +20 -> -1.5 to +2
            reward += score / 100.0

            # Extra bonus for winning
            if self.env.won:
                reward += 1.0

        self.prev_health = self.env.health
        self.prev_deck_size = len(self.env.deck) + len(self.env.room)

        return obs, reward, done, info


def train_episode(env: DenseRewardEnv, model: ActorCritic,
                  optimizer: optim.Optimizer, gamma: float = 0.99,
                  entropy_coef: float = 0.02) -> Tuple[float, bool]:
    """Train on a single episode."""
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

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns)

    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    values = torch.stack(values)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    advantages = returns - values.detach()

    policy_loss = -(log_probs * advantages).mean()
    value_loss = 0.5 * ((returns - values) ** 2).mean()
    entropy_loss = -entropy_coef * entropies.mean()

    total_loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return env.calculate_score(), env.won


def evaluate(model: ActorCritic, num_episodes: int = 1000):
    """Evaluate the policy."""
    wins = 0
    total_score = 0

    for seed in range(num_episodes):
        env = ScoundrelEnv(seed=seed)
        state = env.reset()

        while not env.done:
            valid_actions = env.get_valid_actions()
            state_t = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                logits, _ = model(state_t)
                logits = logits.squeeze(0)

            mask = torch.full((5,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0
            masked_logits = logits + mask
            action = masked_logits.argmax().item()
            state, _, done, _ = env.step(action)

        if env.won:
            score = float(env.health)
            wins += 1
        else:
            remaining = sum(c.value for c in env.deck if c.card_type == CardType.MONSTER)
            remaining += sum(c.value for c in env.room if c.card_type == CardType.MONSTER)
            score = float(env.health - remaining)
        total_score += score

    return wins / num_episodes, total_score / num_episodes


def main():
    parser = argparse.ArgumentParser(description='Train with Dense Rewards')
    parser.add_argument('--episodes', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.02)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--output', type=str, default='ml/rl_dense.pt')
    args = parser.parse_args()

    print("Training with DENSE step-by-step rewards")
    print(f"Episodes: {args.episodes}")
    print()

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recent_scores = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)
    best_avg_score = float('-inf')
    best_model_state = None

    for episode in range(args.episodes):
        env = DenseRewardEnv()
        score, won = train_episode(env, model, optimizer, args.gamma, args.entropy_coef)

        recent_scores.append(score)
        recent_wins.append(1 if won else 0)

        if (episode + 1) % args.eval_interval == 0:
            avg_score = np.mean(recent_scores)
            win_rate = np.mean(recent_wins)
            eval_win_rate, eval_avg_score = evaluate(model, num_episodes=1000)

            print(f"Episode {episode + 1}:")
            print(f"  Training: win_rate={win_rate:.4f}, avg_score={avg_score:.1f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}")

            if eval_avg_score > best_avg_score:
                best_avg_score = eval_avg_score
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  New best! avg_score={eval_avg_score:.1f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    print("\n=== Final Evaluation ===")
    final_win_rate, final_avg_score = evaluate(model, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg score: {final_avg_score:.1f}")

    torch.save({
        'state_dict': model.state_dict(),
        'win_rate': final_win_rate,
        'avg_score': final_avg_score,
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
