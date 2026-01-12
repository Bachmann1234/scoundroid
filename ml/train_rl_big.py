#!/usr/bin/env python3
"""RL training with bigger network and more aggressive exploration."""

import argparse
import numpy as np
from collections import deque
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from scoundrel_env import ScoundrelEnv, CardType


class BigActorCritic(nn.Module):
    """Bigger network with more capacity."""

    def __init__(self, input_size=26, hidden_size=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size // 2, 5)
        self.critic = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)

    def get_action(self, state: np.ndarray, valid_actions: List[int], temperature=1.0):
        state_t = torch.from_numpy(state).unsqueeze(0)
        logits, value = self.forward(state_t)
        logits = logits.squeeze(0) / temperature
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
    def __init__(self, seed=None):
        self.env = ScoundrelEnv(seed=seed)

    def reset(self, seed=None):
        return self.env.reset(seed)

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
        remaining = sum(c.value for c in self.env.deck if c.card_type == CardType.MONSTER)
        remaining += sum(c.value for c in self.env.room if c.card_type == CardType.MONSTER)
        return float(self.env.health - remaining)

    def step(self, action):
        old_health = self.env.health
        old_deck_size = len(self.env.deck) + len(self.env.room)

        obs, _, done, info = self.env.step(action)

        reward = 0.0

        # Progress reward
        cards_cleared = old_deck_size - (len(self.env.deck) + len(self.env.room))
        reward += 0.03 * cards_cleared

        # Health change
        health_change = self.env.health - old_health
        if health_change < 0:
            health_fraction = old_health / 20.0
            reward += 0.05 * health_change * (2 - health_fraction)
        else:
            reward += 0.03 * health_change

        # Terminal
        if done:
            score = self.calculate_score()
            reward += score / 50.0  # Bigger scale for final reward
            if self.env.won:
                reward += 5.0  # Big win bonus

        return obs, reward, done, info


def train_episode(env, model, optimizer, gamma=0.99, entropy_coef=0.05, temp=1.0):
    log_probs, values, rewards, entropies = [], [], [], []
    state = env.reset()

    while not env.done:
        valid_actions = env.get_valid_actions()
        action, log_prob, value, entropy = model.get_action(state, valid_actions, temp)
        next_state, reward, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        entropies.append(entropy)
        state = next_state

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


def evaluate(model, num_episodes=1000):
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
            action = (logits + mask).argmax().item()
            state, _, _, _ = env.step(action)

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--eval-interval', type=int, default=50000)
    parser.add_argument('--output', type=str, default='ml/rl_big.pt')
    args = parser.parse_args()

    print("Training BIG network with aggressive exploration")
    print(f"Episodes: {args.episodes}")

    model = BigActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recent_scores = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)
    best_avg_score = float('-inf')
    best_model_state = None

    # Temperature annealing: start high (more exploration), decrease over time
    initial_temp = 2.0
    final_temp = 0.5

    for episode in range(args.episodes):
        # Anneal temperature
        progress = episode / args.episodes
        temp = initial_temp * (1 - progress) + final_temp * progress

        env = DenseRewardEnv()
        score, won = train_episode(env, model, optimizer, temp=temp)

        recent_scores.append(score)
        recent_wins.append(1 if won else 0)

        if (episode + 1) % args.eval_interval == 0:
            avg_score = np.mean(recent_scores)
            win_rate = np.mean(recent_wins)
            eval_win_rate, eval_avg_score = evaluate(model, num_episodes=2000)

            print(f"Episode {episode + 1} (temp={temp:.2f}):")
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


if __name__ == '__main__':
    main()
