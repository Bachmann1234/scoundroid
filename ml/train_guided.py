#!/usr/bin/env python3
"""
Guided RL: Train RL with regular episodes but periodically inject winning trajectory seeds.

This approach lets the model learn from both:
1. Random exploration (to generalize)
2. Winning seed episodes (to learn what winning looks like)
"""

import pickle
import numpy as np
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from scoundrel_env import ScoundrelEnv, CardType


class ActorCritic(nn.Module):
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


class ShapedRewardEnv:
    """Wrapper with shaped rewards."""

    def __init__(self, seed=None):
        self.env = ScoundrelEnv(seed=seed)
        self.rooms_cleared = 0

    def reset(self, seed=None):
        obs = self.env.reset(seed)
        self.rooms_cleared = 0
        return obs

    @property
    def done(self):
        return self.env.done

    @property
    def won(self):
        return self.env.won

    @property
    def health(self):
        return self.env.health

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

        reward = 0.0

        # Progress reward
        cards_cleared = old_deck_size - (len(self.env.deck) + len(self.env.room))
        if cards_cleared > 0:
            progress_fraction = 1 - len(self.env.deck) / 44.0
            reward += 0.05 * cards_cleared * (1 + progress_fraction)
            self.rooms_cleared += 1

        # Health management
        health_change = self.env.health - old_health
        if health_change < 0:
            health_fraction = old_health / 20.0
            penalty = 0.03 * abs(health_change) * (2 - health_fraction)
            reward -= penalty
        elif health_change > 0:
            reward += 0.02 * health_change

        # Terminal rewards - MUCH BIGGER bonus for winning
        if done:
            if self.env.won:
                reward += 50.0 + self.env.health * 2.0  # Massive bonus!
            else:
                score = self.calculate_score()
                progress = self.rooms_cleared / 12.0
                reward += score / 50.0
                reward += progress * 0.5

        return obs, reward, done, info


def train_episode(env: ShapedRewardEnv, model: ActorCritic,
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
    """Evaluate the policy deterministically."""
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
    import argparse
    parser = argparse.ArgumentParser(description='Guided RL Training')
    parser.add_argument('--trajectories', type=str, default='winning_trajectories.pkl')
    parser.add_argument('--episodes', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-coef', type=float, default=0.03)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--winning-seed-ratio', type=float, default=0.1,
                        help='Fraction of episodes to use winning seeds')
    parser.add_argument('--output', type=str, default='rl_guided.pt')
    args = parser.parse_args()

    # Load winning trajectories for their seeds
    print("Loading winning trajectories...")
    with open(args.trajectories, 'rb') as f:
        trajectories = pickle.load(f)
    winning_seeds = [t['seed'] for t in trajectories]
    print(f"Loaded {len(winning_seeds)} winning seeds: {winning_seeds}")

    print(f"\nTraining with GUIDED exploration")
    print(f"Episodes: {args.episodes}")
    print(f"Winning seed ratio: {args.winning_seed_ratio}")
    print(f"Entropy coef: {args.entropy_coef}")
    print()

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recent_scores = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)
    best_win_rate = 0
    best_avg_score = float('-inf')
    best_model_state = None

    wins_on_winning_seeds = 0
    attempts_on_winning_seeds = 0

    for episode in range(args.episodes):
        # Decide whether to use a winning seed
        use_winning_seed = np.random.random() < args.winning_seed_ratio

        if use_winning_seed and winning_seeds:
            seed = int(np.random.choice(winning_seeds))
            attempts_on_winning_seeds += 1
        else:
            seed = None

        env = ShapedRewardEnv(seed=seed)
        score, won = train_episode(env, model, optimizer, args.gamma, args.entropy_coef)

        if use_winning_seed and won:
            wins_on_winning_seeds += 1

        recent_scores.append(score)
        recent_wins.append(1 if won else 0)

        if (episode + 1) % args.eval_interval == 0:
            avg_score = np.mean(recent_scores)
            win_rate = np.mean(recent_wins)
            eval_win_rate, eval_avg_score = evaluate(model, num_episodes=1000)

            print(f"Episode {episode + 1}:")
            print(f"  Training: win_rate={win_rate:.4f}, avg_score={avg_score:.1f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}")
            if attempts_on_winning_seeds > 0:
                seed_win_rate = wins_on_winning_seeds / attempts_on_winning_seeds
                print(f"  Winning seeds: {wins_on_winning_seeds}/{attempts_on_winning_seeds} ({seed_win_rate:.2%})")

            # Save best by win rate (primary) or score (secondary)
            if eval_win_rate > best_win_rate or (eval_win_rate == best_win_rate and eval_avg_score > best_avg_score):
                best_win_rate = eval_win_rate
                best_avg_score = eval_avg_score
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  New best! win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    print("\n=== Final Evaluation ===")
    final_win_rate, final_avg_score = evaluate(model, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg score: {final_avg_score:.1f}")

    # Test on winning seeds specifically
    print("\nTesting on winning seeds:")
    for seed in winning_seeds:
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
        result = "WON" if env.won else f"LOST (health={env.health})"
        print(f"  Seed {seed}: {result}")

    torch.save({
        'state_dict': model.state_dict(),
        'win_rate': final_win_rate,
        'avg_score': final_avg_score,
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
