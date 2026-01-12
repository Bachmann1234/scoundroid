#!/usr/bin/env python3
"""
RL training using game SCORE as the reward signal.

Score provides dense feedback:
- Win: remaining health (1-20)
- Lose: health - sum of remaining monster values (very negative)

This lets the agent learn from "how well" it did, not just win/lose.
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
    """Combined actor-critic network."""

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


def calculate_score(env: ScoundrelEnv) -> float:
    """Calculate game score (same as Kotlin version)."""
    if env.won:
        return float(env.health)
    else:
        # Negative score based on remaining monsters
        remaining_monster_value = sum(
            c.value for c in env.deck if c.card_type == CardType.MONSTER
        )
        remaining_monster_value += sum(
            c.value for c in env.room if c.card_type == CardType.MONSTER
        )
        return float(env.health - remaining_monster_value)


def train_episode(env: ScoundrelEnv, model: ActorCritic,
                  optimizer: optim.Optimizer, gamma: float = 0.99,
                  entropy_coef: float = 0.01) -> Tuple[float, bool, float]:
    """Train on a single episode using score as reward."""
    log_probs = []
    values = []
    entropies = []

    state = env.reset()

    while not env.done:
        valid_actions = env.get_valid_actions()
        action, log_prob, value, entropy = model.get_action(state, valid_actions)

        next_state, _, done, _ = env.step(action)

        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)

        state = next_state

    # Use final SCORE as the reward (normalized)
    score = calculate_score(env)
    # Normalize: typical scores range from -150 to +20
    # Map to roughly [-1, 1] range
    normalized_score = (score + 50) / 70  # Shifts and scales

    # All steps get the same final reward (episodic)
    rewards = [normalized_score] * len(log_probs)

    # Compute returns (with discount, reward is at end)
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

    return score, env.won, normalized_score


def evaluate(model: ActorCritic, num_episodes: int = 1000) -> Tuple[float, float, float]:
    """Evaluate the policy. Returns (win_rate, avg_score, best_score)."""
    wins = 0
    total_score = 0
    best_score = float('-inf')

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

        score = calculate_score(env)
        total_score += score
        best_score = max(best_score, score)

        if env.won:
            wins += 1

    return wins / num_episodes, total_score / num_episodes, best_score


def main():
    parser = argparse.ArgumentParser(description='Train Scoundrel with Score-based RL')
    parser.add_argument('--episodes', type=int, default=200000, help='Training episodes')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--eval-interval', type=int, default=10000, help='Evaluation interval')
    parser.add_argument('--output', type=str, default='ml/rl_score.pt', help='Output path')
    args = parser.parse_args()

    print("Training Scoundrel with SCORE-based rewards")
    print(f"Episodes: {args.episodes}")
    print(f"Learning rate: {args.lr}")
    print()

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    recent_scores = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)

    best_avg_score = float('-inf')
    best_model_state = None

    for episode in range(args.episodes):
        env = ScoundrelEnv()
        score, won, norm_score = train_episode(env, model, optimizer, args.gamma, args.entropy_coef)

        recent_scores.append(score)
        recent_wins.append(1 if won else 0)

        if (episode + 1) % args.eval_interval == 0:
            avg_score = np.mean(recent_scores)
            win_rate = np.mean(recent_wins)

            eval_win_rate, eval_avg_score, eval_best = evaluate(model, num_episodes=1000)

            print(f"Episode {episode + 1}:")
            print(f"  Training: win_rate={win_rate:.4f}, avg_score={avg_score:.1f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}, best={eval_best:.0f}")

            if eval_avg_score > best_avg_score:
                best_avg_score = eval_avg_score
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  New best avg score: {eval_avg_score:.1f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    print("\n=== Final Evaluation ===")
    final_win_rate, final_avg_score, final_best = evaluate(model, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg score: {final_avg_score:.1f}")
    print(f"Best score: {final_best:.0f}")

    torch.save({
        'state_dict': model.state_dict(),
        'win_rate': final_win_rate,
        'avg_score': final_avg_score,
    }, args.output)
    print(f"\nSaved model to {args.output}")


if __name__ == '__main__':
    main()
