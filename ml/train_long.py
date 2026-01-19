#!/usr/bin/env python3
"""
Long training run with PPO and entropy annealing for overnight training.
"""

import pickle
import numpy as np
from collections import deque
from typing import List, Tuple
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from scoundrel_env import ScoundrelEnv, CardType


class ActorCritic(nn.Module):
    def __init__(self, input_size=26, hidden_size=512):
        super().__init__()
        # Larger network for longer training
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_size // 4, 5)
        self.critic = nn.Linear(hidden_size // 4, 1)

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

        # Terminal rewards
        if done:
            if self.env.won:
                reward += 100.0 + self.env.health * 5.0  # HUGE bonus
            else:
                score = self.calculate_score()
                progress = self.rooms_cleared / 12.0
                reward += score / 50.0
                reward += progress * 0.5

        return obs, reward, done, info


def collect_episode(env, model, entropy_coef):
    """Collect a single episode of experience."""
    states = []
    actions = []
    log_probs = []
    values = []
    rewards = []
    entropies = []

    state = env.reset()

    while not env.done:
        valid_actions = env.get_valid_actions()
        action, log_prob, value, entropy = model.get_action(state, valid_actions)

        states.append(state)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        entropies.append(entropy)

        next_state, reward, done, _ = env.step(action)
        rewards.append(reward)

        state = next_state

    return states, actions, log_probs, values, rewards, entropies, env.calculate_score(), env.won


def train_batch(model, optimizer, batch_data, gamma=0.99, entropy_coef=0.02,
                clip_epsilon=0.2, value_coef=0.5):
    """Train on a batch of episodes (PPO-style)."""
    all_returns = []
    all_advantages = []
    all_log_probs = []
    all_values = []
    all_entropies = []

    for log_probs, values, rewards, entropies in batch_data:
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns)

        values_t = torch.stack(values)
        advantages = returns - values_t.detach()

        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        all_returns.append(returns)
        all_advantages.append(advantages)
        all_log_probs.append(torch.stack(log_probs))
        all_values.append(values_t)
        all_entropies.append(torch.stack(entropies))

    # Concatenate all episodes
    returns = torch.cat(all_returns)
    advantages = torch.cat(all_advantages)
    log_probs = torch.cat(all_log_probs)
    values = torch.cat(all_values)
    entropies = torch.cat(all_entropies)

    # Compute losses
    policy_loss = -(log_probs * advantages).mean()
    value_loss = value_coef * ((returns - values) ** 2).mean()
    entropy_loss = -entropy_coef * entropies.mean()

    total_loss = policy_loss + value_loss + entropy_loss

    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()

    return total_loss.item()


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
    parser = argparse.ArgumentParser(description='Long Training Run')
    parser.add_argument('--trajectories', type=str, default='winning_trajectories.pkl')
    parser.add_argument('--episodes', type=int, default=2000000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--entropy-start', type=float, default=0.05)
    parser.add_argument('--entropy-end', type=float, default=0.01)
    parser.add_argument('--eval-interval', type=int, default=25000)
    parser.add_argument('--winning-seed-ratio', type=float, default=0.15)
    parser.add_argument('--output', type=str, default='rl_long.pt')
    args = parser.parse_args()

    # Load winning trajectories for their seeds
    print("Loading winning trajectories...")
    with open(args.trajectories, 'rb') as f:
        trajectories = pickle.load(f)
    winning_seeds = [t['seed'] for t in trajectories]
    print(f"Loaded {len(winning_seeds)} winning seeds")

    print(f"\n=== LONG TRAINING RUN ===")
    print(f"Episodes: {args.episodes:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Entropy: {args.entropy_start} -> {args.entropy_end}")
    print(f"Winning seed ratio: {args.winning_seed_ratio}")
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

    start_time = time.time()
    episode = 0

    while episode < args.episodes:
        # Entropy annealing
        progress = episode / args.episodes
        entropy_coef = args.entropy_start + (args.entropy_end - args.entropy_start) * progress

        # Collect batch of episodes
        batch_data = []
        batch_scores = []
        batch_wins = []

        for _ in range(args.batch_size):
            use_winning_seed = np.random.random() < args.winning_seed_ratio
            if use_winning_seed and winning_seeds:
                seed = int(np.random.choice(winning_seeds))
                attempts_on_winning_seeds += 1
            else:
                seed = None

            env = ShapedRewardEnv(seed=seed)
            states, actions, log_probs, values, rewards, entropies, score, won = \
                collect_episode(env, model, entropy_coef)

            batch_data.append((log_probs, values, rewards, entropies))
            batch_scores.append(score)
            batch_wins.append(won)

            if use_winning_seed and won:
                wins_on_winning_seeds += 1

            recent_scores.append(score)
            recent_wins.append(1 if won else 0)

            episode += 1

        # Train on batch
        train_batch(model, optimizer, batch_data, args.gamma, entropy_coef)

        # Periodic evaluation
        if episode % args.eval_interval < args.batch_size:
            elapsed = time.time() - start_time
            eps_per_sec = episode / elapsed
            remaining = (args.episodes - episode) / eps_per_sec

            avg_score = np.mean(recent_scores)
            win_rate = np.mean(recent_wins)
            eval_win_rate, eval_avg_score = evaluate(model, num_episodes=2000)

            print(f"Episode {episode:,} ({100*episode/args.episodes:.1f}%):")
            print(f"  Training: win_rate={win_rate:.4f}, avg_score={avg_score:.1f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}")
            print(f"  Entropy:  {entropy_coef:.4f}")
            if attempts_on_winning_seeds > 0:
                seed_win_rate = wins_on_winning_seeds / attempts_on_winning_seeds
                print(f"  Winning seeds: {wins_on_winning_seeds}/{attempts_on_winning_seeds} ({seed_win_rate:.2%})")
            print(f"  Speed: {eps_per_sec:.1f} eps/sec, ETA: {remaining/3600:.1f}h")

            if eval_win_rate > best_win_rate or (eval_win_rate == best_win_rate and eval_avg_score > best_avg_score):
                best_win_rate = eval_win_rate
                best_avg_score = eval_avg_score
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  *** New best! win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f} ***")

                # Save checkpoint
                torch.save({
                    'state_dict': model.state_dict(),
                    'win_rate': eval_win_rate,
                    'avg_score': eval_avg_score,
                    'episode': episode,
                }, f'{args.output}.checkpoint')

            print()

    if best_model_state:
        model.load_state_dict(best_model_state)

    total_time = time.time() - start_time
    print(f"\n=== TRAINING COMPLETE ===")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total episodes: {episode:,}")

    print("\n=== Final Evaluation (10k episodes) ===")
    final_win_rate, final_avg_score = evaluate(model, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg score: {final_avg_score:.1f}")

    # Test on winning seeds
    print("\nTesting on known winning seeds:")
    seed_wins = 0
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
        if env.won:
            seed_wins += 1
        print(f"  Seed {seed}: {result}")
    print(f"Won {seed_wins}/{len(winning_seeds)} known winning seeds")

    torch.save({
        'state_dict': model.state_dict(),
        'win_rate': final_win_rate,
        'avg_score': final_avg_score,
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
