#!/usr/bin/env python3
"""
Imitation Learning: Train from winning trajectories, then fine-tune with RL.

Strategy:
1. Pre-train on winning trajectories using behavioral cloning
2. Fine-tune with PPO while preserving winning behaviors
"""

import pickle
import numpy as np
from collections import deque
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader

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


class TrajectoryDataset(Dataset):
    """Dataset of (state, action, valid_actions) from winning trajectories."""

    def __init__(self, trajectories: List[dict]):
        self.samples = []
        for traj in trajectories:
            for state, action, valid_actions in traj['trajectory']:
                # Skip malformed entries (last entry sometimes has int instead of list)
                if isinstance(valid_actions, list):
                    self.samples.append((state, action, valid_actions))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        state, action, valid_actions = self.samples[idx]
        # Create valid action mask
        mask = torch.zeros(5)
        for a in valid_actions:
            mask[a] = 1.0
        return torch.from_numpy(state), torch.tensor(action, dtype=torch.long), mask


def pretrain_on_trajectories(model: ActorCritic, trajectories: List[dict],
                              epochs: int = 100, lr: float = 0.001):
    """Behavioral cloning: train to mimic winning actions."""
    dataset = TrajectoryDataset(trajectories)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"Pre-training on {len(dataset)} state-action pairs from {len(trajectories)} winning games")

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for states, actions, masks in dataloader:
            optimizer.zero_grad()

            logits, _ = model(states)

            # Mask out invalid actions before computing loss
            masked_logits = logits.clone()
            masked_logits[masks == 0] = float('-inf')

            loss = criterion(masked_logits, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predictions = masked_logits.argmax(dim=1)
            correct += (predictions == actions).sum().item()
            total += len(actions)

        if (epoch + 1) % 10 == 0:
            accuracy = correct / total
            print(f"Epoch {epoch + 1}: loss={total_loss / len(dataloader):.4f}, accuracy={accuracy:.2%}")

    return model


class ShapedRewardEnv:
    """Wrapper with shaped rewards for RL fine-tuning."""

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
                # Big bonus for winning
                reward += 10.0 + self.env.health / 2.0
            else:
                score = self.calculate_score()
                progress = self.rooms_cleared / 12.0
                reward += score / 50.0
                reward += progress * 0.5

        return obs, reward, done, info


def finetune_with_rl(model: ActorCritic, num_episodes: int = 50000,
                      lr: float = 0.0001, gamma: float = 0.99,
                      entropy_coef: float = 0.02, kl_penalty: float = 0.1):
    """Fine-tune with RL while preserving pre-trained behaviors."""
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Clone the pretrained model for KL regularization
    pretrained_model = ActorCritic()
    pretrained_model.load_state_dict(model.state_dict())
    pretrained_model.eval()

    recent_scores = deque(maxlen=1000)
    recent_wins = deque(maxlen=1000)
    best_win_rate = 0
    best_avg_score = float('-inf')
    best_model_state = None

    for episode in range(num_episodes):
        env = ShapedRewardEnv()
        log_probs = []
        values = []
        rewards = []
        entropies = []
        kl_terms = []

        state = env.reset()

        while not env.done:
            valid_actions = env.get_valid_actions()
            action, log_prob, value, entropy = model.get_action(state, valid_actions)

            # Compute KL divergence from pretrained policy
            with torch.no_grad():
                state_t = torch.from_numpy(state).unsqueeze(0)
                old_logits, _ = pretrained_model(state_t)
                old_logits = old_logits.squeeze(0)

            new_logits, _ = model(state_t)
            new_logits = new_logits.squeeze(0)

            # Mask for valid actions
            mask = torch.full((5,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0

            old_probs = F.softmax(old_logits + mask, dim=0)
            new_probs = F.softmax(new_logits + mask, dim=0)

            # KL(new || old) - penalize deviation from pretrained
            kl = (new_probs * (new_probs.log() - old_probs.log())).sum()
            kl_terms.append(kl)

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
        kl_terms = torch.stack(kl_terms)

        advantages = returns - values.detach()

        policy_loss = -(log_probs * advantages).mean()
        value_loss = 0.5 * ((returns - values) ** 2).mean()
        entropy_loss = -entropy_coef * entropies.mean()
        kl_loss = kl_penalty * kl_terms.mean()

        total_loss = policy_loss + value_loss + entropy_loss + kl_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        score = env.calculate_score()
        won = env.won
        recent_scores.append(score)
        recent_wins.append(1 if won else 0)

        if (episode + 1) % 5000 == 0:
            avg_score = np.mean(recent_scores)
            win_rate = np.mean(recent_wins)
            eval_win_rate, eval_avg_score = evaluate(model)

            print(f"Episode {episode + 1}:")
            print(f"  Training: win_rate={win_rate:.4f}, avg_score={avg_score:.1f}")
            print(f"  Eval:     win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}")

            # Save best by win rate (primary) or score (secondary)
            if eval_win_rate > best_win_rate or (eval_win_rate == best_win_rate and eval_avg_score > best_avg_score):
                best_win_rate = eval_win_rate
                best_avg_score = eval_avg_score
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                print(f"  New best! win_rate={eval_win_rate:.4f}, avg_score={eval_avg_score:.1f}")

    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


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
    parser = argparse.ArgumentParser(description='Imitation Learning + RL Fine-tuning')
    parser.add_argument('--trajectories', type=str, default='winning_trajectories.pkl')
    parser.add_argument('--pretrain-epochs', type=int, default=200)
    parser.add_argument('--finetune-episodes', type=int, default=100000)
    parser.add_argument('--output', type=str, default='ml/rl_imitation.pt')
    args = parser.parse_args()

    # Load winning trajectories
    print("Loading winning trajectories...")
    with open(args.trajectories, 'rb') as f:
        trajectories = pickle.load(f)
    print(f"Loaded {len(trajectories)} winning trajectories")

    # Create model
    model = ActorCritic()

    # Phase 1: Behavioral cloning
    print("\n=== Phase 1: Behavioral Cloning ===")
    pretrain_on_trajectories(model, trajectories, epochs=args.pretrain_epochs)

    # Evaluate after pretraining
    print("\nEvaluating pre-trained model...")
    win_rate, avg_score = evaluate(model)
    print(f"Pre-trained: win_rate={win_rate:.4f}, avg_score={avg_score:.1f}")

    # Phase 2: RL fine-tuning with KL penalty
    print("\n=== Phase 2: RL Fine-tuning ===")
    finetune_with_rl(model, num_episodes=args.finetune_episodes)

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_win_rate, final_avg_score = evaluate(model, num_episodes=10000)
    print(f"Win rate: {final_win_rate:.4f} ({final_win_rate * 10000:.0f}/10000)")
    print(f"Avg score: {final_avg_score:.1f}")

    # Save model
    torch.save({
        'state_dict': model.state_dict(),
        'win_rate': final_win_rate,
        'avg_score': final_avg_score,
    }, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == '__main__':
    main()
