#!/usr/bin/env python3
"""
Ensemble evaluation: Load multiple models and have them vote on actions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List
from scoundrel_env import ScoundrelEnv, CardType


class ActorCritic(nn.Module):
    def __init__(self, input_size=26, hidden_size=512):
        super().__init__()
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


class EnsemblePlayer:
    """Ensemble of models that vote on actions."""

    def __init__(self, model_paths: List[str]):
        self.models = []
        for path in model_paths:
            model = ActorCritic()
            checkpoint = torch.load(path, weights_only=True)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            self.models.append(model)
        print(f"Loaded {len(self.models)} models for ensemble")

    def get_action(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Get action by majority vote."""
        state_t = torch.from_numpy(state).unsqueeze(0)

        votes = []
        for model in self.models:
            with torch.no_grad():
                logits, _ = model(state_t)
                logits = logits.squeeze(0)

            # Mask invalid actions
            mask = torch.full((5,), float('-inf'))
            for a in valid_actions:
                mask[a] = 0
            masked_logits = logits + mask

            # Each model votes for its preferred action
            action = masked_logits.argmax().item()
            votes.append(action)

        # Majority vote (most common action)
        from collections import Counter
        vote_counts = Counter(votes)
        winner = vote_counts.most_common(1)[0][0]
        return winner

    def get_action_soft(self, state: np.ndarray, valid_actions: List[int]) -> int:
        """Get action by averaging probabilities (soft voting)."""
        state_t = torch.from_numpy(state).unsqueeze(0)

        # Mask for valid actions
        mask = torch.full((5,), float('-inf'))
        for a in valid_actions:
            mask[a] = 0

        # Average the probabilities from all models
        avg_probs = torch.zeros(5)
        for model in self.models:
            with torch.no_grad():
                logits, _ = model(state_t)
                logits = logits.squeeze(0)
            masked_logits = logits + mask
            probs = torch.softmax(masked_logits, dim=0)
            avg_probs += probs

        avg_probs /= len(self.models)
        return avg_probs.argmax().item()


def evaluate_single(model_path: str, num_episodes: int = 10000):
    """Evaluate a single model."""
    model = ActorCritic()
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

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
            wins += 1
            total_score += env.health
        else:
            remaining = sum(c.value for c in env.deck if c.card_type == CardType.MONSTER)
            remaining += sum(c.value for c in env.room if c.card_type == CardType.MONSTER)
            total_score += env.health - remaining

    return wins, total_score / num_episodes


def evaluate_ensemble(model_paths: List[str], num_episodes: int = 10000, soft_vote: bool = False):
    """Evaluate ensemble of models."""
    ensemble = EnsemblePlayer(model_paths)

    wins = 0
    total_score = 0

    for seed in range(num_episodes):
        env = ScoundrelEnv(seed=seed)
        state = env.reset()

        while not env.done:
            valid_actions = env.get_valid_actions()
            if soft_vote:
                action = ensemble.get_action_soft(state, valid_actions)
            else:
                action = ensemble.get_action(state, valid_actions)
            state, _, _, _ = env.step(action)

        if env.won:
            wins += 1
            total_score += env.health
        else:
            remaining = sum(c.value for c in env.deck if c.card_type == CardType.MONSTER)
            remaining += sum(c.value for c in env.room if c.card_type == CardType.MONSTER)
            total_score += env.health - remaining

    return wins, total_score / num_episodes


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Ensemble Evaluation')
    parser.add_argument('models', nargs='+', help='Model paths')
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--soft', action='store_true', help='Use soft voting')
    args = parser.parse_args()

    print(f"\n=== Ensemble Evaluation ===")
    print(f"Models: {args.models}")
    print(f"Episodes: {args.episodes}")
    print(f"Voting: {'soft' if args.soft else 'hard (majority)'}")
    print()

    # Evaluate each model individually
    print("Individual model performance:")
    for path in args.models:
        wins, avg_score = evaluate_single(path, args.episodes)
        win_rate = wins / args.episodes
        print(f"  {path}: {win_rate:.2%} ({wins}/{args.episodes}), avg_score={avg_score:.1f}")

    # Evaluate ensemble with hard voting
    print(f"\nEnsemble (hard vote):")
    wins, avg_score = evaluate_ensemble(args.models, args.episodes, soft_vote=False)
    win_rate = wins / args.episodes
    print(f"  Win rate: {win_rate:.2%} ({wins}/{args.episodes})")
    print(f"  Avg score: {avg_score:.1f}")

    # Evaluate ensemble with soft voting
    print(f"\nEnsemble (soft vote):")
    wins, avg_score = evaluate_ensemble(args.models, args.episodes, soft_vote=True)
    win_rate = wins / args.episodes
    print(f"  Win rate: {win_rate:.2%} ({wins}/{args.episodes})")
    print(f"  Avg score: {avg_score:.1f}")


if __name__ == '__main__':
    main()
