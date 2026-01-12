#!/usr/bin/env python3
"""Analyze score distributions for different policies."""

import numpy as np
import torch
from scoundrel_env import ScoundrelEnv, CardType
from train_rl_dense import ActorCritic


def calculate_score(env):
    if env.won:
        return float(env.health)
    remaining = sum(c.value for c in env.deck if c.card_type == CardType.MONSTER)
    remaining += sum(c.value for c in env.room if c.card_type == CardType.MONSTER)
    return float(env.health - remaining)


def random_policy_scores(num_games=10000):
    """Get score distribution for random policy."""
    import random
    scores = []
    for seed in range(num_games):
        env = ScoundrelEnv(seed=seed)
        env.reset()
        while not env.done:
            action = random.choice(env.get_valid_actions())
            env.step(action)
        scores.append(calculate_score(env))
    return scores


def rl_policy_scores(model_path='ml/rl_dense.pt', num_games=10000):
    """Get score distribution for trained RL policy."""
    checkpoint = torch.load(model_path, weights_only=True)
    model = ActorCritic()
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    scores = []
    for seed in range(num_games):
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
        scores.append(calculate_score(env))
    return scores


def simple_heuristic_scores(num_games=10000):
    """Simple heuristic: leave biggest monster."""
    scores = []
    for seed in range(num_games):
        env = ScoundrelEnv(seed=seed)
        env.reset()
        while not env.done:
            room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))
            monsters = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.MONSTER]
            if monsters:
                leave_idx = max(monsters, key=lambda x: x[1].value)[0]
            else:
                leave_idx = 0
            env.step(leave_idx + 1)
        scores.append(calculate_score(env))
    return scores


def print_stats(name, scores):
    """Print score statistics."""
    scores = np.array(scores)
    wins = np.sum(scores > 0)
    print(f"\n{name}:")
    print(f"  Games: {len(scores)}")
    print(f"  Wins: {wins} ({100*wins/len(scores):.3f}%)")
    print(f"  Mean: {np.mean(scores):.1f}")
    print(f"  Median: {np.median(scores):.1f}")
    print(f"  Std: {np.std(scores):.1f}")
    print(f"  Min: {np.min(scores):.0f}")
    print(f"  Max: {np.max(scores):.0f}")
    print(f"  25th percentile: {np.percentile(scores, 25):.1f}")
    print(f"  75th percentile: {np.percentile(scores, 75):.1f}")

    # Score brackets
    brackets = [(-200, -150), (-150, -100), (-100, -50), (-50, 0), (0, 20)]
    print("  Score brackets:")
    for low, high in brackets:
        count = np.sum((scores >= low) & (scores < high))
        print(f"    [{low}, {high}): {count} ({100*count/len(scores):.1f}%)")


if __name__ == '__main__':
    print("Analyzing score distributions (10000 games each)...")

    random_scores = random_policy_scores()
    print_stats("Random Policy", random_scores)

    heuristic_scores = simple_heuristic_scores()
    print_stats("Simple Heuristic (leave biggest monster)", heuristic_scores)

    try:
        rl_scores = rl_policy_scores()
        print_stats("Trained RL Policy", rl_scores)
    except Exception as e:
        print(f"\nCouldn't load RL model: {e}")
