#!/usr/bin/env python3
"""
Automated bootstrapping: collect wins -> train -> repeat
Run this after v4 training completes.
"""

import subprocess
import time
import pickle
import torch
import numpy as np
from scoundrel_env import ScoundrelEnv, CardType


class ActorCritic(torch.nn.Module):
    def __init__(self, input_size=26, hidden_size=512):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 2, hidden_size // 4),
            torch.nn.ReLU(),
        )
        self.actor = torch.nn.Linear(hidden_size // 4, 5)
        self.critic = torch.nn.Linear(hidden_size // 4, 1)

    def forward(self, x):
        features = self.shared(x)
        return self.actor(features), self.critic(features)


def collect_winning_seeds(model_path, output_path, num_seeds=1500000):
    """Collect winning seeds using a trained model."""
    print(f"\n{'='*50}")
    print(f"Collecting winning seeds from {model_path}")
    print(f"Searching {num_seeds:,} seeds...")
    print(f"{'='*50}\n")

    model = ActorCritic()
    checkpoint = torch.load(model_path, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    trajectories = []
    start_time = time.time()

    for seed in range(num_seeds):
        env = ScoundrelEnv(seed=seed)
        state = env.reset()
        trajectory = []

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

            trajectory.append((state.copy(), action, valid_actions.copy()))
            state, _, _, _ = env.step(action)

        if env.won:
            trajectories.append({
                'seed': seed,
                'health': env.health,
                'trajectory': trajectory
            })
            if len(trajectories) <= 10 or len(trajectories) % 1000 == 0:
                print(f"Win #{len(trajectories)}: seed={seed}, health={env.health}")

        if seed % 100000 == 0 and seed > 0:
            elapsed = time.time() - start_time
            rate = seed / elapsed
            eta = (num_seeds - seed) / rate
            print(f"Progress: {seed:,} seeds, {len(trajectories)} wins ({100*len(trajectories)/seed:.3f}%), ETA: {eta/60:.1f}min")

    win_rate = len(trajectories) / num_seeds
    print(f"\nFound {len(trajectories)} winning seeds ({100*win_rate:.3f}%)")

    with open(output_path, 'wb') as f:
        pickle.dump(trajectories, f)
    print(f"Saved to {output_path}")

    return len(trajectories), win_rate


def train_model(trajectories_path, output_path, episodes=2000000):
    """Train a model on collected trajectories."""
    print(f"\n{'='*50}")
    print(f"Training on {trajectories_path}")
    print(f"Output: {output_path}")
    print(f"{'='*50}\n")

    cmd = [
        'python', '-u', 'train_long.py',
        '--trajectories', trajectories_path,
        '--episodes', str(episodes),
        '--winning-seed-ratio', '0.20',
        '--output', output_path
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    for line in process.stdout:
        print(line, end='')

    process.wait()
    return process.returncode == 0


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Automated Bootstrap Training')
    parser.add_argument('--start-version', type=int, default=4, help='Starting version number')
    parser.add_argument('--iterations', type=int, default=3, help='Number of bootstrap iterations')
    parser.add_argument('--collect-seeds', type=int, default=1500000, help='Seeds to search per iteration')
    parser.add_argument('--train-episodes', type=int, default=2000000, help='Training episodes per iteration')
    args = parser.parse_args()

    version = args.start_version

    print(f"\n{'#'*60}")
    print(f"# AUTOMATED BOOTSTRAP TRAINING")
    print(f"# Starting from v{version}, running {args.iterations} iterations")
    print(f"{'#'*60}\n")

    results = []

    for i in range(args.iterations):
        print(f"\n{'*'*60}")
        print(f"* ITERATION {i+1}/{args.iterations} - Creating v{version+1}")
        print(f"{'*'*60}")

        # Collect winning seeds
        model_path = f'rl_v{version}.pt'
        trajectories_path = f'winning_trajectories_v{version+1}.pkl'

        num_wins, win_rate = collect_winning_seeds(
            model_path,
            trajectories_path,
            num_seeds=args.collect_seeds
        )

        # Train new model
        new_model_path = f'rl_v{version+1}.pt'
        success = train_model(
            trajectories_path,
            new_model_path,
            episodes=args.train_episodes
        )

        if not success:
            print(f"Training failed for v{version+1}!")
            break

        results.append({
            'version': version + 1,
            'seeds_collected': num_wins,
            'collection_rate': win_rate
        })

        version += 1

    # Print summary
    print(f"\n{'#'*60}")
    print(f"# BOOTSTRAP COMPLETE - SUMMARY")
    print(f"{'#'*60}\n")

    for r in results:
        print(f"v{r['version']}: {r['seeds_collected']:,} seeds collected ({100*r['collection_rate']:.3f}%)")

    print(f"\nFinal model: rl_v{version}.pt")


if __name__ == '__main__':
    main()
