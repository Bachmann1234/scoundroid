#!/usr/bin/env python3
"""
Neural network training script for Scoundrel game bot - Version 2.

This version predicts a score for EACH card (leave or not), rather than
predicting which card index to leave. This handles the permutation-invariant
nature of the cards better.

The network takes: game state + ONE card's features
And outputs: score for leaving that card

During inference, we evaluate all 4 cards and pick the one with highest leave score.
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Install with: pip install torch")


class CardScorerNet(nn.Module):
    """
    Neural network that scores a single card for leaving.

    Input:
        - Game state features (health, weapon, deck info) = 18 features
        - Single card features (type, value) = 2 features
        - Card context (is it the max monster, best weapon, etc.) = 6 features
    Total: 26 features

    Output: Score for leaving this card (higher = more likely to leave)
    """

    def __init__(self, input_size=26, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),  # Single score output
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def create_card_scoring_data(csv_path: str) -> tuple:
    """
    Transform the training data for card scoring.

    For each decision where we left a card, create:
    - Positive example: the card that was left (label=1)
    - Negative examples: the cards that were NOT left (label=0)

    For skip decisions, we create examples showing skip was better than leaving any card.
    """
    print(f"Loading and transforming data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Only use "leave card" decisions (decision_type == 0)
    leave_df = df[df['decision_type'] == 0].copy()

    print(f"  Total leave decisions: {len(leave_df)}")

    # We need to create a new representation
    # Game state features (same for all cards in a room):
    game_features = [
        'health', 'health_fraction',
        'has_weapon', 'weapon_value', 'weapon_max_monster', 'weapon_is_fresh',
        'cards_remaining', 'monsters_remaining', 'weapons_remaining', 'potions_remaining',
        'last_room_skipped', 'can_skip',
        'total_monster_value', 'max_monster_value', 'total_potion_value', 'max_weapon_value',
    ]  # 16 features

    examples = []
    labels = []
    weights = []

    for idx, row in leave_df.iterrows():
        card_left = int(row['decision_card_index'])
        game_won = row['game_won']

        # Extract game state
        game_state = [row[f] for f in game_features]

        # For each card in the room
        for card_idx in range(4):
            card_type = row[f'card{card_idx}_type']
            card_value = row[f'card{card_idx}_value']

            # Card context features
            is_max_monster = 1.0 if (card_type == 0 and card_value == row['max_monster_value']) else 0.0
            is_max_weapon = 1.0 if (card_type == 0.5 and card_value == row['max_weapon_value']) else 0.0
            is_monster = 1.0 if card_type == 0 else 0.0
            is_weapon = 1.0 if card_type == 0.5 else 0.0
            is_potion = 1.0 if card_type == 1.0 else 0.0
            value_normalized = card_value  # Already normalized in data

            # Combine features
            features = game_state + [
                card_type, card_value,
                is_monster, is_weapon, is_potion,
                is_max_monster, is_max_weapon, value_normalized
            ]

            examples.append(features)

            # Label: 1 if this card was left, 0 otherwise
            labels.append(1.0 if card_idx == card_left else 0.0)

            # Weight winning examples more
            weights.append(2.0 if game_won else 1.0)

    X = np.array(examples, dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    w = np.array(weights, dtype=np.float32)

    print(f"  Created {len(X)} card examples ({y.sum():.0f} positive, {len(y) - y.sum():.0f} negative)")

    return X, y, w


def train_card_scorer(X: np.ndarray, y: np.ndarray, weights: np.ndarray,
                      epochs: int = 50, batch_size: int = 512) -> CardScorerNet:
    """Train the card scoring network."""

    # Split into train/val
    n = len(X)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.8 * n)]
    val_idx = indices[int(0.8 * n):]

    X_train, y_train, w_train = X[train_idx], y[train_idx], weights[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    w_train_t = torch.from_numpy(w_train)
    X_val_t = torch.from_numpy(X_val)
    y_val_t = torch.from_numpy(y_val)

    # Create data loader
    train_dataset = TensorDataset(X_train_t, y_train_t, w_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create model
    input_size = X.shape[1]
    model = CardScorerNet(input_size=input_size)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining on {len(X_train)} examples, validating on {len(X_val)}...")
    print(f"Input size: {input_size}")

    best_val_acc = 0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch_X, batch_y, batch_w in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            weighted_loss = (loss * batch_w).mean()
            weighted_loss.backward()
            optimizer.step()
            total_loss += weighted_loss.item()

        # Validation - compute accuracy on "which card to leave" decisions
        # Group by rooms (every 4 consecutive examples)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_probs = torch.sigmoid(val_outputs)

            # Accuracy: for each room, did we pick the right card?
            correct = 0
            total = 0
            for i in range(0, len(val_probs), 4):
                if i + 4 > len(val_probs):
                    break
                room_probs = val_probs[i:i+4]
                room_labels = y_val_t[i:i+4]

                pred_card = room_probs.argmax().item()
                true_card = room_labels.argmax().item()

                if pred_card == true_card:
                    correct += 1
                total += 1

            val_acc = correct / total if total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, room_acc={val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation room accuracy: {best_val_acc:.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description='Train Scoundrel Card Scorer')
    parser.add_argument('--data', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--output', type=str, default='ml/card_scorer.pt', help='Output model path')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("Please install PyTorch first: pip install torch")
        return

    # Load and transform data
    X, y, w = create_card_scoring_data(args.data)

    # Train model
    model = train_card_scorer(X, y, w, epochs=args.epochs)

    # Save model
    torch.save({
        'state_dict': model.state_dict(),
        'input_size': X.shape[1],
    }, args.output)
    print(f"\nSaved model to {args.output}")


if __name__ == '__main__':
    main()
