#!/usr/bin/env python3
"""
Neural network training script for Scoundrel game bot.

This trains a neural network to predict:
1. Whether to skip a room
2. Which card to leave behind

Usage:
    python train_scoundrel_nn.py --data ../app/build/training_data.csv
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not found. Install with: pip install torch")


class ScoundrelNet(nn.Module):
    """
    Neural network for Scoundrel game decisions.

    Input: 26 game state features
    Output: 5 values (skip, leave_card_0, leave_card_1, leave_card_2, leave_card_3)
    """

    def __init__(self, input_size=26, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 5),  # skip + 4 cards
        )

    def forward(self, x):
        return self.network(x)


def load_data(csv_path: str) -> tuple:
    """Load training data from CSV."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"  Total examples: {len(df)}")
    print(f"  Winning games: {df['game_won'].sum()}")
    print(f"  Skip decisions: {(df['decision_type'] == 1).sum()}")
    print(f"  Leave card decisions: {(df['decision_type'] == 0).sum()}")

    # Feature columns (first 26)
    feature_cols = df.columns[:26]
    X = df[feature_cols].values.astype(np.float32)

    # Target: combine decision_type and card_index into single label
    # 0 = skip, 1-4 = leave card 0-3
    y = np.where(
        df['decision_type'] == 1,
        0,  # skip
        df['decision_card_index'] + 1  # leave card (1-4)
    ).astype(np.int64)

    # Also get game_won for weighting
    won = df['game_won'].values.astype(np.float32)

    return X, y, won


def train_model(X: np.ndarray, y: np.ndarray, won: np.ndarray,
                epochs: int = 50, batch_size: int = 256,
                weight_wins: float = 2.0) -> ScoundrelNet:
    """Train the neural network."""

    # Create sample weights (weight winning games more)
    sample_weights = np.where(won == 1, weight_wins, 1.0).astype(np.float32)

    # Split into train/val
    n = len(X)
    indices = np.random.permutation(n)
    train_idx = indices[:int(0.8 * n)]
    val_idx = indices[int(0.8 * n):]

    X_train, y_train, w_train = X[train_idx], y[train_idx], sample_weights[train_idx]
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
    model = ScoundrelNet()
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"\nTraining on {len(X_train)} examples, validating on {len(X_val)}...")

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

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_preds = val_outputs.argmax(dim=1)
            val_acc = (val_preds == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.4f}")

    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    return model


def analyze_model(model: ScoundrelNet, X: np.ndarray, y: np.ndarray):
    """Analyze model predictions."""
    model.eval()

    X_t = torch.from_numpy(X)

    with torch.no_grad():
        outputs = model(X_t)
        preds = outputs.argmax(dim=1).numpy()

    # Overall accuracy
    acc = (preds == y).mean()
    print(f"\nOverall accuracy: {acc:.4f}")

    # Per-class accuracy
    for i in range(5):
        mask = y == i
        if mask.sum() > 0:
            class_acc = (preds[mask] == y[mask]).mean()
            label = "skip" if i == 0 else f"leave card {i-1}"
            print(f"  {label}: {class_acc:.4f} ({mask.sum()} examples)")

    # Confusion analysis
    print("\nConfusion (rows=actual, cols=predicted):")
    for i in range(5):
        row = []
        for j in range(5):
            count = ((y == i) & (preds == j)).sum()
            row.append(f"{count:5d}")
        label = "skip" if i == 0 else f"card{i-1}"
        print(f"  {label}: {' '.join(row)}")


def export_model(model: ScoundrelNet, output_path: str):
    """Export model weights for use in Android."""
    model.eval()

    # Save PyTorch model
    pt_path = output_path.replace('.onnx', '.pt')
    torch.save(model.state_dict(), pt_path)
    print(f"\nSaved PyTorch model to {pt_path}")

    # Try ONNX export (optional)
    try:
        dummy_input = torch.randn(1, 26)
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['features'],
            output_names=['decision'],
            dynamic_axes={
                'features': {0: 'batch_size'},
                'decision': {0: 'batch_size'}
            }
        )
        print(f"Exported ONNX model to {output_path}")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("PyTorch model saved successfully, ONNX optional.")


def main():
    parser = argparse.ArgumentParser(description='Train Scoundrel NN')
    parser.add_argument('--data', type=str, required=True, help='Path to training CSV')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size')
    parser.add_argument('--weight-wins', type=float, default=2.0, help='Weight for winning game examples')
    parser.add_argument('--output', type=str, default='scoundrel_model.onnx', help='Output model path')
    args = parser.parse_args()

    if not HAS_TORCH:
        print("Please install PyTorch first: pip install torch")
        return

    # Load data
    X, y, won = load_data(args.data)

    # Train model
    model = train_model(X, y, won, epochs=args.epochs, batch_size=args.batch_size, weight_wins=args.weight_wins)

    # Analyze
    analyze_model(model, X, y)

    # Export
    export_model(model, args.output)

    # Also save PyTorch model
    torch.save(model.state_dict(), args.output.replace('.onnx', '.pt'))
    print(f"Saved PyTorch model to {args.output.replace('.onnx', '.pt')}")


if __name__ == '__main__':
    main()
