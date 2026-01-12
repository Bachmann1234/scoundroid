#!/usr/bin/env python3
"""
Evaluate the trained neural network on the training data to understand
what it's learning.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class CardScorerNet(nn.Module):
    def __init__(self, input_size=24, hidden_size=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def analyze_predictions():
    # Load model
    checkpoint = torch.load('ml/card_scorer.pt')
    model = CardScorerNet(input_size=checkpoint['input_size'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Load data
    df = pd.read_csv('app/build/training_data.csv')
    leave_df = df[df['decision_type'] == 0].copy()

    print(f"Analyzing {len(leave_df)} leave decisions...")

    # Game state features
    game_features = [
        'health', 'health_fraction',
        'has_weapon', 'weapon_value', 'weapon_max_monster', 'weapon_is_fresh',
        'cards_remaining', 'monsters_remaining', 'weapons_remaining', 'potions_remaining',
        'last_room_skipped', 'can_skip',
        'total_monster_value', 'max_monster_value', 'total_potion_value', 'max_weapon_value',
    ]

    # Stats by card type
    results = {
        'monster_left_correct': 0, 'monster_left_total': 0,
        'weapon_left_correct': 0, 'weapon_left_total': 0,
        'potion_left_correct': 0, 'potion_left_total': 0,
    }

    # Track disagreements
    disagreements = []

    for idx, row in leave_df.head(10000).iterrows():  # Sample for speed
        card_left = int(row['decision_card_index'])

        # Build features for each card
        features_list = []
        for card_idx in range(4):
            card_type = row[f'card{card_idx}_type']
            card_value = row[f'card{card_idx}_value']

            game_state = [row[f] for f in game_features]

            is_max_monster = 1.0 if (card_type == 0 and card_value == row['max_monster_value']) else 0.0
            is_max_weapon = 1.0 if (card_type == 0.5 and card_value == row['max_weapon_value']) else 0.0
            is_monster = 1.0 if card_type == 0 else 0.0
            is_weapon = 1.0 if card_type == 0.5 else 0.0
            is_potion = 1.0 if card_type == 1.0 else 0.0
            value_normalized = card_value

            features = game_state + [
                card_type, card_value,
                is_monster, is_weapon, is_potion,
                is_max_monster, is_max_weapon, value_normalized
            ]
            features_list.append(features)

        # Get model predictions
        X = torch.tensor(features_list, dtype=torch.float32)
        with torch.no_grad():
            scores = model(X)
            pred_card = scores.argmax().item()

        # What type of card did the heuristic leave?
        left_type = row[f'card{card_left}_type']
        if left_type == 0:  # Monster
            results['monster_left_total'] += 1
            if pred_card == card_left:
                results['monster_left_correct'] += 1
        elif left_type == 0.5:  # Weapon
            results['weapon_left_total'] += 1
            if pred_card == card_left:
                results['weapon_left_correct'] += 1
        else:  # Potion
            results['potion_left_total'] += 1
            if pred_card == card_left:
                results['potion_left_correct'] += 1

        # Track disagreements for analysis
        if pred_card != card_left:
            pred_type = row[f'card{pred_card}_type']
            disagreements.append({
                'heuristic_type': left_type,
                'nn_type': pred_type,
                'heuristic_value': row[f'card{card_left}_value'],
                'nn_value': row[f'card{pred_card}_value'],
                'game_won': row['game_won'],
            })

    # Print results
    print("\nAccuracy by card type left:")
    for card_type, name in [(0, 'Monster'), (0.5, 'Weapon'), (1.0, 'Potion')]:
        key = name.lower()
        total = results[f'{key}_left_total']
        correct = results[f'{key}_left_correct']
        if total > 0:
            print(f"  {name}: {correct}/{total} = {100*correct/total:.1f}%")

    # Analyze disagreements
    print(f"\nDisagreements: {len(disagreements)}")
    if disagreements:
        dis_df = pd.DataFrame(disagreements)

        print("\nWhen heuristic leaves MONSTER, NN prefers:")
        monster_dis = dis_df[dis_df['heuristic_type'] == 0]
        if len(monster_dis) > 0:
            for t, name in [(0, 'Monster'), (0.5, 'Weapon'), (1.0, 'Potion')]:
                count = (monster_dis['nn_type'] == t).sum()
                print(f"  {name}: {count} ({100*count/len(monster_dis):.1f}%)")

        print("\nWhen heuristic leaves WEAPON, NN prefers:")
        weapon_dis = dis_df[dis_df['heuristic_type'] == 0.5]
        if len(weapon_dis) > 0:
            for t, name in [(0, 'Monster'), (0.5, 'Weapon'), (1.0, 'Potion')]:
                count = (weapon_dis['nn_type'] == t).sum()
                print(f"  {name}: {count} ({100*count/len(weapon_dis):.1f}%)")

        print("\nWhen heuristic leaves POTION, NN prefers:")
        potion_dis = dis_df[dis_df['heuristic_type'] == 1.0]
        if len(potion_dis) > 0:
            for t, name in [(0, 'Monster'), (0.5, 'Weapon'), (1.0, 'Potion')]:
                count = (potion_dis['nn_type'] == t).sum()
                print(f"  {name}: {count} ({100*count/len(potion_dis):.1f}%)")


if __name__ == '__main__':
    analyze_predictions()
