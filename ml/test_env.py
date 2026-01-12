#!/usr/bin/env python3
"""Test the environment with different policies."""

from scoundrel_env import ScoundrelEnv, CardType


def random_policy(env):
    """Random action selection."""
    import random
    wins = 0
    for seed in range(10000):
        env = ScoundrelEnv(seed=seed)
        env.reset()
        while not env.done:
            action = random.choice(env.get_valid_actions())
            env.step(action)
        if env.won:
            wins += 1
    return wins


def simple_heuristic_policy(env):
    """Simple heuristic: leave biggest monster, prefer not skipping."""
    wins = 0
    for seed in range(10000):
        env = ScoundrelEnv(seed=seed)
        env.reset()
        while not env.done:
            # Sort room for consistent indexing
            room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))

            # Find biggest monster to leave
            best_leave = 0
            best_score = -1
            for i, card in enumerate(room):
                if card.card_type == CardType.MONSTER:
                    # Higher monster = more likely to leave
                    score = card.value
                else:
                    # Prefer to process weapons/potions
                    score = -card.value

                if score > best_score:
                    best_score = score
                    best_leave = i

            # Action: 1-4 for leave card 0-3
            action = best_leave + 1
            env.step(action)

        if env.won:
            wins += 1
    return wins


def leave_biggest_monster_policy(env):
    """Always leave the biggest monster."""
    wins = 0
    for seed in range(10000):
        env = ScoundrelEnv(seed=seed)
        env.reset()
        while not env.done:
            room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))

            # Find biggest monster
            biggest_monster_idx = None
            biggest_value = -1
            for i, card in enumerate(room):
                if card.card_type == CardType.MONSTER and card.value > biggest_value:
                    biggest_value = card.value
                    biggest_monster_idx = i

            if biggest_monster_idx is not None:
                action = biggest_monster_idx + 1
            else:
                # No monster, leave first card
                action = 1

            env.step(action)

        if env.won:
            wins += 1
    return wins


def smart_heuristic_policy():
    """Smarter heuristic mimicking the Kotlin bot."""
    wins = 0
    for seed in range(10000):
        env = ScoundrelEnv(seed=seed)
        env.reset()

        while not env.done:
            room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))

            monsters = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.MONSTER]
            weapons = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.WEAPON]
            potions = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.POTION]

            # Score each card for leaving
            scores = []
            for i, card in enumerate(room):
                score = 0.0

                if card.card_type == CardType.MONSTER:
                    # Leave big monsters (to fight later with weapon)
                    # But not if we have no weapon and low health
                    if env.weapon:
                        # If we can defeat it with weapon, don't leave
                        if env.weapon.can_defeat(card.value):
                            score = card.value * 0.5  # Less priority to leave
                        else:
                            score = card.value * 2  # Strongly leave if can't defeat
                    else:
                        # No weapon - leave big monsters we can't handle
                        if card.value > env.health:
                            score = card.value * 3  # Very high priority
                        else:
                            score = card.value * 0.3

                elif card.card_type == CardType.WEAPON:
                    # Leave weapons we don't need
                    if env.weapon and env.weapon.value >= card.value:
                        score = 10 + card.value  # Already have better
                    else:
                        score = -card.value  # Want to equip good weapons

                else:  # Potion
                    # Leave potions if near full health
                    if env.health >= 18:
                        score = 20 + card.value  # Don't need it
                    else:
                        score = -card.value  # Want to heal

                scores.append((i, score))

            # Leave the card with highest score
            best_leave = max(scores, key=lambda x: x[1])[0]
            action = best_leave + 1

            env.step(action)

        if env.won:
            wins += 1

    return wins


if __name__ == '__main__':
    print("Testing environment with different policies (10000 games each)...")
    print()

    print("Random policy:", random_policy(None), "wins")
    print("Leave biggest monster:", leave_biggest_monster_policy(None), "wins")
    print("Smart heuristic:", smart_heuristic_policy(), "wins")
