#!/usr/bin/env python3
"""
Compare Python and Kotlin environments by tracing the same game.
This helps find discrepancies in the implementations.
"""

import random
from scoundrel_env import ScoundrelEnv, CardType


def trace_python_game(seed):
    """Trace a game in Python and print every step."""
    # Use the same RNG as Kotlin's Random(seed)
    # But Python's random is different from Kotlin's!
    env = ScoundrelEnv(seed=seed)
    env.reset()

    print(f"=== Python Game: Seed {seed} ===")
    print(f"Initial deck (first 8 cards from shuffled):")

    # Show initial state
    deck_preview = env.deck[:8]
    for i, card in enumerate(deck_preview):
        print(f"  {i}: {card}")

    print(f"\nRoom: {env.room}")
    print(f"Health: {env.health}")

    step = 0
    while not env.done:
        room = env.room.copy()
        print(f"\n--- Step {step} ---")
        print(f"Room (raw): {room}")
        print(f"Health: {env.health}, Weapon: {env.weapon}")

        # Simple policy: leave biggest monster
        sorted_room = sorted(room, key=lambda c: (c.card_type.value, c.value))
        monsters = [(i, c) for i, c in enumerate(sorted_room) if c.card_type == CardType.MONSTER]

        if monsters:
            best_idx = max(monsters, key=lambda x: x[1].value)[0]
        else:
            best_idx = 0

        action = best_idx + 1
        card_left = sorted_room[best_idx]
        print(f"Leave: {card_left} (sorted idx {best_idx}, action {action})")

        obs, reward, done, _ = env.step(action)
        print(f"After step: health={env.health}, done={done}")

        step += 1
        if step > 15:
            print("Breaking - too many steps")
            break

    print(f"\n=== Result ===")
    print(f"Won: {env.won}, Health: {env.health}")
    return env.won


def compare_rng():
    """Check if Python random matches Kotlin random for same seeds."""
    print("=== Comparing RNG ===")

    # Test seed 1
    py_rng = random.Random(1)
    print(f"Python Random(1) first 5 floats:")
    for _ in range(5):
        print(f"  {py_rng.random():.10f}")

    # In Kotlin, kotlin.random.Random(1) produces different values
    print("\nNote: Python and Kotlin use different RNG algorithms!")
    print("Same seed will produce different card orders.")


def test_simple_deck():
    """Test with a manually defined deck to verify game logic."""
    print("\n=== Testing with simple deck ===")

    # Create environment with known deck
    env = ScoundrelEnv()

    # Manually set up a simple test deck
    from scoundrel_env import Card, WeaponState

    # Simple 8-card deck: should be winnable
    test_deck = [
        Card(CardType.WEAPON, 10),   # Best weapon first
        Card(CardType.MONSTER, 5),   # Easy monster
        Card(CardType.POTION, 5),    # Healing
        Card(CardType.MONSTER, 3),   # Easy monster
        Card(CardType.MONSTER, 2),   # Easy monster
        Card(CardType.POTION, 3),    # More healing
        Card(CardType.MONSTER, 4),   # Easy monster
        Card(CardType.MONSTER, 6),   # Moderate monster
    ]

    # Override the deck
    env.deck = test_deck.copy()
    env.room = []
    env.health = 20
    env.weapon = None
    env.last_room_skipped = False
    env.used_potion_this_turn = False
    env.done = False
    env.won = False

    # Draw initial room
    env._draw_room()

    print(f"Setup complete. Deck: {len(env.deck)}, Room: {len(env.room)}")
    print(f"Room: {env.room}")

    # Play the game
    step = 0
    while not env.done:
        room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))
        print(f"\nStep {step}: Room={room}, Health={env.health}, Weapon={env.weapon}")

        # Strategy: leave biggest monster, process rest
        monsters = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.MONSTER]
        if monsters:
            leave_idx = max(monsters, key=lambda x: x[1].value)[0]
        else:
            leave_idx = 0

        action = leave_idx + 1
        print(f"  Action {action}: leave {room[leave_idx]}")

        obs, reward, done, _ = env.step(action)
        print(f"  After: health={env.health}, done={done}, reward={reward}")

        step += 1
        if step > 10:
            break

    print(f"\nResult: won={env.won}, health={env.health}")
    return env.won


if __name__ == '__main__':
    compare_rng()
    print()

    # Test with simple deck first
    won = test_simple_deck()
    print(f"\nSimple deck test: {'PASS' if won else 'FAIL'}")

    if not won:
        print("\nBug detected in simple deck test!")
    else:
        # Try with random seeds
        print("\n\nTesting random seeds...")
        trace_python_game(1)
