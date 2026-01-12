#!/usr/bin/env python3
"""Debug the environment by tracing a single game."""

from scoundrel_env import ScoundrelEnv, CardType


def trace_game(seed=42):
    """Play a game and print every step."""
    env = ScoundrelEnv(seed=seed)
    env.reset()

    print(f"=== Game with seed {seed} ===")
    print(f"Initial deck size: {len(env.deck)}")
    print(f"Initial health: {env.health}")

    step = 0
    while not env.done:
        room = env.room.copy()
        room_sorted = sorted(room, key=lambda c: (c.card_type.value, c.value))

        print(f"\n--- Step {step} ---")
        print(f"Health: {env.health}")
        print(f"Weapon: {env.weapon}")
        print(f"Deck remaining: {len(env.deck)}")
        print(f"Room: {room}")
        print(f"Room (sorted): {room_sorted}")
        print(f"Last room skipped: {env.last_room_skipped}")

        # Simple strategy: leave biggest monster
        monsters = [(i, c) for i, c in enumerate(room_sorted) if c.card_type == CardType.MONSTER]
        if monsters:
            best_idx = max(monsters, key=lambda x: x[1].value)[0]
        else:
            best_idx = 0

        action = best_idx + 1
        print(f"Action: leave card {best_idx} ({room_sorted[best_idx]})")

        obs, reward, done, _ = env.step(action)
        print(f"Reward: {reward}, Done: {done}")

        step += 1
        if step > 50:
            print("Breaking - too many steps")
            break

    print(f"\n=== Game Over ===")
    print(f"Won: {env.won}")
    print(f"Final health: {env.health}")
    print(f"Deck remaining: {len(env.deck)}")


def check_deck_composition():
    """Check the deck is set up correctly."""
    env = ScoundrelEnv(seed=0)
    env.reset()

    # Count cards by type
    all_cards = env.deck + env.room
    monsters = [c for c in all_cards if c.card_type == CardType.MONSTER]
    weapons = [c for c in all_cards if c.card_type == CardType.WEAPON]
    potions = [c for c in all_cards if c.card_type == CardType.POTION]

    print("Deck composition check:")
    print(f"  Monsters: {len(monsters)} (expected 26)")
    print(f"  Weapons: {len(weapons)} (expected 9)")
    print(f"  Potions: {len(potions)} (expected 9)")
    print(f"  Total: {len(all_cards)} (expected 44)")

    # Check monster values
    monster_values = sorted([c.value for c in monsters])
    print(f"  Monster values: {monster_values}")
    print(f"  Expected: 2x each of 2-14")


def trace_winning_game():
    """Try many seeds to find one that can win."""
    print("Searching for a winning game...")
    for seed in range(100000):
        env = ScoundrelEnv(seed=seed)
        env.reset()

        while not env.done:
            room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))
            monsters = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.MONSTER]
            if monsters:
                best_idx = max(monsters, key=lambda x: x[1].value)[0]
            else:
                best_idx = 0
            action = best_idx + 1
            env.step(action)

        if env.won:
            print(f"Found winning seed: {seed}")
            trace_game(seed)
            return seed

        if seed % 10000 == 0:
            print(f"  Tried {seed} seeds...")

    print("No winning seed found in 100000 tries")
    return None


if __name__ == '__main__':
    check_deck_composition()
    print()
    trace_game(42)
    print()
    trace_winning_game()
