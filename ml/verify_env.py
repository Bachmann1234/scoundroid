#!/usr/bin/env python3
"""Verify the Python environment can produce wins with controlled decks."""

from scoundrel_env import ScoundrelEnv, Card, CardType, WeaponState


def test_easy_win():
    """Test with an extremely easy deck that should always win."""
    env = ScoundrelEnv()

    # Easy deck: weapon first, then small monsters
    easy_deck = [
        Card(CardType.WEAPON, 10),  # Best weapon
        Card(CardType.MONSTER, 2),
        Card(CardType.MONSTER, 2),
        Card(CardType.MONSTER, 3),
        Card(CardType.MONSTER, 3),
        Card(CardType.MONSTER, 4),
        Card(CardType.MONSTER, 4),
        Card(CardType.POTION, 10),
    ]

    # Override deck
    env.deck = easy_deck.copy()
    env.room = []
    env.health = 20
    env.weapon = None
    env.last_room_skipped = False
    env.used_potion_this_turn = False
    env.done = False
    env.won = False
    env._draw_room()

    print("=== Easy Win Test ===")
    print(f"Room: {env.room}")

    # Simple strategy: always leave first monster
    step = 0
    while not env.done and step < 10:
        room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))
        print(f"Step {step}: room={room}, health={env.health}, weapon={env.weapon}")

        # Leave biggest monster
        monsters = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.MONSTER]
        if monsters:
            leave_idx = max(monsters, key=lambda x: x[1].value)[0]
        else:
            leave_idx = 0

        env.step(leave_idx + 1)
        step += 1

    print(f"\nResult: won={env.won}, health={env.health}")
    return env.won


def test_all_potions():
    """Test with mostly potions - should definitely win."""
    env = ScoundrelEnv()

    # Mostly potions, few small monsters
    easy_deck = [
        Card(CardType.POTION, 10),
        Card(CardType.POTION, 9),
        Card(CardType.POTION, 8),
        Card(CardType.MONSTER, 2),
        Card(CardType.MONSTER, 3),
        Card(CardType.WEAPON, 10),
        Card(CardType.MONSTER, 4),
        Card(CardType.POTION, 7),
    ]

    env.deck = easy_deck.copy()
    env.room = []
    env.health = 20
    env.weapon = None
    env.last_room_skipped = False
    env.used_potion_this_turn = False
    env.done = False
    env.won = False
    env._draw_room()

    print("\n=== All Potions Test ===")
    print(f"Room: {env.room}")

    step = 0
    while not env.done and step < 10:
        room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))
        print(f"Step {step}: room={room}, health={env.health}, weapon={env.weapon}")

        # Leave biggest monster
        monsters = [(i, c) for i, c in enumerate(room) if c.card_type == CardType.MONSTER]
        if monsters:
            leave_idx = max(monsters, key=lambda x: x[1].value)[0]
        else:
            leave_idx = 0

        env.step(leave_idx + 1)
        step += 1

    print(f"\nResult: won={env.won}, health={env.health}")
    return env.won


def test_weapon_degradation():
    """Test weapon degradation mechanic."""
    env = ScoundrelEnv()

    # Weapon then big monster then small monsters
    deck = [
        Card(CardType.WEAPON, 10),
        Card(CardType.MONSTER, 14),  # Ace - will degrade weapon to 14
        Card(CardType.MONSTER, 13),  # King - still beatable (13 <= 14)
        Card(CardType.MONSTER, 2),   # Easy
        Card(CardType.MONSTER, 3),
        Card(CardType.MONSTER, 4),
        Card(CardType.POTION, 5),
        Card(CardType.POTION, 6),
    ]

    env.deck = deck.copy()
    env.room = []
    env.health = 20
    env.weapon = None
    env.last_room_skipped = False
    env.used_potion_this_turn = False
    env.done = False
    env.won = False
    env._draw_room()

    print("\n=== Weapon Degradation Test ===")
    print(f"Room: {env.room}")

    step = 0
    while not env.done and step < 10:
        room = sorted(env.room, key=lambda c: (c.card_type.value, c.value))
        print(f"Step {step}: room={room}, health={env.health}, weapon={env.weapon}")

        # Leave smallest card to maximize processing
        leave_idx = 0
        env.step(leave_idx + 1)
        step += 1

    print(f"\nResult: won={env.won}, health={env.health}")
    return env.won


if __name__ == '__main__':
    results = []
    results.append(("Easy Win", test_easy_win()))
    results.append(("All Potions", test_all_potions()))
    results.append(("Weapon Degradation", test_weapon_degradation()))

    print("\n=== Summary ===")
    for name, won in results:
        status = "PASS" if won else "FAIL"
        print(f"  {name}: {status}")

    if all(won for _, won in results):
        print("\nAll tests passed! Environment works correctly.")
    else:
        print("\nSome tests failed - environment has bugs!")
