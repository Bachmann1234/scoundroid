#!/usr/bin/env python3
"""Find winning seeds to verify the Python environment can produce wins."""

from scoundrel_env import ScoundrelEnv, CardType
from heuristic_player import HeuristicPlayer
import time


def simple_smart_policy(env):
    """A simple but smarter policy that tries to win."""
    room = env.room.copy()
    sorted_room = sorted(room, key=lambda c: (c.card_type.value, c.value))

    monsters = [c for c in room if c.card_type == CardType.MONSTER]
    weapons = [c for c in room if c.card_type == CardType.WEAPON]
    potions = [c for c in room if c.card_type == CardType.POTION]

    # If we have no weapon and there's a weapon in the room, leave smallest monster
    if env.weapon is None and weapons:
        # Best to leave the biggest monster we can't fight
        if monsters:
            leave = max(monsters, key=lambda c: c.value)
            return sorted_room.index(leave) + 1
        else:
            return 1  # Leave first card

    # If we have a usable weapon, leave the biggest monster
    if env.weapon:
        max_monster = env.weapon.max_monster if env.weapon.max_monster else float('inf')
        beatable = [m for m in monsters if m.value <= max_monster]
        unbeatable = [m for m in monsters if m.value > max_monster]

        if unbeatable:
            # Leave an unbeatable monster
            leave = max(unbeatable, key=lambda c: c.value)
            return sorted_room.index(leave) + 1

    # Default: leave the biggest monster
    if monsters:
        leave = max(monsters, key=lambda c: c.value)
        return sorted_room.index(leave) + 1

    # No monsters - leave the worst card (lowest value potion or weapon we don't need)
    return 1


def search_winning_seeds_simple(num_seeds=100000):
    """Search for wins using simple smart policy."""
    print(f"Searching {num_seeds} seeds with simple smart policy...")
    wins = 0
    start = time.time()

    for seed in range(num_seeds):
        env = ScoundrelEnv(seed=seed)
        env.reset()

        step = 0
        while not env.done:
            if len(env.room) == 4:
                action = simple_smart_policy(env)
            else:
                action = 1
            env.step(action)
            step += 1
            if step > 30:
                break

        if env.won:
            wins += 1
            if wins <= 5:
                print(f"  Found win at seed {seed}!")

        if seed % 10000 == 0 and seed > 0:
            elapsed = time.time() - start
            rate = seed / elapsed
            print(f"  Progress: {seed} seeds, {wins} wins ({100*wins/seed:.3f}%), {rate:.0f} games/sec")

    elapsed = time.time() - start
    print(f"\nDone! {wins}/{num_seeds} wins ({100*wins/num_seeds:.3f}%)")
    print(f"Time: {elapsed:.1f}s ({num_seeds/elapsed:.0f} games/sec)")
    return wins


def search_with_skipping(num_seeds=100000):
    """Search using policy that can skip dangerous rooms."""
    print(f"\nSearching {num_seeds} seeds with skipping policy...")
    wins = 0
    start = time.time()

    for seed in range(num_seeds):
        env = ScoundrelEnv(seed=seed)
        env.reset()

        step = 0
        while not env.done:
            room = env.room.copy()
            sorted_room = sorted(room, key=lambda c: (c.card_type.value, c.value))

            if len(room) == 4:
                monsters = [c for c in room if c.card_type == CardType.MONSTER]
                weapons = [c for c in room if c.card_type == CardType.WEAPON]
                potions = [c for c in room if c.card_type == CardType.POTION]

                # Calculate potential damage
                total_monster = sum(m.value for m in monsters)
                weapon_reduction = 0
                if env.weapon:
                    beatable = [m for m in monsters
                               if env.weapon.max_monster is None or m.value <= env.weapon.max_monster]
                    weapon_reduction = len(beatable) * env.weapon.value
                    weapon_reduction = min(weapon_reduction, total_monster)

                # Also count new weapon in room
                best_weapon_in_room = max([w.value for w in weapons], default=0)
                if best_weapon_in_room > (env.weapon.value if env.weapon else 0):
                    weapon_reduction = sum(min(m.value, best_weapon_in_room) for m in monsters)

                net_damage = total_monster - weapon_reduction
                best_potion = max([p.value for p in potions], default=0)
                net_damage -= best_potion

                # Skip if dangerous
                if not env.last_room_skipped and net_damage >= env.health - 5:
                    env.step(0)  # Skip
                else:
                    action = simple_smart_policy(env)
                    env.step(action)
            else:
                env.step(1)

            step += 1
            if step > 30:
                break

        if env.won:
            wins += 1
            if wins <= 5:
                print(f"  Found win at seed {seed}!")

        if seed % 10000 == 0 and seed > 0:
            elapsed = time.time() - start
            print(f"  Progress: {seed} seeds, {wins} wins ({100*wins/seed:.3f}%)")

    elapsed = time.time() - start
    print(f"\nDone! {wins}/{num_seeds} wins ({100*wins/num_seeds:.3f}%)")
    return wins


def search_heuristic_player(num_seeds=100000):
    """Search using the HeuristicPlayer."""
    print(f"\nSearching {num_seeds} seeds with HeuristicPlayer...")
    player = HeuristicPlayer()
    wins = 0
    start = time.time()

    for seed in range(num_seeds):
        env = ScoundrelEnv(seed=seed)
        try:
            won = player.play_game(env)
            if won:
                wins += 1
                if wins <= 5:
                    print(f"  Found win at seed {seed}!")
        except Exception as e:
            pass  # Skip errors

        if seed % 10000 == 0 and seed > 0:
            elapsed = time.time() - start
            print(f"  Progress: {seed} seeds, {wins} wins ({100*wins/seed:.3f}%)")

    elapsed = time.time() - start
    print(f"\nDone! {wins}/{num_seeds} wins ({100*wins/num_seeds:.3f}%)")
    return wins


if __name__ == '__main__':
    # Try multiple strategies
    search_winning_seeds_simple(100000)
    search_with_skipping(100000)
    search_heuristic_player(100000)
