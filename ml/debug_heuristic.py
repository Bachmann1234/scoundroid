#!/usr/bin/env python3
"""Debug the Python HeuristicPlayer."""

from scoundrel_env import ScoundrelEnv, CardType
from heuristic_player import HeuristicPlayer, WEAPON_PRESERVATION_THRESHOLD


def trace_heuristic_game(seed):
    """Trace the heuristic player's game."""
    env = ScoundrelEnv(seed=seed)
    env.reset()

    print(f"=== Heuristic Player Trace: Seed {seed} ===")
    print(f"Initial room: {env.room}")
    print(f"Health: {env.health}")

    player = HeuristicPlayer()
    step = 0

    while not env.done:
        room = env.room.copy()
        print(f"\n--- Step {step} ---")
        print(f"Health: {env.health}, Deck: {len(env.deck)}")
        print(f"Weapon: {env.weapon}")
        print(f"Room: {room}")
        print(f"Last skipped: {env.last_room_skipped}")

        if len(room) == 4:
            # Check skip decision
            should_skip = player._should_skip_room(env, room)
            print(f"Should skip: {should_skip}")

            if not env.last_room_skipped and should_skip:
                print("ACTION: SKIP")
                env.step(0)
            else:
                # Get card to leave
                card_idx = player._choose_card_to_leave(env, room)
                card_to_leave = room[card_idx]

                # Calculate sorted index for step()
                sorted_room = sorted(room, key=lambda c: (c.card_type.value, c.value))
                sorted_idx = sorted_room.index(card_to_leave)

                print(f"Leaving: {card_to_leave} (room idx {card_idx}, sorted idx {sorted_idx})")

                # Show what we're processing
                cards_to_process = [c for i, c in enumerate(room) if i != card_idx]
                net_damage = player._simulate_net_damage(env, cards_to_process)
                print(f"Processing: {cards_to_process}")
                print(f"Estimated net damage: {net_damage}")

                env.step(sorted_idx + 1)

        step += 1
        if step > 20:
            print("Breaking - too many steps")
            break

    print(f"\n=== Result ===")
    print(f"Won: {env.won}, Health: {env.health}")
    return env.won


def analyze_losses(num_seeds=1000):
    """Analyze why we're losing games."""
    player = HeuristicPlayer()

    death_causes = {
        'no_weapon': 0,
        'degraded_weapon': 0,
        'overwhelmed': 0,
        'unknown': 0,
    }

    for seed in range(num_seeds):
        env = ScoundrelEnv(seed=seed)
        env.reset()

        # Track state when dying
        last_state = {
            'health': env.health,
            'weapon': env.weapon,
            'room': env.room.copy(),
        }

        try:
            won = player.play_game(env)

            if not won:
                # Analyze the death
                if last_state['weapon'] is None:
                    death_causes['no_weapon'] += 1
                elif last_state['weapon'].max_monster is not None and last_state['weapon'].max_monster < 5:
                    death_causes['degraded_weapon'] += 1
                else:
                    death_causes['overwhelmed'] += 1
        except Exception as e:
            death_causes['unknown'] += 1
            print(f"Error on seed {seed}: {e}")

    print(f"\n=== Death Analysis ({num_seeds} games) ===")
    for cause, count in death_causes.items():
        print(f"  {cause}: {count} ({100*count/num_seeds:.1f}%)")


def test_skip_logic():
    """Test the skip room logic."""
    print("\n=== Testing Skip Logic ===")

    # Create a scenario where skipping should help
    env = ScoundrelEnv(seed=0)
    env.reset()

    player = HeuristicPlayer()

    print(f"Initial room: {env.room}")
    print(f"Health: {env.health}")
    print(f"Last skipped: {env.last_room_skipped}")

    # Check if we should skip
    should_skip = player._should_skip_room(env, env.room)
    print(f"Should skip: {should_skip}")

    if not env.last_room_skipped:
        print("\nTrying skip action...")
        env.step(0)  # Skip
        print(f"After skip - Room: {env.room}")
        print(f"Last skipped: {env.last_room_skipped}")


if __name__ == '__main__':
    # First trace a single game
    trace_heuristic_game(1)

    print("\n" + "="*50 + "\n")

    # Analyze many losses
    analyze_losses(100)

    print("\n" + "="*50 + "\n")

    test_skip_logic()
