#!/usr/bin/env python3
"""
Port of the Kotlin HeuristicPlayer to Python.
This should achieve similar win rates to the Kotlin version.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from scoundrel_env import ScoundrelEnv, Card, CardType, WeaponState


# Evolved constants from genetic algorithm
WEAPON_PRESERVATION_THRESHOLD = 9
SKIP_DAMAGE_HEALTH_BUFFER = 5
SKIP_WITHOUT_WEAPON_FRACTION = 0.444
EQUIP_FRESH_IF_DEGRADED_BELOW = 10


class HeuristicPlayer:
    """Port of the Kotlin HeuristicPlayer."""

    def play_game(self, env: ScoundrelEnv) -> bool:
        """Play a complete game. Returns True if won."""
        env.reset()

        while not env.done:
            # Get unsorted room (as the game presents it)
            room = env.room.copy()

            if len(room) == 4:
                # Decide: skip or process?
                if not env.last_room_skipped and self._should_skip_room(env, room):
                    env.step(0)  # Skip
                else:
                    # Choose card to leave
                    card_to_leave_idx = self._choose_card_to_leave(env, room)
                    # Map to sorted index for step()
                    sorted_room = sorted(room, key=lambda c: (c.card_type.value, c.value))
                    card_to_leave = room[card_to_leave_idx]
                    sorted_idx = sorted_room.index(card_to_leave)
                    env.step(sorted_idx + 1)  # Action 1-4
            else:
                # End game - process remaining cards
                # The env handles this automatically
                break

        return env.won

    def _should_skip_room(self, env: ScoundrelEnv, room: List[Card]) -> bool:
        """Decide whether to skip the current room."""
        monsters = [c for c in room if c.card_type == CardType.MONSTER]
        if not monsters:
            return False

        # Calculate estimated damage
        card_to_leave = self._choose_card_to_leave(env, room)
        cards_to_process = [c for i, c in enumerate(room) if i != card_to_leave]
        net_damage = self._simulate_net_damage(env, cards_to_process)

        # Would this kill us?
        if net_damage >= env.health - SKIP_DAMAGE_HEALTH_BUFFER:
            return True

        # Check if we have weapon help
        has_weapon_in_room = any(c.card_type == CardType.WEAPON for c in room)
        current_weapon_useful = False
        if env.weapon:
            current_weapon_useful = any(env.weapon.can_defeat(m.value) for m in monsters)

        if not current_weapon_useful and not has_weapon_in_room:
            if net_damage > env.health * SKIP_WITHOUT_WEAPON_FRACTION:
                return True

        return False

    def _choose_card_to_leave(self, env: ScoundrelEnv, room: List[Card]) -> int:
        """Choose which card index to leave."""
        best_idx = 0
        best_score = float('inf')

        for i, candidate in enumerate(room):
            cards_to_process = [c for j, c in enumerate(room) if j != i]
            score = self._evaluate_leave_choice(env, candidate, cards_to_process)

            if score < best_score:
                best_score = score
                best_idx = i

        return best_idx

    def _simulate_net_damage(self, env: ScoundrelEnv, cards: List[Card]) -> int:
        """Simulate processing cards and return net damage."""
        monsters = sorted([c for c in cards if c.card_type == CardType.MONSTER],
                         key=lambda c: -c.value)  # Big first
        weapons = [c for c in cards if c.card_type == CardType.WEAPON]
        potions = [c for c in cards if c.card_type == CardType.POTION]

        # Determine weapon situation
        current_weapon = env.weapon
        best_new_weapon = max(weapons, key=lambda c: c.value) if weapons else None

        if best_new_weapon and self._should_equip_weapon(current_weapon, best_new_weapon):
            effective_weapon_value = best_new_weapon.value
            weapon_is_fresh = True
        elif current_weapon:
            effective_weapon_value = current_weapon.value
            weapon_is_fresh = current_weapon.max_monster is None
        else:
            effective_weapon_value = 0
            weapon_is_fresh = False

        # Calculate damage
        total_damage = 0
        weapon_max_monster = None if weapon_is_fresh else (current_weapon.max_monster if current_weapon else None)

        for monster in monsters:
            can_use_weapon = (effective_weapon_value > 0 and
                            (weapon_max_monster is None or monster.value <= weapon_max_monster))

            if can_use_weapon:
                should_use = not weapon_is_fresh or monster.value >= WEAPON_PRESERVATION_THRESHOLD

                if should_use:
                    damage = max(0, monster.value - effective_weapon_value)
                    total_damage += damage
                    weapon_max_monster = monster.value
                    weapon_is_fresh = False
                else:
                    total_damage += monster.value  # Barehanded to preserve
            else:
                total_damage += monster.value  # Barehanded

        # Calculate effective healing
        health_deficit = 20 - env.health + total_damage
        total_potion = sum(p.value for p in potions)
        effective_healing = min(total_potion, max(0, health_deficit))

        return total_damage - effective_healing

    def _evaluate_leave_choice(self, env: ScoundrelEnv, card_to_leave: Card,
                               cards_to_process: List[Card]) -> int:
        """Evaluate leaving a card. Lower is better."""
        net_damage = self._simulate_net_damage(env, cards_to_process)

        # Penalty for leaving the card
        if card_to_leave.card_type == CardType.MONSTER:
            leftover_penalty = card_to_leave.value
        elif card_to_leave.card_type == CardType.WEAPON:
            current_value = env.weapon.value if env.weapon else 0
            leftover_penalty = card_to_leave.value if card_to_leave.value > current_value else 0
        else:  # Potion
            leftover_penalty = 0

        return net_damage + leftover_penalty

    def _should_equip_weapon(self, current: Optional[WeaponState], new_weapon: Card) -> bool:
        """Decide whether to equip a new weapon."""
        if current is None:
            return True

        if new_weapon.value > current.value:
            return True

        # Current is degraded below threshold
        if current.max_monster is not None and current.max_monster < EQUIP_FRESH_IF_DEGRADED_BELOW:
            return new_weapon.value >= current.max_monster

        return False


def test_heuristic():
    """Test the heuristic player."""
    player = HeuristicPlayer()
    wins = 0

    for seed in range(10000):
        env = ScoundrelEnv(seed=seed)
        won = player.play_game(env)
        if won:
            wins += 1

        if seed % 1000 == 0 and seed > 0:
            print(f"  Seed {seed}: {wins} wins ({100*wins/seed:.2f}%)")

    print(f"\nFinal: {wins}/10000 wins ({100*wins/10000:.2f}%)")
    return wins


if __name__ == '__main__':
    print("Testing Python HeuristicPlayer...")
    test_heuristic()
