"""
Scoundrel game environment for reinforcement learning.
Implements the game logic in Python for fast RL training.
"""

import random
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import numpy as np


class CardType(Enum):
    MONSTER = 0
    WEAPON = 1
    POTION = 2


@dataclass
class Card:
    card_type: CardType
    value: int  # 2-14 (J=11, Q=12, K=13, A=14)

    def __repr__(self):
        type_char = {CardType.MONSTER: 'M', CardType.WEAPON: 'W', CardType.POTION: 'P'}
        return f"{type_char[self.card_type]}{self.value}"


@dataclass
class WeaponState:
    value: int
    max_monster: Optional[int] = None  # None = fresh

    def can_defeat(self, monster_value: int) -> bool:
        if self.max_monster is None:
            return True
        return monster_value <= self.max_monster


class ScoundrelEnv:
    """
    Scoundrel game environment.

    Actions:
        0: Skip room (if allowed)
        1-4: Leave card at index 0-3

    Observation: 26-dimensional feature vector
    """

    MAX_HEALTH = 20
    ROOM_SIZE = 4

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.reset()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self.rng = random.Random(seed)

        # Create deck: 26 monsters, 9 weapons, 9 potions
        self.deck: List[Card] = []

        # Monsters: clubs and spades (2-14 each) = 26 cards
        for value in range(2, 15):  # 2-14
            self.deck.append(Card(CardType.MONSTER, value))
            self.deck.append(Card(CardType.MONSTER, value))

        # Weapons: diamonds 2-10 = 9 cards
        for value in range(2, 11):
            self.deck.append(Card(CardType.WEAPON, value))

        # Potions: hearts 2-10 = 9 cards
        for value in range(2, 11):
            self.deck.append(Card(CardType.POTION, value))

        self.rng.shuffle(self.deck)

        self.health = self.MAX_HEALTH
        self.weapon: Optional[WeaponState] = None
        self.room: List[Card] = []
        self.last_room_skipped = False
        self.used_potion_this_turn = False
        self.done = False
        self.won = False

        # Draw first room
        self._draw_room()

        return self._get_observation()

    def _draw_room(self):
        """Draw cards to fill the room to 4 cards."""
        while len(self.room) < self.ROOM_SIZE and self.deck:
            self.room.append(self.deck.pop(0))

    def _get_observation(self) -> np.ndarray:
        """Get the current state as a feature vector."""
        if len(self.room) != 4:
            # Pad with zeros if room not full (shouldn't happen in normal play)
            room = self.room + [Card(CardType.MONSTER, 0)] * (4 - len(self.room))
        else:
            room = self.room

        # Sort room for consistent ordering
        room = sorted(room, key=lambda c: (c.card_type.value, c.value))

        monsters = [c for c in room if c.card_type == CardType.MONSTER]
        weapons = [c for c in room if c.card_type == CardType.WEAPON]
        potions = [c for c in room if c.card_type == CardType.POTION]

        deck_monsters = sum(1 for c in self.deck if c.card_type == CardType.MONSTER)
        deck_weapons = sum(1 for c in self.deck if c.card_type == CardType.WEAPON)
        deck_potions = sum(1 for c in self.deck if c.card_type == CardType.POTION)

        features = [
            # Card encodings (normalized)
            room[0].card_type.value / 2.0, room[0].value / 14.0,
            room[1].card_type.value / 2.0, room[1].value / 14.0,
            room[2].card_type.value / 2.0, room[2].value / 14.0,
            room[3].card_type.value / 2.0, room[3].value / 14.0,
            # Health
            self.health / 20.0,
            self.health / 20.0,  # health_fraction (same)
            # Weapon
            1.0 if self.weapon else 0.0,
            (self.weapon.value if self.weapon else 0) / 10.0,
            (self.weapon.max_monster if self.weapon and self.weapon.max_monster else 15) / 15.0,
            1.0 if self.weapon and self.weapon.max_monster is None else 0.0,
            # Deck
            len(self.deck) / 44.0,
            deck_monsters / 26.0,
            deck_weapons / 9.0,
            deck_potions / 9.0,
            # State
            1.0 if self.last_room_skipped else 0.0,
            0.0 if self.last_room_skipped else 1.0,  # can_skip
            # Derived
            sum(c.value for c in monsters) / 56.0,
            (max(c.value for c in monsters) if monsters else 0) / 14.0,
            sum(c.value for c in potions) / 30.0,
            (max(c.value for c in weapons) if weapons else 0) / 10.0,
            1.0 if weapons else 0.0,
            1.0 if potions else 0.0,
        ]

        return np.array(features, dtype=np.float32)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Take an action in the environment.

        Args:
            action: 0 = skip, 1-4 = leave card at index 0-3

        Returns:
            observation, reward, done, info
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}

        # Sort room for consistent indexing with observation
        self.room = sorted(self.room, key=lambda c: (c.card_type.value, c.value))

        if action == 0:  # Skip
            if self.last_room_skipped:
                # Invalid action - can't skip twice
                # Treat as leaving first card
                action = 1
            else:
                # Put room at bottom of deck
                self.deck.extend(self.room)
                self.room = []
                self.last_room_skipped = True
                self._draw_room()

                if self._check_game_over():
                    return self._get_observation(), self._get_final_reward(), True, {}

                return self._get_observation(), 0.0, False, {}

        # Leave card at index (action - 1)
        card_idx = action - 1
        if card_idx < 0 or card_idx >= len(self.room):
            card_idx = 0  # Default to first card

        card_to_leave = self.room[card_idx]
        cards_to_process = [c for i, c in enumerate(self.room) if i != card_idx]

        # Process the cards
        self._process_cards(cards_to_process)

        # Set up next room
        self.room = [card_to_leave]
        self.last_room_skipped = False
        self.used_potion_this_turn = False

        if self._check_game_over():
            return self._get_observation(), self._get_final_reward(), True, {}

        # Draw next room
        self._draw_room()

        if self._check_game_over():
            return self._get_observation(), self._get_final_reward(), True, {}

        return self._get_observation(), 0.0, False, {}

    def _process_cards(self, cards: List[Card]):
        """Process a list of cards (monsters, weapons, potions).

        Processing order matches the Kotlin HeuristicPlayer for optimal play:
        1. Equip best weapon first
        2. If health is critical, use a potion before combat
        3. Fight monsters (big first - preserves weapon utility)
        4. Use remaining potions after combat
        """
        weapons = sorted([c for c in cards if c.card_type == CardType.WEAPON],
                        key=lambda c: -c.value)
        monsters = sorted([c for c in cards if c.card_type == CardType.MONSTER],
                         key=lambda c: -c.value)
        potions = sorted([c for c in cards if c.card_type == CardType.POTION],
                        key=lambda c: -c.value)

        # 1. Equip best weapon using smart logic (considering degradation)
        # Thresholds from evolved genome
        EQUIP_FRESH_IF_DEGRADED_BELOW = 10
        ALWAYS_SWAP_TO_FRESH_IF_DEGRADED_BELOW = 8

        for weapon in weapons:
            should_equip = False

            if self.weapon is None:
                should_equip = True
            elif weapon.value > self.weapon.value:
                # New weapon has higher value - definitely equip
                should_equip = True
            elif self.weapon.max_monster is not None:
                # Current weapon is degraded - check swap conditions
                if self.weapon.max_monster < ALWAYS_SWAP_TO_FRESH_IF_DEGRADED_BELOW:
                    # Severely degraded - swap to ANY fresh weapon
                    should_equip = True
                elif self.weapon.max_monster < EQUIP_FRESH_IF_DEGRADED_BELOW:
                    # Moderately degraded - swap if fresh weapon can hit more
                    should_equip = weapon.value >= self.weapon.max_monster

            if should_equip:
                self.weapon = WeaponState(value=weapon.value)

        # 2. Estimate damage to decide if we need to heal first
        estimated_damage = 0
        for monster in monsters:
            if self.weapon and self.weapon.can_defeat(monster.value):
                estimated_damage += max(0, monster.value - self.weapon.value)
            else:
                estimated_damage += monster.value

        # If we would die without healing first, use a potion now
        needs_healing_first = self.health <= estimated_damage // 2 and potions
        potion_used_first = False

        if needs_healing_first and not self.used_potion_this_turn:
            best_potion = potions[0]  # Already sorted by value descending
            self.health = min(self.MAX_HEALTH, self.health + best_potion.value)
            self.used_potion_this_turn = True
            potion_used_first = True

        # 3. Fight monsters (big first)
        for monster in monsters:
            self._fight_monster(monster)
            if self.health <= 0:
                return

        # 4. Use remaining potions after combat
        for i, potion in enumerate(potions):
            # Skip first potion if already used
            if i == 0 and potion_used_first:
                continue
            if not self.used_potion_this_turn:
                self.health = min(self.MAX_HEALTH, self.health + potion.value)
                self.used_potion_this_turn = True

    # Weapon preservation threshold - only use fresh weapon on monsters >= this value
    # Evolved via genetic algorithm in Kotlin version
    WEAPON_PRESERVATION_THRESHOLD = 10

    def _fight_monster(self, monster: Card):
        """Fight a monster with intelligent weapon preservation.

        Key insight: Using a fresh weapon on small monsters degrades it permanently,
        making it useless against big monsters later. We preserve fresh weapons for
        monsters >= WEAPON_PRESERVATION_THRESHOLD (9).
        """
        if self.weapon and self.weapon.can_defeat(monster.value):
            # Check if we should preserve the weapon
            weapon_is_fresh = self.weapon.max_monster is None

            if weapon_is_fresh and monster.value < self.WEAPON_PRESERVATION_THRESHOLD:
                # Fight barehanded to preserve fresh weapon for bigger monsters
                self.health -= monster.value
            else:
                # Use weapon (either already degraded, or monster is big enough)
                damage = max(0, monster.value - self.weapon.value)
                self.health -= damage
                self.weapon.max_monster = monster.value
        else:
            # Fight barehanded (no weapon or can't hit this monster)
            self.health -= monster.value

    def _check_game_over(self) -> bool:
        """Check if the game is over."""
        if self.health <= 0:
            self.done = True
            self.won = False
            return True

        if not self.deck and len(self.room) <= 1:
            # Process any remaining card
            if self.room:
                self._process_cards(self.room)
                self.room = []

            self.done = True
            self.won = self.health > 0
            return True

        return False

    def _get_final_reward(self) -> float:
        """Get the final reward."""
        if self.won:
            return 1.0 + self.health / 20.0  # Bonus for remaining health
        else:
            return -1.0

    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions."""
        actions = [1, 2, 3, 4]  # Leave card 0-3
        if not self.last_room_skipped:
            actions = [0] + actions  # Can also skip
        return actions


def test_env():
    """Test the environment."""
    env = ScoundrelEnv(seed=42)
    obs = env.reset()

    print(f"Initial observation shape: {obs.shape}")
    print(f"Room: {env.room}")
    print(f"Health: {env.health}")

    total_reward = 0
    steps = 0

    while not env.done:
        # Random policy
        valid_actions = env.get_valid_actions()
        action = random.choice(valid_actions)

        obs, reward, done, info = env.step(action)
        total_reward += reward
        steps += 1

        if steps > 100:  # Safety limit
            break

    print(f"\nGame finished after {steps} steps")
    print(f"Won: {env.won}")
    print(f"Final health: {env.health}")
    print(f"Total reward: {total_reward}")


if __name__ == '__main__':
    test_env()
