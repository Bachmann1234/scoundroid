# Scoundroid Rules

A single-player roguelike card game for Android, based on Scoundrel by Zach Gage and Kurt Bieg.

## Overview

You are a dungeon crawler with 20 health, fighting your way through a deck of monsters using weapons and potions. Survive the entire dungeon to win.

## Deck Setup

Start with a standard 52-card deck and remove:
- All Jokers
- Red face cards (J/Q/K of Hearts and Diamonds)
- Red Aces (Ace of Hearts and Ace of Diamonds)

This leaves 44 cards:
- **26 Monsters**: All Clubs and Spades (2-10, J, Q, K, A)
- **9 Weapons**: Diamonds 2-10
- **9 Potions**: Hearts 2-10

Shuffle these cards to form the **Dungeon** deck.

## Card Types

### Monsters (Clubs and Spades)
Monsters deal damage equal to their rank:
- Number cards: face value (2-10)
- Jack: 11
- Queen: 12
- King: 13
- Ace: 14

### Weapons (Diamonds)
Weapons reduce incoming monster damage by their value. When you pick up a weapon, you must equip it, discarding any previously equipped weapon.

### Potions (Hearts)
Potions restore health equal to their value. Health cannot exceed 20. You may only use one potion per turn—if you take a second potion in the same turn, it is discarded with no effect.

## Gameplay

### The Room
Each turn, draw cards until you have 4 face-up cards in front of you. This is your current **Room**.

### Avoiding a Room
You may choose to skip a room by placing all 4 cards at the bottom of the dungeon deck. However, you cannot skip two rooms in a row.

### Processing a Room
If you don't skip, you must resolve exactly 3 of the 4 cards. The fourth card remains face-up and becomes part of your next room.

For each card you resolve:
- **Weapon**: Equip it (discarding your old weapon if any)
- **Potion**: Heal for its value (max 20 health, one per turn)
- **Monster**: Fight it (see Combat)

## Combat

When fighting a monster, you have two options:

### Barehanded
Take the monster's full value as damage.

### With Weapon
Subtract your weapon's value from the monster's value. Take any remainder as damage (minimum 0).

**Example**: Fighting a Jack (11) with a 5 weapon: 11 - 5 = 6 damage taken.

### Weapon Degradation
Once you use a weapon against a monster, that weapon can only be used against monsters of **equal or lesser value** than the last monster it defeated.

**Example**:
1. You have a 7 weapon and fight a Queen (12) → weapon can now only hit monsters ≤ 12
2. You then fight a 5 → weapon can now only hit monsters ≤ 5
3. A Jack (11) appears → you must fight it barehanded (11 > 5)

The weapon stays equipped and can still be used against weaker monsters.

## Game End

The game ends when either:
- Your health reaches 0 (you lose)
- You clear the entire dungeon (you win)

## Scoring

**If you win:**
- Score = remaining health
- Special case: if you have exactly 20 health and your last card was a potion, score = 20 + that potion's value

**If you lose:**
- Find all remaining monsters in the dungeon
- Score = current health minus the sum of all remaining monster values (will be negative)

## Credits

Game design by Zach Gage and Kurt Bieg. Original game: [Scoundrel](http://www.stfj.net/scoundrel/)

This is a fan implementation for Android.
