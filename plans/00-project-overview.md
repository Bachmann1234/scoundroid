# Scoundroid - Project Overview

## Project Description
Android implementation of Scoundrel, a single-player rogue-like card game by Zach Gage and Kurt Bieg.

## Target Device
- Pixel 10 Pro Fold
- Latest Android version
- Should support both folded and unfolded screen modes

## Game Summary
Scoundrel is a solitaire card game where players navigate through a dungeon (deck of cards) making strategic decisions about which cards to face. The game uses a modified 44-card deck with three card types:
- **Monsters** (26 Clubs & Spades): Deal damage equal to their value (2-14)
- **Weapons** (9 Diamonds): Reduce monster damage but degrade with use
- **Health Potions** (9 Hearts): Restore health up to max of 20

Key mechanic: Weapons can only be used on monsters of equal or lesser value than the last monster they defeated.

## Technology Stack
- **Language**: Kotlin
- **UI Framework**: Jetpack Compose (modern, declarative UI)
- **Architecture**: MVVM or MVI pattern
- **Persistence**: Room database for game saves and statistics
- **Build System**: Gradle with Kotlin DSL

## Core Features (MVP)
1. Game setup (deck initialization)
2. Room drawing and display
3. Card selection and interaction
4. Combat system with weapon degradation logic
5. Health tracking
6. Score calculation
7. Game over detection
8. **Essential help system** (rules reference, card values, weapon state)

## Extended Features
1. Save/load game state
2. Statistics tracking (games played, high scores, win rate)
3. **Interactive tutorial and enhanced help**
4. Undo last action
5. Game history/replay
6. Animations and visual polish
7. Sound effects (optional)
8. Foldable screen optimization
9. Dark mode support

## UI Considerations
- Large, touch-friendly card representations
- Clear visual distinction between card types
- Easy-to-read health and score display
- Intuitive drag-and-drop or tap-to-select interaction
- Visual feedback for weapon degradation state
- Room avoidance mechanism (swipe gesture?)
- Combat preview (show damage calculation before committing)

## Project Status
- Status: Planning
- Started: 2026-01-05
