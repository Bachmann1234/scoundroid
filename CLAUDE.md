# Claude Instructions for Scoundroid

This file provides context and guidelines for working on the Scoundroid Android app.

## Project Context

**Scoundroid** is a personal Android app implementing the card game "Scoundrel" by Zach Gage and Kurt Bieg. This is being developed specifically for a Pixel 10 Pro Fold running the latest Android version.

**Original Game Rules**: `docs/Scoundrel.pdf`

## Project Planning

All planning documents are in the `plans/` directory:
- `plans/04-current-state.md` - **START HERE** for current status and next steps
- `plans/03-session-guide.md` - Session-by-session roadmap
- `plans/01-task-breakdown.md` - Detailed task checklist
- `plans/02-technical-architecture.md` - Technical design and architecture decisions

**Always check `plans/04-current-state.md` when starting work to understand where we are in development.**

## Test-Driven Development (TDD)

**CRITICAL**: This project uses a test-first approach. See [`plans/05-testing-strategy.md`](plans/05-testing-strategy.md) for comprehensive testing guidelines.

### TDD Workflow

For every feature (except UI components):

1. **Write failing test FIRST** (RED)
   ```kotlin
   @Test
   fun `feature should behave correctly`() {
       // Test the behavior that doesn't exist yet
   }
   ```

2. **Run test** - Verify it fails

3. **Write minimum code** to make it pass (GREEN)

4. **Run test** - Verify it passes

5. **Refactor** - Improve code while keeping tests green

6. **Repeat** for next test case

### Testing Requirements

- **Phase 1 & 2**: 100% coverage (game logic is critical)
- **Phase 3**: >80% coverage (ViewModel logic)
- **Phase 4**: >90% coverage (persistence)
- All tests must pass before moving to next phase
- All tests must pass before committing
- Never skip writing tests
- Never commit failing tests

## Critical Game Rules

These rules MUST be implemented correctly:

### Deck Composition (44 cards)
- Remove all Jokers, red face cards (J♥, Q♥, K♥, J♦, Q♦, K♦), and red Aces (A♥, A♦)
- Remaining: 26 Monsters (Clubs & Spades), 9 Weapons (Diamonds), 9 Potions (Hearts)

### Card Types & Values
- **Monsters** (♣ ♠): Damage = rank value (2-10 = face value, J=11, Q=12, K=13, A=14)
- **Weapons** (♦): Reduce monster damage by weapon value
- **Potions** (♥): Restore health by value (max 20, only 1 per turn)

### Weapon Degradation (Most Complex Rule)
Once a weapon is used on a monster, it can ONLY be used on monsters with value ≤ the last monster it defeated.

Example:
- Weapon: 5♦
- Defeat Queen (12) → weapon.maxMonsterValue = 12
- Can use on any monster ≤ 12
- Defeat 6 → weapon.maxMonsterValue = 6
- Can NOW only use on monsters ≤ 6
- Must fight 7+ barehanded (weapon still equipped, just can't use it)

### Room Mechanics
- Draw 4 cards to form a Room
- Choose to avoid Room (all 4 to bottom of deck) OR process Room
- Cannot avoid 2 Rooms in a row
- Process Room: choose 3 of 4 cards, leave 4th for next Room

### Scoring
- **Win**: Survive entire dungeon → score = remaining health (or health + last potion value if health = 20)
- **Lose**: Health reaches 0 → score = current health - sum of remaining monsters in deck (negative)

## Technical Architecture

### Stack
- **Language**: Kotlin
- **UI**: Jetpack Compose with Material3
- **Architecture**: MVI (Model-View-Intent) pattern
- **Persistence**: Room Database
- **Min/Target SDK**: 36

### Code Organization
```
dev.mattbachmann.scoundroid/
├── data/model/        # Card, GameState, Weapon, etc.
├── data/repository/   # Data layer
├── domain/usecase/    # Business logic
├── ui/screen/         # Compose screens + ViewModels
├── ui/component/      # Reusable UI components
└── ui/theme/          # Material3 theming
```

### Key Architectural Decisions
- **MVI over MVVM**: Better for game state management with clear user intents
- **Single Activity**: Using Compose Navigation
- **Jetpack Compose**: Modern, declarative UI
- See `plans/02-technical-architecture.md` for full details

## Development Guidelines

### Code Style
- Use Kotlin idioms (data classes, sealed classes, extension functions)
- Prefer immutability (val over var, immutable collections)
- Use meaningful variable names (no abbreviations unless obvious)
- Keep functions small and focused

### Testing
- Write unit tests for all game logic (especially weapon degradation)
- Test edge cases (empty deck, max health, multiple potions)
- UI tests for critical user flows

### Compose Guidelines
- Use `remember` and state hoisting appropriately
- Keep Composables pure (no side effects in composition)
- Use `LaunchedEffect` for one-time effects
- Prefer `State<T>` over mutable state

### Commit Messages
When creating commits for this project:
- Be clear about what changed
- Reference game rules if implementing specific mechanics
- Example: "Implement weapon degradation logic per Scoundrel rules"

## Common Tasks

### Starting a New Session
1. Read `plans/04-current-state.md` for current status
2. Check `plans/03-session-guide.md` for recommended next tasks
3. Update `plans/04-current-state.md` when session is complete

### Implementing Game Logic (Test-First!)
1. **Write tests first** - Define expected behavior in tests
2. Reference original rules in `docs/Scoundrel.pdf` when writing tests
3. Check `plans/02-technical-architecture.md` for data model design
4. Implement minimum code to pass tests
5. Refactor while keeping tests green
6. **Especially critical for**: weapon degradation, scoring, combat

### Adding UI Components
- Follow Material3 design guidelines
- Consider both folded and unfolded screen states
- Keep card visuals large and touch-friendly

### Debugging Game State
- Log state transitions in ViewModel
- Use Compose preview for UI components
- Test on actual Pixel 10 Pro Fold for foldable behavior

## Device-Specific Considerations

**Target Device**: Pixel 10 Pro Fold
- Support both folded (cover screen) and unfolded (inner screen) modes
- Use `WindowSizeClass` for responsive layouts
- Handle fold/unfold state transitions gracefully
- Test on actual device when possible

## Important Notes

- This is a **personal project** - no need for Play Store optimization
- Prioritize functionality over broad device compatibility
- User preference matters - ask if multiple approaches are valid
- Keep it simple - avoid over-engineering

## Resources

- Original Scoundrel: http://www.stfj.net/scoundrel/
- Game rules PDF: `docs/Scoundrel.pdf`
- Planning docs: `plans/` directory
- Jetpack Compose: https://developer.android.com/jetpack/compose

## Quick Reference: Card Deck

```
Monsters (26):
- Clubs: 2♣ 3♣ 4♣ 5♣ 6♣ 7♣ 8♣ 9♣ 10♣ J♣ Q♣ K♣ A♣
- Spades: 2♠ 3♠ 4♠ 5♠ 6♠ 7♠ 8♠ 9♠ 10♠ J♠ Q♠ K♠ A♠

Weapons (9):
- Diamonds: 2♦ 3♦ 4♦ 5♦ 6♦ 7♦ 8♦ 9♦ 10♦

Potions (9):
- Hearts: 2♥ 3♥ 4♥ 5♥ 6♥ 7♥ 8♥ 9♥ 10♥
```

**Removed**: Jokers, J♥ Q♥ K♥ A♥, J♦ Q♦ K♦ A♦

## Session Progress Tracking

When completing a session:
1. Update `plans/04-current-state.md` with what was accomplished
2. Mark completed tasks in `plans/01-task-breakdown.md`
3. Note any important decisions or discoveries
4. Identify next session goals

---

**Last Updated**: 2026-01-05
**Current Phase**: Planning complete, ready for implementation
