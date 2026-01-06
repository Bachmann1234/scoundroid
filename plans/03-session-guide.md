# Scoundroid - Session Guide

This document helps us track progress and plan what to work on in each session.

## Current Status
- **Phase**: Not Started
- **Last Updated**: 2026-01-05
- **Next Session Goal**: Project setup and foundation

## Suggested Session Breakdown

### Session 1: Foundation (Recommended First Session)
**Goal**: Get a working Android project with basic structure and TDD workflow

**Testing**: Write tests FIRST, then implement - TDD approach

Tasks:
1. Create new Android Studio project with Jetpack Compose
2. Configure build.gradle with dependencies (including test dependencies)
3. Set up package structure (including test directories)
4. Set up test utilities and builders
5. **Write tests for Card class and enums**
6. Create core data models (Card, Suit, Rank, CardType)
7. **Verify all tests pass**
8. **Write tests for deck initialization**
9. Implement deck initialization (44 cards, remove red faces/aces)
10. **Verify deck tests pass**

**Deliverable**: A buildable Android app with valid deck creation AND passing test suite

**Estimated Complexity**: Low-Medium
**Prerequisites**: Android Studio installed, Pixel 10 Pro Fold for testing (optional for this session)
**Success Criteria**: `./gradlew test` runs and all tests pass

---

### Session 2: Game State & Logic
**Goal**: Implement core game state and turn mechanics

**Testing**: Continue TDD - tests before code

Tasks:
1. **Write tests for room drawing** (4 cards from deck)
2. Implement room drawing logic
3. **Verify tests pass**
4. **Write tests for room avoidance** (can't avoid twice)
5. Implement room avoidance logic
6. **Verify tests pass**
7. **Write tests for card selection** (choose 3 of 4)
8. Implement card selection tracking
9. **Verify tests pass**
10. **Run full test suite**

**Deliverable**: Game logic that can manage turns (no UI yet) with comprehensive tests

**Estimated Complexity**: Medium
**Prerequisites**: Session 1 complete, all Session 1 tests passing
**Success Criteria**: All tests pass, coverage >95%

---

### Session 3: Combat System
**Goal**: Implement all combat mechanics

**Testing**: CRITICAL - Weapon degradation is complex, must be thoroughly tested

Tasks:
1. **Write comprehensive weapon degradation tests** (see 05-testing-strategy.md)
2. Implement weapon equipping logic
3. Implement weapon degradation tracking
4. **Verify all weapon tests pass**
5. **Write tests for barehanded combat**
6. Implement barehanded combat
7. **Verify tests pass**
8. **Write tests for weapon combat** (damage calculation)
9. Implement weapon combat
10. **Verify tests pass**
11. **Write tests for health potion logic** (max 20, 1 per turn)
12. Implement health potion logic
13. **Verify tests pass**
14. **Run full test suite**

**Deliverable**: Complete combat system with exhaustive test coverage

**Estimated Complexity**: Medium-High (weapon degradation is tricky)
**Prerequisites**: Session 2 complete, all tests passing
**Success Criteria**: 100% coverage on combat logic, all edge cases tested

---

### Session 4: Basic UI
**Goal**: Create minimal playable UI with essential help

Tasks:
1. Create basic card composable
2. Create room display
3. Create health display
4. Create simple game screen layout
5. Wire up ViewModel to UI
6. **Add basic help system:**
   - Help button (always visible)
   - Card values reference
   - Weapon state display
   - Damage preview
7. Make it playable (even if ugly)

**Deliverable**: Functional game you can actually play and understand

**Estimated Complexity**: Medium
**Prerequisites**: Session 3 complete
**Why help now**: You need to understand rules to test the game!

---

### Session 5: UI Polish & Card Design
**Goal**: Make it look like a real card game

Tasks:
1. Design better card visuals
2. Implement card type differentiation (colors, symbols)
3. Improve layout and spacing
4. Add weapon area with monster stack
5. Add discard pile indicator
6. Test on actual device

**Deliverable**: Nice-looking game interface

**Estimated Complexity**: Medium
**Prerequisites**: Session 4 complete

---

### Session 6: Game Flow & Screens
**Goal**: Complete the game flow with all screens

Tasks:
1. Create main menu screen
2. Create game over screen with score
3. Implement navigation
4. Add new game / continue options
5. Implement scoring logic
6. Test complete game flow

**Deliverable**: Complete game from start to finish

**Estimated Complexity**: Low-Medium
**Prerequisites**: Session 5 complete

---

### Session 7: Save System
**Goal**: Persist game state

Tasks:
1. Set up Room database
2. Create entities and DAOs
3. Implement save game functionality
4. Implement load game functionality
5. Add auto-save feature
6. Test save/load thoroughly

**Deliverable**: Game state persists between sessions

**Estimated Complexity**: Medium
**Prerequisites**: Session 6 complete

---

### Session 8: Enhanced Help & QoL
**Goal**: Add comprehensive help system and quality of life features

Tasks:
1. Implement statistics tracking
2. Create statistics screen
3. **Enhance help system:**
   - Interactive tutorial
   - Contextual tooltips
   - Weapon degradation explanations
   - Quick reference overlay
4. Add undo functionality
5. Polish and bug fixes

**Deliverable**: Feature-complete game with excellent UX

**Estimated Complexity**: Medium
**Prerequisites**: Session 7 complete

---

### Session 9: Foldable Optimization
**Goal**: Optimize for Pixel 10 Pro Fold

Tasks:
1. Test on folded screen
2. Test on unfolded screen
3. Implement responsive layouts
4. Handle fold/unfold transitions
5. Optimize for both configurations
6. Final device testing

**Deliverable**: Optimized experience for foldable

**Estimated Complexity**: Medium
**Prerequisites**: Session 8 complete, physical device needed

---

### Session 10: Polish & Release
**Goal**: Final polish and personal "release"

Tasks:
1. Add animations
2. Add sound effects (optional)
3. Performance optimization
4. Final bug fixes
5. Code cleanup
6. Celebration!

**Deliverable**: Finished game ready for daily use

**Estimated Complexity**: Low-Medium
**Prerequisites**: Session 9 complete

---

## Notes for Resuming Work

When starting a new session:
1. Check this file for current status
2. Review the task breakdown (01-task-breakdown.md)
3. Check technical architecture if needed (02-technical-architecture.md)
4. **Review testing strategy (05-testing-strategy.md)**
5. **Run test suite to verify current state**
6. Update current status when session is complete

When completing a session:
1. **Run full test suite - all tests must pass**
2. **Check test coverage meets phase targets**
3. Update "Current Status" section above
4. Mark completed tasks in 01-task-breakdown.md
5. Note any decisions or discoveries
6. Identify any blockers for next session
7. **Commit with passing tests**

## TDD Workflow (Every Session)

**Red → Green → Refactor**

1. **Write Test** (RED)
   - Write a failing test for the feature
   - Run test, verify it fails

2. **Implement** (GREEN)
   - Write minimum code to pass
   - Run test, verify it passes

3. **Refactor**
   - Improve code quality
   - Keep tests passing

4. **Repeat** for next feature

**Never commit failing tests. Never move to next phase with incomplete tests.**

## Session Template

When starting a session, you can use this template:

```markdown
## Session [N]: [Date]
**Goal**: [What we want to accomplish]

**Tasks Completed**:
- [ ] Task 1
- [ ] Task 2

**Decisions Made**:
- Decision 1
- Decision 2

**Blockers/Issues**:
- Issue 1

**Next Session**:
- What to tackle next
```

## Quick Reference

**Project Name**: Scoundroid (Scoundrel + Android)
**Deck Size**: 44 cards (52 - 8 red faces/aces)
**Card Types**: 26 Monsters, 9 Weapons, 9 Potions
**Starting Health**: 20
**Max Health**: 20

**Critical Rules to Remember**:
1. Weapon degradation: Can only defeat monsters ≤ last defeated value
2. Room avoidance: Can't avoid twice in a row
3. Potions: Only 1 per turn, can't exceed 20 health
4. Room: Draw 4, choose 3, keep 1 for next room
