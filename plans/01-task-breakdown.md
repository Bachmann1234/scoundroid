# Scoundroid - Task Breakdown

**Testing Approach**: Test-First Development (TDD) - Write tests before implementation for all game logic.
See [`05-testing-strategy.md`](05-testing-strategy.md) for detailed testing guidelines.

## Phase 1: Project Setup & Foundation - COMPLETE
**Goal**: Create the Android project and establish core architecture
**Coverage Target**: 100%

### Task 1.1: Project Initialization & Test Setup
- [x] Create new Android project with Jetpack Compose
- [x] Configure build.gradle with necessary dependencies
- [x] **Add testing dependencies (JUnit, Kotlin Test, Turbine, MockK)**
- [x] Set up test directory structure
- [x] Set up project structure (packages, modules)
- [x] Configure for Pixel 10 Pro Fold (foldable support)
- [x] Set up version control (.gitignore)
- [x] Create test utility classes and builders

### Task 1.2: Core Data Models (TDD)
- [x] **Write tests for Card class (value, type determination)**
- [x] Define Card data class (suit, rank, type)
- [x] **Verify all Card tests pass**
- [x] **Write tests for Suit, Rank, CardType enums**
- [x] Define enums (Suit, Rank, CardType)
- [x] **Verify all enum tests pass**
- [x] **Write tests for deck initialization (44 cards, correct composition)**
- [x] Implement deck initialization logic (remove red face cards and aces)
- [x] **Verify deck initialization tests pass**
- [x] **Write tests for shuffle (maintains count, changes order)**
- [x] Implement shuffle functionality
- [x] **Verify shuffle tests pass**
- [x] **Run full test suite - all tests must pass**

### Task 1.3: Game Logic Foundation (TDD)
- [x] **Write tests for room drawing (4 cards from deck)**
- [x] Implement room drawing logic (draw 4 cards)
- [x] **Verify room drawing tests pass**
- [x] **Write tests for room avoidance (can't avoid twice in a row)**
- [x] Implement room avoidance logic (can't avoid twice in a row)
- [x] **Verify room avoidance tests pass**
- [x] **Write tests for card selection (choose 3 of 4, keep 1)**
- [x] Implement card selection logic (choose 3 of 4)
- [x] **Verify card selection tests pass**
- [x] **Write tests for health tracking (damage, min/max bounds)**
- [x] Implement health tracking
- [x] **Verify health tracking tests pass**
- [x] **Run full test suite - all tests must pass**

## Phase 2: Combat & Game Mechanics (TDD) - COMPLETE
**Goal**: Implement all core game rules
**Coverage Target**: 100% (CRITICAL - Complex game logic)

### Task 2.1: Weapon System (TDD)
- [x] **Write tests for WeaponState data class**
- [x] Define WeaponState data class
- [x] **Write tests for weapon equipping (replaces old weapon)**
- [x] Implement weapon equipping logic
- [x] **Write comprehensive weapon degradation tests:**
  - [x] New weapon can defeat any monster
  - [x] Weapon tracks max monster value defeated
  - [x] Weapon can defeat monsters <= max value
  - [x] Weapon cannot defeat monsters > max value
  - [x] Defeating lower value monster downgrades weapon
  - [x] Weapon remains equipped even if unusable
- [x] Implement weapon degradation (track highest monster value defeated)
- [x] **Verify all weapon degradation tests pass**
- [x] **Write tests for damage calculation with weapon**
- [x] Implement damage calculation with weapon
- [x] **Verify damage calculation tests pass**
- [x] **Run full test suite - all tests must pass**

### Task 2.2: Combat System (TDD)
- [x] **Write tests for barehanded combat (full damage)**
- [x] Implement barehanded combat (full damage)
- [x] **Verify barehanded combat tests pass**
- [x] **Write tests for weapon combat:**
  - [x] Damage = max(0, monster - weapon)
  - [x] Monster placed on weapon stack
  - [x] Health reduced by calculated damage
- [x] Implement weapon combat (damage calculation)
- [x] **Verify weapon combat tests pass**
- [x] **Write tests for monster defeat and discard placement**
- [x] Handle monster defeat and card placement
- [x] **Write tests for monster stack on weapon**
- [x] Track monsters on equipped weapon
- [x] **Verify all tests pass**
- [x] **Run full test suite - all tests must pass**

### Task 2.3: Health & Potions (TDD)
- [x] **Write tests for health potion:**
  - [x] Adds value to health
  - [x] Cannot exceed 20
  - [x] Overflow ignored (health=18 + potion=5 = 20, not 23)
  - [x] Only first potion per turn is used
  - [x] Second potion discarded without effect
  - [x] Potion flag resets each turn
- [x] Implement health potion logic
- [x] **Verify potion tests pass**
- [x] **Write tests for max health enforcement**
- [x] Enforce max health of 20
- [x] **Write tests for one potion per turn limit**
- [x] Enforce one potion per turn limit
- [x] **Write tests for multiple potions in single room**
- [x] Handle multiple potions in single room
- [x] **Verify all potion tests pass**
- [x] **Run full test suite - all tests must pass**

### Task 2.4: Game End & Scoring (TDD)
- [x] **Write tests for game over detection (health <= 0)**
- [x] Detect game over (health = 0)
- [x] **Verify game over tests pass**
- [x] **Write tests for game win detection (dungeon empty)**
- [x] Detect game win (dungeon empty)
- [x] **Verify game win tests pass**
- [x] **Write tests for losing score calculation:**
  - [x] Negative score = current health - sum of remaining monsters
  - [x] Edge case: died exactly at 0 with no monsters left
- [x] Calculate losing score (negative monster sum)
- [x] **Verify losing score tests pass**
- [x] **Write tests for winning score calculation:**
  - [x] Score = remaining health
  - [x] Special case: health=20 AND last card was potion
- [x] Calculate winning score (remaining health)
- [x] **Write tests for special case (health=20 + last potion)**
- [x] Handle special case (health=20 + last potion)
- [x] **Verify all scoring tests pass**
- [x] **Run full test suite - all tests must pass**

## Phase 3: User Interface - COMPLETE
**Goal**: Build the game UI with Jetpack Compose
**Coverage Target**: >80% (ViewModel logic), UI tests for critical flows

### Task 3.1: Card UI Components
- [x] **Write Compose preview tests for card states**
- [x] Design card composable (visual representation)
- [x] Implement card type differentiation (colors, symbols)
- [x] Create card back design
- [x] **Verify card displays correctly in preview**
- [x] **Write tests for card animations (if stateful)**
- [x] Implement card animations (flip, move)

### Task 3.2: Game Board Layout
- [x] **Write Compose tests for layout rendering**
- [x] Design table layout (dungeon, room, discard, weapon area)
- [x] Implement room display (4 cards face up)
- [x] Implement equipped weapon display
- [x] Implement monster stack on weapon display
- [x] Implement discard pile indicator
- [x] **Verify layout renders correctly in different states**

### Task 3.3: Game Controls & HUD
- [x] Implement health display
- [x] Implement score display
- [x] Implement room avoidance button/gesture
- [x] Implement card selection mechanism
- [x] Implement combat choice dialog (barehanded vs weapon)
- [x] Show remaining cards in dungeon
- [x] **Write UI tests for critical interactions:**
  - [x] Card selection flow
  - [x] Room avoidance
  - [x] Combat choice dialog

### Task 3.4: Screens & Navigation
- [x] Create main menu screen
- [x] Create game screen
- [x] Create game over screen (with score)
- [x] Implement navigation between screens
- [x] Add New Game / Continue options
- [x] **Write navigation tests (screen transitions)**

### Task 3.5: Basic Help (Essential for playability)
- [x] Add help button to game screen (always visible)
- [x] Create basic rules reference screen with:
  - [x] Card values table (J=11, Q=12, K=13, A=14)
  - [x] Card types explanation
  - [x] Weapon degradation basics
  - [x] Scrollable rules summary
- [x] Show weapon current max value in UI
- [x] Damage preview before combat choice
- [ ] Link to view full rules (PDF) in menu

## Phase 4: State Management & Persistence - PARTIAL
**Goal**: Save game state and enable continue functionality
**Coverage Target**: >90%

### Task 4.1: ViewModel & State (TDD) - COMPLETE
- [x] **Write tests for GameViewModel:**
  - [x] Initial state is correct
  - [x] State updates on intents
  - [x] State flows emit correct values
  - [x] Error states handled
- [x] Create GameViewModel
- [x] **Verify ViewModel tests pass**
- [x] **Write tests for intent handling:**
  - [x] StartNewGame intent
  - [x] AvoidRoom intent
  - [x] SelectCard intent
  - [x] ChooseCombatMethod intent
- [x] Handle user actions (intents/events)
- [x] **Verify intent tests pass**
- [x] **Write tests for state flow updates**
- [x] Implement state flows for UI updates
- [x] **Verify state flow tests pass**
- [x] **Run full test suite - all tests must pass**

### Task 4.2: Data Persistence (TDD) - PARTIAL (High Scores Only)
- [x] **Write tests for game state serialization/deserialization**
- [x] Set up Room database
- [x] Create entities for game state
- [x] **Verify serialization tests pass**
- [ ] **Write tests for save game:**
  - [ ] State saved correctly
  - [ ] Can save mid-game
  - [ ] Overwrites previous save
- [ ] Implement save game functionality
- [ ] **Verify save tests pass**
- [ ] **Write tests for load game:**
  - [ ] Loads correct state
  - [ ] Handles missing save
  - [ ] Handles corrupted save
- [ ] Implement load game functionality
- [ ] **Verify load tests pass**
- [x] **Run full test suite - all tests must pass**
- [ ] Handle multiple save slots (optional)

### Task 4.3: Statistics Tracking (TDD) - NOT STARTED
- [ ] **Write tests for statistics:**
  - [ ] Games played increments
  - [ ] Wins/losses tracked correctly
  - [ ] High score updates
  - [ ] Low score updates
- [ ] Create statistics entity
- [ ] Track games played
- [ ] Track wins/losses
- [ ] Track high scores
- [ ] **Verify statistics tests pass**
- [ ] Create statistics screen
- [ ] **Run full test suite - all tests must pass**

## Phase 5: Polish & Enhancement - PARTIAL
**Goal**: Improve UX and add quality-of-life features

### Task 5.1: Visual Polish - PARTIAL
- [x] Add card animations (smooth transitions)
- [ ] Add damage number animations
- [x] Improve card graphics/design
- [ ] Add visual feedback for weapon degradation
- [ ] Implement dark mode

### Task 5.2: Help & Tutorial System - PARTIAL
- [x] **Create rules reference screen:**
  - [x] Card types and their functions
  - [x] Weapon degradation explanation with examples
  - [x] Room mechanics
  - [x] Scoring system
  - [x] Searchable/scrollable format
- [ ] **Add interactive tutorial:**
  - [ ] First-time user tutorial (optional)
  - [ ] Step-by-step walkthrough of first room
  - [ ] Weapon degradation demonstration
  - [ ] Practice mode (no score)
- [ ] **In-game contextual help:**
  - [x] Help button/icon always accessible during gameplay
  - [ ] Tooltips for complex actions (weapon degradation state)
  - [ ] "Why can't I use this weapon?" explanations
  - [x] Damage calculation preview
- [ ] **Quick reference overlay:**
  - [ ] Card values table (J=11, Q=12, K=13, A=14)
  - [ ] Current weapon degradation state display
  - [ ] Rules summary (swipe up from bottom?)
- [ ] Add rules PDF viewer (optional - view Scoundrel.pdf in-app)

### Task 5.3: Quality of Life - NOT STARTED
- [ ] Add undo functionality
- [x] Show damage preview before combat
- [ ] Add settings screen
- [ ] Implement haptic feedback

### Task 5.4: Foldable Optimization - COMPLETE
- [x] Test on folded screen
- [x] Test on unfolded screen
- [x] Optimize layout for different configurations
- [x] Handle fold/unfold state changes

### Task 5.5: Audio (Optional) - NOT STARTED
- [ ] Add card flip sounds
- [ ] Add combat sounds
- [ ] Add victory/defeat sounds
- [ ] Add background music (toggleable)
- [ ] Implement volume controls

## Phase 6: Integration Testing & Release - NOT STARTED
**Goal**: Ensure quality and prepare for personal use

### Task 6.1: Integration & E2E Testing
- [ ] **Write end-to-end test: Complete winning game**
- [ ] **Write end-to-end test: Complete losing game**
- [ ] **Write end-to-end test: Room avoidance flow**
- [ ] **Write end-to-end test: Weapon degradation scenario**
- [ ] **Verify all E2E tests pass**
- [x] Manual testing on Pixel 10 Pro Fold
- [ ] **Test all edge cases:**
  - [ ] Empty deck scenarios
  - [ ] Max health scenarios
  - [ ] All monsters remaining
  - [ ] All potions in a row
  - [ ] Weapon at exact limit
- [ ] **Verify 100% of game rules from Scoundrel.pdf are tested**
- [ ] **Run full test suite with coverage report**
- [ ] **Coverage must meet phase targets**

### Task 6.2: Bug Fixes & Optimization
- [x] Address any discovered bugs
- [ ] Performance optimization
- [ ] Memory leak checks
- [ ] Battery usage optimization

### Task 6.3: Documentation
- [ ] Code documentation
- [ ] README with setup instructions
- [x] Game rules reference in app

## Notes
- Since this is a personal app, we can skip Play Store optimization
- Focus on functionality and personal preferences over broad compatibility
- Can iterate on features based on actual gameplay experience
