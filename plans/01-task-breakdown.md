# Scoundroid - Task Breakdown

**Testing Approach**: Test-First Development (TDD) - Write tests before implementation for all game logic.
See [`05-testing-strategy.md`](05-testing-strategy.md) for detailed testing guidelines.

## Phase 1: Project Setup & Foundation
**Goal**: Create the Android project and establish core architecture
**Coverage Target**: 100%

### Task 1.1: Project Initialization & Test Setup
- [ ] Create new Android project with Jetpack Compose
- [ ] Configure build.gradle with necessary dependencies
- [ ] **Add testing dependencies (JUnit, Kotlin Test, Turbine, MockK)**
- [ ] Set up test directory structure
- [ ] Set up project structure (packages, modules)
- [ ] Configure for Pixel 10 Pro Fold (foldable support)
- [ ] Set up version control (.gitignore)
- [ ] Create test utility classes and builders

### Task 1.2: Core Data Models (TDD)
- [ ] **Write tests for Card class (value, type determination)**
- [ ] Define Card data class (suit, rank, type)
- [ ] **Verify all Card tests pass**
- [ ] **Write tests for Suit, Rank, CardType enums**
- [ ] Define enums (Suit, Rank, CardType)
- [ ] **Verify all enum tests pass**
- [ ] **Write tests for deck initialization (44 cards, correct composition)**
- [ ] Implement deck initialization logic (remove red face cards and aces)
- [ ] **Verify deck initialization tests pass**
- [ ] **Write tests for shuffle (maintains count, changes order)**
- [ ] Implement shuffle functionality
- [ ] **Verify shuffle tests pass**
- [ ] **Run full test suite - all tests must pass**

### Task 1.3: Game Logic Foundation (TDD)
- [ ] **Write tests for room drawing (4 cards from deck)**
- [ ] Implement room drawing logic (draw 4 cards)
- [ ] **Verify room drawing tests pass**
- [ ] **Write tests for room avoidance (can't avoid twice in a row)**
- [ ] Implement room avoidance logic (can't avoid twice in a row)
- [ ] **Verify room avoidance tests pass**
- [ ] **Write tests for card selection (choose 3 of 4, keep 1)**
- [ ] Implement card selection logic (choose 3 of 4)
- [ ] **Verify card selection tests pass**
- [ ] **Write tests for health tracking (damage, min/max bounds)**
- [ ] Implement health tracking
- [ ] **Verify health tracking tests pass**
- [ ] **Run full test suite - all tests must pass**

## Phase 2: Combat & Game Mechanics (TDD)
**Goal**: Implement all core game rules
**Coverage Target**: 100% (CRITICAL - Complex game logic)

### Task 2.1: Weapon System (TDD)
- [ ] **Write tests for WeaponState data class**
- [ ] Define WeaponState data class
- [ ] **Write tests for weapon equipping (replaces old weapon)**
- [ ] Implement weapon equipping logic
- [ ] **Write comprehensive weapon degradation tests:**
  - [ ] New weapon can defeat any monster
  - [ ] Weapon tracks max monster value defeated
  - [ ] Weapon can defeat monsters <= max value
  - [ ] Weapon cannot defeat monsters > max value
  - [ ] Defeating lower value monster downgrades weapon
  - [ ] Weapon remains equipped even if unusable
- [ ] Implement weapon degradation (track highest monster value defeated)
- [ ] **Verify all weapon degradation tests pass**
- [ ] **Write tests for damage calculation with weapon**
- [ ] Implement damage calculation with weapon
- [ ] **Verify damage calculation tests pass**
- [ ] **Run full test suite - all tests must pass**

### Task 2.2: Combat System (TDD)
- [ ] **Write tests for barehanded combat (full damage)**
- [ ] Implement barehanded combat (full damage)
- [ ] **Verify barehanded combat tests pass**
- [ ] **Write tests for weapon combat:**
  - [ ] Damage = max(0, monster - weapon)
  - [ ] Monster placed on weapon stack
  - [ ] Health reduced by calculated damage
- [ ] Implement weapon combat (damage calculation)
- [ ] **Verify weapon combat tests pass**
- [ ] **Write tests for monster defeat and discard placement**
- [ ] Handle monster defeat and card placement
- [ ] **Write tests for monster stack on weapon**
- [ ] Track monsters on equipped weapon
- [ ] **Verify all tests pass**
- [ ] **Run full test suite - all tests must pass**

### Task 2.3: Health & Potions (TDD)
- [ ] **Write tests for health potion:**
  - [ ] Adds value to health
  - [ ] Cannot exceed 20
  - [ ] Overflow ignored (health=18 + potion=5 = 20, not 23)
  - [ ] Only first potion per turn is used
  - [ ] Second potion discarded without effect
  - [ ] Potion flag resets each turn
- [ ] Implement health potion logic
- [ ] **Verify potion tests pass**
- [ ] **Write tests for max health enforcement**
- [ ] Enforce max health of 20
- [ ] **Write tests for one potion per turn limit**
- [ ] Enforce one potion per turn limit
- [ ] **Write tests for multiple potions in single room**
- [ ] Handle multiple potions in single room
- [ ] **Verify all potion tests pass**
- [ ] **Run full test suite - all tests must pass**

### Task 2.4: Game End & Scoring (TDD)
- [ ] **Write tests for game over detection (health <= 0)**
- [ ] Detect game over (health = 0)
- [ ] **Verify game over tests pass**
- [ ] **Write tests for game win detection (dungeon empty)**
- [ ] Detect game win (dungeon empty)
- [ ] **Verify game win tests pass**
- [ ] **Write tests for losing score calculation:**
  - [ ] Negative score = current health - sum of remaining monsters
  - [ ] Edge case: died exactly at 0 with no monsters left
- [ ] Calculate losing score (negative monster sum)
- [ ] **Verify losing score tests pass**
- [ ] **Write tests for winning score calculation:**
  - [ ] Score = remaining health
  - [ ] Special case: health=20 AND last card was potion
- [ ] Calculate winning score (remaining health)
- [ ] **Write tests for special case (health=20 + last potion)**
- [ ] Handle special case (health=20 + last potion)
- [ ] **Verify all scoring tests pass**
- [ ] **Run full test suite - all tests must pass**

## Phase 3: User Interface
**Goal**: Build the game UI with Jetpack Compose
**Coverage Target**: >80% (ViewModel logic), UI tests for critical flows

### Task 3.1: Card UI Components
- [ ] **Write Compose preview tests for card states**
- [ ] Design card composable (visual representation)
- [ ] Implement card type differentiation (colors, symbols)
- [ ] Create card back design
- [ ] **Verify card displays correctly in preview**
- [ ] **Write tests for card animations (if stateful)**
- [ ] Implement card animations (flip, move)

### Task 3.2: Game Board Layout
- [ ] **Write Compose tests for layout rendering**
- [ ] Design table layout (dungeon, room, discard, weapon area)
- [ ] Implement room display (4 cards face up)
- [ ] Implement equipped weapon display
- [ ] Implement monster stack on weapon display
- [ ] Implement discard pile indicator
- [ ] **Verify layout renders correctly in different states**

### Task 3.3: Game Controls & HUD
- [ ] Implement health display
- [ ] Implement score display
- [ ] Implement room avoidance button/gesture
- [ ] Implement card selection mechanism
- [ ] Implement combat choice dialog (barehanded vs weapon)
- [ ] Show remaining cards in dungeon
- [ ] **Write UI tests for critical interactions:**
  - [ ] Card selection flow
  - [ ] Room avoidance
  - [ ] Combat choice dialog

### Task 3.4: Screens & Navigation
- [ ] Create main menu screen
- [ ] Create game screen
- [ ] Create game over screen (with score)
- [ ] Implement navigation between screens
- [ ] Add New Game / Continue options
- [ ] **Write navigation tests (screen transitions)**

### Task 3.5: Basic Help (Essential for playability)
- [ ] Add help button to game screen (always visible)
- [ ] Create basic rules reference screen with:
  - [ ] Card values table (J=11, Q=12, K=13, A=14)
  - [ ] Card types explanation
  - [ ] Weapon degradation basics
  - [ ] Scrollable rules summary
- [ ] Show weapon current max value in UI
- [ ] Damage preview before combat choice
- [ ] Link to view full rules (PDF) in menu

## Phase 4: State Management & Persistence
**Goal**: Save game state and enable continue functionality
**Coverage Target**: >90%

### Task 4.1: ViewModel & State (TDD)
- [ ] **Write tests for GameViewModel:**
  - [ ] Initial state is correct
  - [ ] State updates on intents
  - [ ] State flows emit correct values
  - [ ] Error states handled
- [ ] Create GameViewModel
- [ ] **Verify ViewModel tests pass**
- [ ] **Write tests for intent handling:**
  - [ ] StartNewGame intent
  - [ ] AvoidRoom intent
  - [ ] SelectCard intent
  - [ ] ChooseCombatMethod intent
- [ ] Handle user actions (intents/events)
- [ ] **Verify intent tests pass**
- [ ] **Write tests for state flow updates**
- [ ] Implement state flows for UI updates
- [ ] **Verify state flow tests pass**
- [ ] **Run full test suite - all tests must pass**

### Task 4.2: Data Persistence (TDD)
- [ ] **Write tests for game state serialization/deserialization**
- [ ] Set up Room database
- [ ] Create entities for game state
- [ ] **Verify serialization tests pass**
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
- [ ] **Run full test suite - all tests must pass**
- [ ] Handle multiple save slots (optional)

### Task 4.3: Statistics Tracking (TDD)
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

## Phase 5: Polish & Enhancement
**Goal**: Improve UX and add quality-of-life features

### Task 5.1: Visual Polish
- [ ] Add card animations (smooth transitions)
- [ ] Add damage number animations
- [ ] Improve card graphics/design
- [ ] Add visual feedback for weapon degradation
- [ ] Implement dark mode

### Task 5.2: Help & Tutorial System
- [ ] **Create rules reference screen:**
  - [ ] Card types and their functions
  - [ ] Weapon degradation explanation with examples
  - [ ] Room mechanics
  - [ ] Scoring system
  - [ ] Searchable/scrollable format
- [ ] **Add interactive tutorial:**
  - [ ] First-time user tutorial (optional)
  - [ ] Step-by-step walkthrough of first room
  - [ ] Weapon degradation demonstration
  - [ ] Practice mode (no score)
- [ ] **In-game contextual help:**
  - [ ] Help button/icon always accessible during gameplay
  - [ ] Tooltips for complex actions (weapon degradation state)
  - [ ] "Why can't I use this weapon?" explanations
  - [ ] Damage calculation preview
- [ ] **Quick reference overlay:**
  - [ ] Card values table (J=11, Q=12, K=13, A=14)
  - [ ] Current weapon degradation state display
  - [ ] Rules summary (swipe up from bottom?)
- [ ] Add rules PDF viewer (optional - view Scoundrel.pdf in-app)

### Task 5.3: Quality of Life
- [ ] Add undo functionality
- [ ] Show damage preview before combat
- [ ] Add settings screen
- [ ] Implement haptic feedback

### Task 5.4: Foldable Optimization
- [ ] Test on folded screen
- [ ] Test on unfolded screen
- [ ] Optimize layout for different configurations
- [ ] Handle fold/unfold state changes

### Task 5.5: Audio (Optional)
- [ ] Add card flip sounds
- [ ] Add combat sounds
- [ ] Add victory/defeat sounds
- [ ] Add background music (toggleable)
- [ ] Implement volume controls

## Phase 6: Integration Testing & Release
**Goal**: Ensure quality and prepare for personal use

### Task 6.1: Integration & E2E Testing
- [ ] **Write end-to-end test: Complete winning game**
- [ ] **Write end-to-end test: Complete losing game**
- [ ] **Write end-to-end test: Room avoidance flow**
- [ ] **Write end-to-end test: Weapon degradation scenario**
- [ ] **Verify all E2E tests pass**
- [ ] Manual testing on Pixel 10 Pro Fold
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
- [ ] Address any discovered bugs
- [ ] Performance optimization
- [ ] Memory leak checks
- [ ] Battery usage optimization

### Task 6.3: Documentation
- [ ] Code documentation
- [ ] README with setup instructions
- [ ] Game rules reference in app

## Notes
- Since this is a personal app, we can skip Play Store optimization
- Focus on functionality and personal preferences over broad compatibility
- Can iterate on features based on actual gameplay experience
