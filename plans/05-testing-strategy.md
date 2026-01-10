# Scoundroid - Testing Strategy

## Testing Philosophy

**Test-First Approach**: Write tests before implementation code for all game logic and business rules.

### Why Test-First for Scoundroid?

1. **Complex Game Rules**: Weapon degradation and scoring logic are intricate
2. **Rule Verification**: Tests serve as executable documentation of game rules
3. **Regression Prevention**: Ensure changes don't break existing mechanics
4. **Confidence**: Deploy with certainty that the game works correctly

## Testing Pyramid

```
        /\
       /  \      E2E Tests (Few)
      /----\     - Complete game flows
     /      \    - Critical user journeys
    /--------\
   /          \  Integration Tests (Some)
  /------------\ - ViewModel + UseCase integration
 /--------------\
/                \ Unit Tests (Many)
------------------  - Game logic
                    - Business rules
                    - Data models
```

## Test Coverage Goals

### Phase 1: Foundation (100% coverage required)
- ✅ Deck initialization (44 cards, correct composition)
- ✅ Card value calculations
- ✅ Card type determination
- ✅ Shuffle functionality

### Phase 2: Game Mechanics (100% coverage required)
- ✅ Room drawing (4 cards)
- ✅ Room avoidance logic (can't avoid twice)
- ✅ Card selection tracking (3 of 4)
- ✅ Turn state transitions
- ✅ Health tracking

### Phase 3: Combat System (100% coverage required - CRITICAL)
- ✅ Weapon equipping
- ✅ Weapon degradation tracking
- ✅ Weapon usability determination
- ✅ Barehanded combat damage
- ✅ Weapon combat damage calculation
- ✅ Health potion logic (max 20, 1 per turn)
- ✅ Edge cases (multiple potions, weapon at limit, etc.)

### Phase 4: Scoring & Game End (100% coverage required)
- ✅ Win condition detection
- ✅ Loss condition detection
- ✅ Winning score calculation
- ✅ Losing score calculation
- ✅ Special case: health=20 + last potion

### Phase 5: UI Layer (>80% coverage)
- ✅ ViewModel state updates
- ✅ Intent handling
- ✅ State flows
- ⚠️ Composable UI tests (critical flows only)

### Phase 6: Persistence (>90% coverage)
- ✅ Save game serialization
- ✅ Load game deserialization
- ✅ Statistics tracking
- ✅ Database operations

## Test Organization

```
app/src/test/java/dev/mattbachmann/scoundroid/
├── data/model/
│   ├── CardTest.kt
│   ├── GameStateTest.kt
│   └── WeaponStateTest.kt
├── domain/usecase/
│   ├── InitializeDeckUseCaseTest.kt
│   ├── DrawRoomUseCaseTest.kt
│   ├── ProcessCardUseCaseTest.kt
│   ├── CombatUseCaseTest.kt
│   └── CalculateScoreUseCaseTest.kt
└── ui/screen/game/
    └── GameViewModelTest.kt

app/src/androidTest/java/dev/mattbachmann/scoundroid/
└── ui/
    └── GameFlowTest.kt (E2E critical paths)
```

## Test-First Workflow

### For Each Feature:

1. **Write Failing Test**
   ```kotlin
   @Test
   fun `weapon can only defeat monsters less than or equal to max defeated`() {
       // Arrange
       val weapon = WeaponState(card = Card(Suit.DIAMONDS, Rank.FIVE))
       val defeatedQueen = weapon.defeatMonster(Card(Suit.CLUBS, Rank.QUEEN))

       // Act & Assert
       assertTrue(defeatedQueen.canDefeat(Card(Suit.CLUBS, Rank.SIX)))
       assertFalse(defeatedQueen.canDefeat(Card(Suit.CLUBS, Rank.SEVEN)))
   }
   ```

2. **Run Test** - Verify it fails (RED)

3. **Implement Minimum Code** - Make test pass

4. **Run Test** - Verify it passes (GREEN)

5. **Refactor** - Improve code while keeping tests green

6. **Repeat** - Next test case

## Critical Test Cases by Feature

### Deck Initialization
```kotlin
- deck contains exactly 44 cards
- deck contains all clubs 2-A (13 cards)
- deck contains all spades 2-A (13 cards)
- deck contains diamonds 2-10 only (9 cards)
- deck contains hearts 2-10 only (9 cards)
- deck does NOT contain jokers
- deck does NOT contain red face cards (J♥ Q♥ K♥ J♦ Q♦ K♦)
- deck does NOT contain red aces (A♥ A♦)
- shuffle produces different order
- shuffle maintains card count
```

### Weapon Degradation (MOST COMPLEX)
```kotlin
- new weapon has no maxMonsterValue
- new weapon can defeat any monster
- weapon tracks highest monster value defeated
- weapon with maxValue=12 can defeat monster with value=12
- weapon with maxValue=12 can defeat monster with value=6
- weapon with maxValue=12 CANNOT defeat monster with value=13
- weapon with maxValue=12 used on value=6 updates maxValue to 6
- weapon with maxValue=6 CANNOT defeat monster with value=7
- weapon remains equipped even if unusable
- defeating lower value monster downgrades weapon permanently
```

### Health Potion Logic
```kotlin
- potion adds its value to health
- health cannot exceed 20
- using potion at health=18 with value=5 results in health=20 (not 23)
- first potion in turn is used
- second potion in turn is discarded without effect
- potion usage flag resets each turn
- potion is discarded after use
```

### Room Mechanics
```kotlin
- draw room creates 4-card room
- cannot avoid first room
- can avoid room if last was not avoided
- cannot avoid two rooms in a row
- avoiding room places all 4 cards at bottom of dungeon
- processing room allows selection of 3 of 4 cards
- 4th card remains for next room
- turn ends after 3 cards processed
```

### Combat Damage
```kotlin
- barehanded vs monster value=10 deals 10 damage
- weapon value=5 vs monster value=3 deals 0 damage
- weapon value=5 vs monster value=10 deals 5 damage (10-5)
- weapon value=5 vs monster value=5 deals 0 damage
- health=20, take 8 damage = health=12
- health=5, take 10 damage = health=-5 (game over)
```

### Scoring
```kotlin
- survive with health=15 scores 15
- survive with health=20, last card=potion(5) scores 25
- survive with health=20, last card=monster scores 20
- die with health=-3, remaining monsters sum=50 scores -53
- die at health=0 with no remaining monsters scores 0
```

## Test Utilities

### Test Data Builders
```kotlin
// Create test cards easily
fun testMonster(value: Int) = Card(Suit.CLUBS, Rank.fromValue(value))
fun testWeapon(value: Int) = Card(Suit.DIAMONDS, Rank.fromValue(value))
fun testPotion(value: Int) = Card(Suit.HEARTS, Rank.fromValue(value))

// Create test game states
fun testGameState(
    health: Int = 20,
    dungeon: List<Card> = emptyList(),
    weapon: WeaponState? = null
) = GameState(/* ... */)
```

### Assertion Helpers
```kotlin
fun assertHealth(gameState: GameState, expected: Int) {
    assertEquals(expected, gameState.health, "Health mismatch")
}

fun assertWeaponCanDefeat(weapon: WeaponState, monster: Card) {
    assertTrue(weapon.canDefeat(monster),
        "Weapon ${weapon.card} should defeat monster ${monster}")
}
```

## Running Tests

### Command Line
```bash
# Run all unit tests
./gradlew test

# Run unit tests with coverage
./gradlew testDebugUnitTestCoverage

# Run specific test class
./gradlew test --tests CardTest

# Run instrumented tests
./gradlew connectedAndroidTest

# Run all tests
./gradlew check
```

### Android Studio
- Right-click test class/method → Run Test
- View coverage: Run → Run with Coverage

## Continuous Testing

### Pre-Commit Checks
Before committing:
```bash
./gradlew test
```
All tests must pass.

### Phase Completion Criteria
A phase is NOT complete until:
- ✅ All tests written
- ✅ All tests pass
- ✅ Coverage goals met
- ✅ No regressions in previous phases

## Test Dependencies

### build.gradle.kts
```kotlin
dependencies {
    // Unit Testing
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.jetbrains.kotlin:kotlin-test")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.7.3")
    testImplementation("app.cash.turbine:turbine:1.0.0") // For Flow testing
    testImplementation("io.mockk:mockk:1.13.8") // For mocking (if needed)

    // Android Testing
    androidTestImplementation("androidx.test.ext:junit:1.1.5")
    androidTestImplementation("androidx.test.espresso:espresso-core:3.5.1")
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
```

## Example: TDD Session Flow

### Session Goal: Implement Weapon Degradation

**1. Write First Test (RED)**
```kotlin
@Test
fun `new weapon can defeat any monster`() {
    val weapon = WeaponState(Card(Suit.DIAMONDS, Rank.FIVE))
    assertTrue(weapon.canDefeat(Card(Suit.CLUBS, Rank.ACE)))
}
```
Run: ❌ Fails (method doesn't exist)

**2. Implement (GREEN)**
```kotlin
data class WeaponState(
    val card: Card,
    val maxMonsterValue: Int? = null
) {
    fun canDefeat(monster: Card): Boolean = true // Simplest implementation
}
```
Run: ✅ Passes

**3. Write Next Test (RED)**
```kotlin
@Test
fun `weapon used on queen cannot defeat king`() {
    val weapon = WeaponState(Card(Suit.DIAMONDS, Rank.FIVE))
    val afterQueen = weapon.defeatMonster(Card(Suit.CLUBS, Rank.QUEEN))
    assertFalse(afterQueen.canDefeat(Card(Suit.CLUBS, Rank.KING)))
}
```
Run: ❌ Fails (returns true, should be false)

**4. Implement Properly (GREEN)**
```kotlin
fun canDefeat(monster: Card): Boolean {
    return maxMonsterValue?.let { monster.value <= it } ?: true
}

fun defeatMonster(monster: Card): WeaponState {
    return copy(maxMonsterValue = monster.value)
}
```
Run: ✅ Both tests pass

**5. Continue until all cases covered**

## Testing Checklist per Phase

Before marking a phase complete:

- [ ] All game rules from docs/rules.md are tested
- [ ] All edge cases have tests
- [ ] All tests pass (`./gradlew test`)
- [ ] Coverage meets phase goals
- [ ] No @Ignore or commented-out tests
- [ ] Test names clearly describe what they test
- [ ] Tests are fast (unit tests < 100ms each)
- [ ] Tests are independent (can run in any order)
- [ ] Tests are repeatable (same result every time)

## Common Testing Mistakes to Avoid

❌ Testing implementation details instead of behavior
❌ Writing tests after code (defeats the purpose)
❌ Skipping edge cases
❌ Tests that depend on each other
❌ Not testing the unhappy path
❌ Mocking too much (prefer real objects for data classes)
❌ Ignoring failing tests

## Documentation in Tests

Tests serve as living documentation:
```kotlin
@Test
fun `weapon degradation - example from rulebook page 2`() {
    // Setup: Player has 5 of Diamonds weapon
    val weapon = WeaponState(testWeapon(5))

    // Action: Defeat Queen (value 12)
    val afterQueen = weapon.defeatMonster(testMonster(12))

    // Result: Can now use on any monster <= 12
    assertTrue(afterQueen.canDefeat(testMonster(12)))
    assertTrue(afterQueen.canDefeat(testMonster(6)))

    // Action: Defeat 6 (value 6)
    val afterSix = afterQueen.defeatMonster(testMonster(6))

    // Result: Can now ONLY use on monsters <= 6
    assertTrue(afterSix.canDefeat(testMonster(6)))
    assertFalse(afterSix.canDefeat(testMonster(7)))
    assertFalse(afterSix.canDefeat(testMonster(12))) // Degraded!
}
```

---

**Remember**: If it's not tested, it doesn't work.
**Every feature. Every phase. Tests first.**
