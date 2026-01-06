# Scoundroid - Technical Architecture

## Architecture Pattern: MVI (Model-View-Intent)

We'll use MVI pattern which works well with Jetpack Compose:
- **Model**: Represents the game state
- **View**: Composable UI that renders the state
- **Intent**: User actions that trigger state changes

```
User Action → Intent → ViewModel → State Update → UI Recomposition
```

## Project Structure

```
app/src/main/java/com/scoundroid/
├── data/
│   ├── model/           # Data classes
│   │   ├── Card.kt
│   │   ├── GameState.kt
│   │   ├── Room.kt
│   │   └── Weapon.kt
│   ├── repository/      # Data layer
│   │   ├── GameRepository.kt
│   │   └── StatisticsRepository.kt
│   └── database/        # Room DB
│       ├── AppDatabase.kt
│       ├── dao/
│       └── entity/
├── domain/
│   ├── usecase/         # Business logic
│   │   ├── InitializeDeckUseCase.kt
│   │   ├── DrawRoomUseCase.kt
│   │   ├── ProcessCardUseCase.kt
│   │   └── CalculateScoreUseCase.kt
│   └── model/           # Domain models
├── ui/
│   ├── screen/
│   │   ├── game/        # Main game screen
│   │   │   ├── GameScreen.kt
│   │   │   ├── GameViewModel.kt
│   │   │   └── GameIntent.kt
│   │   ├── menu/        # Main menu
│   │   └── gameover/    # Game over screen
│   ├── component/       # Reusable UI components
│   │   ├── CardView.kt
│   │   ├── HealthBar.kt
│   │   └── RoomView.kt
│   └── theme/           # Theming
└── util/                # Utilities
```

## Core Data Models

### Card
```kotlin
data class Card(
    val suit: Suit,
    val rank: Rank
) {
    val value: Int get() = rank.value
    val type: CardType get() = when(suit) {
        Suit.CLUBS, Suit.SPADES -> CardType.MONSTER
        Suit.DIAMONDS -> CardType.WEAPON
        Suit.HEARTS -> CardType.POTION
    }
}

enum class Suit { CLUBS, SPADES, DIAMONDS, HEARTS }
enum class Rank(val value: Int) {
    TWO(2), THREE(3), FOUR(4), FIVE(5), SIX(6),
    SEVEN(7), EIGHT(8), NINE(9), TEN(10),
    JACK(11), QUEEN(12), KING(13), ACE(14)
}
enum class CardType { MONSTER, WEAPON, POTION }
```

### GameState
```kotlin
data class GameState(
    val dungeon: List<Card>,           // Face-down deck
    val currentRoom: List<Card>,       // 4 cards face up
    val discard: List<Card>,           // Discarded cards
    val equippedWeapon: WeaponState?,  // Current weapon
    val health: Int,                   // Current health (max 20)
    val cardsProcessedThisTurn: Int,   // Count of cards chosen (max 3)
    val usedPotionThisTurn: Boolean,   // Can only use 1 potion per turn
    val lastRoomAvoided: Boolean,      // Can't avoid 2 rooms in a row
    val gameStatus: GameStatus,        // PLAYING, WON, LOST
    val score: Int?                    // Final score (null while playing)
)

data class WeaponState(
    val card: Card,                    // The weapon card
    val monstersDefeated: List<Card>,  // Monsters on this weapon
    val maxMonsterValue: Int?          // Highest value monster defeated (null if unused)
) {
    fun canDefeat(monster: Card): Boolean {
        return maxMonsterValue?.let { monster.value <= it } ?: true
    }
}

enum class GameStatus { PLAYING, WON, LOST }
```

## Game Logic Flow

### Turn Flow
1. **Draw Room**: Draw 4 cards from dungeon
2. **Player Choice**: Avoid room OR process room
3. **Process Room**: Select and process 3 of 4 cards
4. **Next Turn**: Keep 4th card for next room

### Card Processing Logic

**Weapon (Diamond)**:
```
1. Discard current weapon + monsters on it
2. Equip new weapon
3. Reset weapon state (no monsters defeated yet)
```

**Potion (Heart)**:
```
1. Check if potion already used this turn
2. Add value to health (cap at 20)
3. Mark potion as used this turn
4. Discard potion
```

**Monster (Club/Spade)**:
```
1. Player chooses: Barehanded OR Weapon
2. Barehanded:
   - Take full monster damage
   - Discard monster
3. Weapon:
   a. Check if weapon can be used (monster.value <= weapon.maxMonsterValue)
   b. Calculate damage: max(0, monster.value - weapon.value)
   c. Take damage
   d. Place monster on weapon
   e. Update weapon.maxMonsterValue to monster.value
```

### Weapon Degradation Rule
A weapon tracks the highest-value monster it has defeated. It can only be used on monsters with value ≤ that maximum.

Example:
- Weapon: 5 of Diamonds
- Defeat Queen (12): weapon.maxMonsterValue = 12
- Can now use on: any monster with value ≤ 12
- Defeat 6 (6): weapon.maxMonsterValue = 6
- Can now use on: any monster with value ≤ 6
- Cannot use on: 7, 8, 9, 10, J, Q, K, A

## UI State Management

### ViewModel Pattern
```kotlin
class GameViewModel : ViewModel() {
    private val _state = MutableStateFlow(GameState.initial())
    val state: StateFlow<GameState> = _state.asStateFlow()

    fun handleIntent(intent: GameIntent) {
        when (intent) {
            is GameIntent.StartNewGame -> startNewGame()
            is GameIntent.AvoidRoom -> avoidRoom()
            is GameIntent.SelectCard -> selectCard(intent.card)
            is GameIntent.ChooseCombatMethod -> chooseCombatMethod(intent.method)
            // ... other intents
        }
    }
}

sealed class GameIntent {
    object StartNewGame : GameIntent()
    object AvoidRoom : GameIntent()
    data class SelectCard(val card: Card) : GameIntent()
    data class ChooseCombatMethod(val method: CombatMethod) : GameIntent()
}
```

## Persistence Strategy

### Room Database Schema
```kotlin
@Entity(tableName = "saved_games")
data class SavedGameEntity(
    @PrimaryKey val id: Int = 1, // Single save slot for MVP
    val gameStateJson: String,    // Serialize GameState to JSON
    val timestamp: Long
)

@Entity(tableName = "statistics")
data class StatisticsEntity(
    @PrimaryKey val id: Int = 1,
    val gamesPlayed: Int,
    val gamesWon: Int,
    val gamesLost: Int,
    val highestScore: Int,
    val lowestScore: Int
)
```

## Display Considerations for Foldable

### Folded Mode (Cover Screen)
- Vertical layout
- Compact card display
- Simplified UI

### Unfolded Mode (Inner Screen)
- Horizontal layout with more space
- Larger cards
- Room for statistics panel

### Implementation
```kotlin
@Composable
fun GameScreen() {
    val windowSizeClass = calculateWindowSizeClass()

    when (windowSizeClass.widthSizeClass) {
        WindowWidthSizeClass.Compact -> CompactGameLayout()
        WindowWidthSizeClass.Medium,
        WindowWidthSizeClass.Expanded -> ExpandedGameLayout()
    }
}
```

## Animation Strategy

### Card Animations
- Flip animation when drawn from dungeon
- Slide animation when moving to discard
- Fade animation for damage numbers
- Stack animation for monsters on weapon

### Performance
- Use `animateFloatAsState` for simple animations
- Use `Animatable` for complex sequences
- Keep animations under 300ms for snappy feel

## Testing Strategy

### Unit Tests
- Game logic (deck initialization, combat calculations)
- Weapon degradation logic
- Score calculation
- State transitions

### UI Tests
- Critical user flows
- Screen navigation
- State persistence

### Manual Testing Checklist
- [ ] Can complete a full game
- [ ] Weapon degradation works correctly
- [ ] Health never exceeds 20
- [ ] Can't use 2 potions per turn
- [ ] Can't avoid 2 rooms in a row
- [ ] Score calculation is correct
- [ ] Save/load works
- [ ] Works in both folded and unfolded modes

## Dependencies (Preliminary)

```kotlin
// Jetpack Compose
implementation("androidx.compose.ui:ui")
implementation("androidx.compose.material3:material3")
implementation("androidx.compose.ui:ui-tooling-preview")

// ViewModel & Lifecycle
implementation("androidx.lifecycle:lifecycle-viewmodel-compose")
implementation("androidx.lifecycle:lifecycle-runtime-compose")

// Room Database
implementation("androidx.room:room-runtime")
implementation("androidx.room:room-ktx")
ksp("androidx.room:room-compiler")

// Kotlin Coroutines
implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android")

// Navigation
implementation("androidx.navigation:navigation-compose")

// Serialization (for save states)
implementation("org.jetbrains.kotlinx:kotlinx-serialization-json")

// Window Size Classes (foldable support)
implementation("androidx.compose.material3:material3-window-size-class")
```

## Decision Log

### Why MVI over MVVM?
- Unidirectional data flow works well with game state
- Clear separation of user actions (intents)
- Easier to test and reason about state changes

### Why Jetpack Compose over XML?
- Modern, declarative approach
- Better state management
- Easier animations
- Less boilerplate

### Why Room over DataStore?
- Need to store complex game state
- May want multiple save slots in future
- Better for relational data (statistics)

### Why single Activity?
- Compose Navigation works well with single Activity
- Simpler lifecycle management
- Aligns with modern Android architecture
