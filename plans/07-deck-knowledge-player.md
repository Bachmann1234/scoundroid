# Plan: Deck Knowledge-Based Player

## The Insight

The current heuristic player makes decisions based only on what's visible (current room, health, weapon state). But in Scoundrel, you know the exact 44-card deck composition and can track what's been played. This "card counting" information should dramatically improve decision-making.

## Current State (as of this session)

- **Win rate:** ~0.2% with evolved GA parameters
- **Winnable seeds collected:** 1,018 seeds from 500k games
- **Key analysis finding:** Early weapon availability is the strongest predictor of winning (78% of wins had weapon in Room 1 vs 62% of losses)
- **Primary cause of death:** Weapon degradation - player has weapon but it's degraded below the monster's value

## What We're Currently Ignoring

| Information | Why It Matters |
|-------------|----------------|
| Which monsters remain | If Aces are gone, max threat is 13 not 14 |
| Remaining weapons | Worth skipping rooms to find 10♦ if our weapon is degraded |
| Remaining potions | Determines how conservative to be with health |
| Expected future damage | Informs skip decisions and health management |
| Cards left in deck | Probability calculations for next room |

## Implementation Plan

### Phase 1: Deck Tracking Data Structure

Create a `DeckKnowledge` class that tracks the game state:

```kotlin
data class DeckKnowledge(
    // Raw tracking
    val seenCards: Set<Card>,
    val remainingCards: List<Card>,

    // Derived monster info
    val remainingMonsters: List<Card>,
    val maxMonsterRemaining: Int,
    val totalDamageRemaining: Int,
    val monstersRemainingCount: Int,

    // Derived weapon info
    val remainingWeapons: List<Card>,
    val maxWeaponRemaining: Int,
    val weaponsRemainingCount: Int,

    // Derived potion info
    val remainingPotions: List<Card>,
    val totalHealingRemaining: Int,
    val potionsRemainingCount: Int,
) {
    companion object {
        fun initial(): DeckKnowledge // Start with full deck knowledge
    }

    fun cardSeen(card: Card): DeckKnowledge // Update when card is revealed
    fun cardsInRoom(cards: List<Card>): DeckKnowledge // Update for room

    // Probability helpers
    fun expectedDamagePerRoom(): Double
    fun chanceOfWeaponInNextRoom(): Double
    fun chanceOfPotionInNextRoom(): Double
    fun survivalMargin(currentHealth: Int): Int // health + healing - damage
}
```

### Phase 2: Knowledge-Based Decision Functions

Update decision logic to use deck knowledge:

#### 1. Dynamic Weapon Preservation Threshold
```kotlin
// Current: static threshold of 9
// New: based on max monster remaining
fun getWeaponPreservationThreshold(knowledge: DeckKnowledge): Int {
    // If max remaining monster is 10, no need to preserve for 11+
    return minOf(DEFAULT_THRESHOLD, knowledge.maxMonsterRemaining)
}
```

#### 2. Smarter Room Skipping
```kotlin
fun shouldSkipRoom(state: GameState, room: List<Card>, knowledge: DeckKnowledge): Boolean {
    // Factor in:
    // - Expected future difficulty (damage remaining / rooms remaining)
    // - Chance of finding weapon if we skip
    // - Survival margin (can we afford to take damage now?)

    val expectedFutureDamagePerRoom = knowledge.expectedDamagePerRoom()
    val survivalMargin = knowledge.survivalMargin(state.health)
    val weaponChance = knowledge.chanceOfWeaponInNextRoom()

    // Skip more if: low survival margin, good chance of weapon, high future difficulty
    // Skip less if: high survival margin, weapons depleted, low future difficulty
}
```

#### 3. Better Card Leave-Behind Decisions
```kotlin
fun evaluateLeaveChoice(card: Card, knowledge: DeckKnowledge): Int {
    when (card.type) {
        MONSTER -> {
            // Leaving a monster is worse if we have no weapon AND
            // there are few weapons remaining to find
            val weaponChance = knowledge.chanceOfWeaponInNextRoom()
            // Adjust penalty based on weapon availability
        }
        WEAPON -> {
            // Leaving a weapon is worse if it's the best remaining
            // and our current weapon is degraded
            if (card.value >= knowledge.maxWeaponRemaining) {
                // Higher penalty - this might be our last good weapon
            }
        }
        POTION -> {
            // Leaving potion is worse if few potions remain
            val healingScarcity = 1.0 - (knowledge.totalHealingRemaining / MAX_POSSIBLE_HEALING)
            // Adjust based on scarcity
        }
    }
}
```

#### 4. Weapon Equip Decisions
```kotlin
fun shouldEquipWeapon(current: WeaponState?, newWeapon: Card, knowledge: DeckKnowledge): Boolean {
    // Current logic plus:
    // - If new weapon can handle maxMonsterRemaining, definitely equip
    // - If current weapon (even degraded) can handle maxMonsterRemaining, maybe keep it

    val currentCanHandleRemaining = current?.let {
        it.maxMonsterValue == null || it.maxMonsterValue >= knowledge.maxMonsterRemaining
    } ?: false

    val newCanHandleRemaining = newWeapon.value >= knowledge.maxMonsterRemaining

    // Prefer keeping degraded weapon if it can still handle everything remaining
}
```

### Phase 3: Create InformedPlayer Class

```kotlin
class InformedPlayer {
    fun playGame(initialState: GameState): GameState {
        var state = initialState
        var knowledge = DeckKnowledge.initial()

        while (!state.isGameOver && !isWon(state)) {
            // Update knowledge with visible cards
            state.currentRoom?.let { room ->
                knowledge = knowledge.cardsInRoom(room)
            }

            state = playOneStep(state, knowledge)

            // Update knowledge with cards that left play
            // (processed cards, discarded cards)
        }

        return state
    }

    private fun playOneStep(state: GameState, knowledge: DeckKnowledge): GameState {
        // Same structure as HeuristicPlayer but decisions use knowledge
    }
}
```

### Phase 4: Testing & Validation

1. **Unit tests for DeckKnowledge**
   - Verify tracking accuracy
   - Test probability calculations
   - Edge cases (empty deck, all of one type gone)

2. **Benchmark against winnable seeds**
   - Current player: 100% win rate on winnable seeds (by definition)
   - Goal: Maintain or improve while also winning NEW seeds

3. **Benchmark against full seed range**
   - Current: ~0.2% win rate
   - Target: 0.5%+ win rate (2.5x improvement)

4. **Analyze decision differences**
   - Log when InformedPlayer makes different decisions than HeuristicPlayer
   - Verify the knowledge-based decisions are sensible

### Phase 5: Optional GA Re-optimization

After implementing the informed player, we could:
1. Add tunable parameters for how much weight to give various knowledge factors
2. Run GA against winnable seeds for cleaner signal
3. Potentially find even better parameter combinations

## Files to Create/Modify

| File | Action |
|------|--------|
| `DeckKnowledge.kt` | **Create** - Deck tracking data structure |
| `InformedPlayer.kt` | **Create** - New player using deck knowledge |
| `DeckKnowledgeTest.kt` | **Create** - Unit tests for tracking |
| `InformedPlayerTest.kt` | **Create** - Integration tests |
| `LoggingSimulatorTest.kt` | **Modify** - Add benchmark comparing players |

## Implementation Status

**Status:** Phase 1-3 Complete, Phase 4 In Progress

### Implemented Improvements (with 0 regressions)

1. **Dynamic Weapon Preservation Threshold** - Uses `minOf(9, maxMonsterRemaining)` to adjust when to preserve a fresh weapon. If all monsters >= 9 are gone, no need to save the weapon for them.

2. **Use Weapon to Avoid Death** - If fighting barehanded would kill us, use the weapon regardless of preservation threshold. Simple, targeted, saves games in critical moments.

3. **Knowledge-Aware Weapon Equip** - If current weapon (even degraded) can handle all remaining monsters, don't swap to a lower-value fresh weapon.

### Results (50,000 seeds)

| Player | Wins | Win Rate |
|--------|------|----------|
| HeuristicPlayer | 122 | 0.244% |
| InformedPlayer | 137 | 0.274% |

**Improvement:** +15 wins (+12.3%), **0 regressions**

### Attempted Improvements That Caused Regressions (Reverted)

- **Complex combat logic** (survival margin, weapon chance) - 10 regressions
- **Skip when doomed** - 1 regression
- **Skip more when weapon likely** - 3 regressions
- **Leave-behind scarcity logic** - 13 regressions (though gained 7 wins)

### Key Learnings

1. **Simple, targeted improvements work** - The GA-evolved HeuristicPlayer thresholds are well-calibrated. Complex heuristics often backfire.

2. **Clear-cut cases are safe** - "Use weapon to avoid death" only triggers in obvious situations where any other choice means losing.

3. **Knowledge helps at the margins** - Dynamic threshold helps when big monsters are processed early. Most seeds follow similar patterns, so the improvement is incremental.

## Success Metrics

| Metric | Baseline | Target | Achieved |
|--------|----------|--------|----------|
| Win rate (50k seeds) | 0.244% | 0.5%+ | 0.274% |
| Regressions | - | 0 | 0 |
| Win improvement | - | +20% | +12.3% |

## Key Scenarios Where This Helps

1. **Late game, both Aces played**: Weapon degraded to 10 is still useful for everything remaining. Current player might waste it or skip unnecessarily.

2. **No weapons seen yet, 3 weapons in deck**: High chance of weapon soon - maybe take some damage now knowing help is coming.

3. **Most potions played, low health**: Be extremely conservative, skip more rooms.

4. **All big monsters played**: Can fight more aggressively, less need to preserve weapon.

5. **High weapon still in deck, current weapon degraded**: Skip rooms to find it rather than fighting with suboptimal weapon.

## Notes

- The deck in Scoundrel is deterministic given a seed - we know exactly what 44 cards exist
- Cards go to a discard pile, so tracking is straightforward
- This is "perfect information" card counting - completely valid strategy
- The room skip mechanic puts cards at bottom of deck, so they'll come back eventually

---

**Last Updated:** 2026-01-11
**Status:** Implementation complete - 12.3% improvement with 0 regressions
**Priority:** Consider additional improvements or move on to other features
