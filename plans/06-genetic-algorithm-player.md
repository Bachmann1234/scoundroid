# Genetic Algorithm Player Optimizer

## Goal
Use a genetic algorithm to evolve an optimized heuristic player that achieves a higher win rate than the current ~0.1%.

## Background

The current `HeuristicPlayer` in `app/src/main/java/dev/mattbachmann/scoundroid/domain/solver/HeuristicPlayer.kt` wins about 0.094% of games (94 out of 100,000 seeds). It has many hardcoded decision thresholds that could be optimized.

### Current Hardcoded Values
- Room skip: `estimatedNetDamage >= state.health - 2` and `estimatedNetDamage > state.health / 2`
- Card leave penalty: `cardToLeave.value` for monsters (linear)
- Effective healing cap calculation
- No parameters for weapon usage strategy

## Implementation Plan

### Phase 1: Create Parameterized Player

**File:** `app/src/main/java/dev/mattbachmann/scoundroid/domain/solver/ParameterizedPlayer.kt`

Create a player that takes a `PlayerGenome` data class with tunable parameters:

```kotlin
data class PlayerGenome(
    // Room skip thresholds
    val skipIfDamageExceedsHealthFraction: Double,  // e.g., 0.5 = skip if damage > 50% health
    val skipIfDamageExceedsHealthMinus: Int,        // e.g., 2 = skip if damage >= health - 2
    val skipWithoutWeaponThreshold: Double,         // e.g., 0.3 = more aggressive skipping without weapon

    // Card leave evaluation
    val monsterLeavePenaltyMultiplier: Double,      // e.g., 1.0 = linear, 2.0 = quadratic penalty
    val potionLeaveBonus: Double,                   // e.g., 0.5 = prefer leaving potions
    val weaponLeavePenaltyIfNeeded: Double,         // e.g., 10.0 = big penalty for leaving needed weapons

    // Combat decisions
    val healingThresholdFraction: Double,           // e.g., 0.25 = heal first if health < 25%
    val useWeaponOnSmallMonstersThreshold: Int,     // e.g., 5 = always use weapon if monster >= 5

    // Weapon equip decisions
    val equipFreshWeaponIfDegradedBelow: Int,       // e.g., 6 = equip fresh weapon if current degraded below 6
)
```

### Phase 2: Create GA Framework

**File:** `app/src/main/java/dev/mattbachmann/scoundroid/domain/solver/GeneticOptimizer.kt`

```kotlin
class GeneticOptimizer(
    val populationSize: Int = 50,
    val gamesPerEvaluation: Int = 1000,
    val mutationRate: Double = 0.1,
    val crossoverRate: Double = 0.7,
    val eliteCount: Int = 5,
)
```

**Key functions:**
1. `randomGenome()` - Create random genome within valid bounds
2. `evaluate(genome: PlayerGenome): FitnessResult` - Play N games, return win rate + avg score
3. `select(population: List<ScoredGenome>): List<PlayerGenome>` - Tournament or roulette selection
4. `crossover(parent1: PlayerGenome, parent2: PlayerGenome): PlayerGenome` - Blend parameters
5. `mutate(genome: PlayerGenome): PlayerGenome` - Random perturbation
6. `evolve(generations: Int): PlayerGenome` - Main loop

### Phase 3: Create Test/Runner

**File:** `app/src/test/java/dev/mattbachmann/scoundroid/domain/solver/GeneticOptimizerTest.kt`

```kotlin
@Test
fun `evolve optimal player`() {
    val optimizer = GeneticOptimizer(
        populationSize = 100,
        gamesPerEvaluation = 2000,
        mutationRate = 0.15,
    )

    val bestGenome = optimizer.evolve(generations = 50)

    // Validate on fresh seeds
    val validator = ParameterizedPlayer(bestGenome)
    val results = validator.playSeeds(100_001L..200_000L)

    println("Best genome: $bestGenome")
    println("Validation win rate: ${results.winRate}%")
}
```

### Phase 4: Integration

Once we find good parameters:
1. Update `HeuristicPlayer` with the optimized values
2. Or replace it with `ParameterizedPlayer` using the best genome
3. Consider adding the best genome as a default constant

## Technical Considerations

### Performance
- Current speed: ~50,000-90,000 games/sec
- Population of 100, 2000 games each = 200,000 games per generation
- At 50k games/sec = ~4 seconds per generation
- 50 generations = ~3-4 minutes total

### Fitness Function Options
1. **Win rate only** - Simple, but most games lose so signal is sparse
2. **Win rate + average score** - Better gradient (surviving longer = higher score even in loss)
3. **Weighted combination** - `winRate * 1000 + avgScore`

### Parameter Bounds
```kotlin
val GENOME_BOUNDS = mapOf(
    "skipIfDamageExceedsHealthFraction" to 0.2..0.9,
    "skipIfDamageExceedsHealthMinus" to 0..10,
    "monsterLeavePenaltyMultiplier" to 0.5..3.0,
    // etc.
)
```

### Parallelization
- Genome evaluation is embarrassingly parallel
- Use `kotlinx.coroutines` with `Dispatchers.Default` for parallel fitness evaluation
- Could speed up by 4-8x on multi-core

## Success Criteria

- **Minimum:** Achieve >0.5% win rate (5x improvement)
- **Good:** Achieve >1% win rate (10x improvement)
- **Excellent:** Achieve >5% win rate (50x improvement)

## Files to Create

1. `ParameterizedPlayer.kt` - Player that uses genome parameters
2. `PlayerGenome.kt` - Data class for parameters (or nested in ParameterizedPlayer)
3. `GeneticOptimizer.kt` - GA implementation
4. `GeneticOptimizerTest.kt` - Test runner

## Alternative Approaches (if GA doesn't work well)

1. **Bayesian Optimization** - More sample-efficient for continuous parameters
2. **Monte Carlo Tree Search** - For per-game decisions rather than fixed policy
3. **Reinforcement Learning** - Train a neural network policy (more complex)

## References

- Current heuristic player: `app/src/main/java/dev/mattbachmann/scoundroid/domain/solver/HeuristicPlayer.kt`
- Simulator: `HeuristicSimulator` class in same file
- Test file: `app/src/test/java/dev/mattbachmann/scoundroid/domain/solver/HeuristicPlayerTest.kt`
