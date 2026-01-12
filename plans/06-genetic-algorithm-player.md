# Genetic Algorithm Player Optimizer

## Goal
Use a genetic algorithm to evolve an optimized heuristic player that achieves a higher win rate than the current ~0.14%.

## Background

The current `HeuristicPlayer` wins about 0.14% of games (14 out of 10,000 seeds) after implementing the weapon preservation strategy. Analysis of 10,000 losses revealed:

- **95% of deaths have an unused weapon** - weapon degradation is the #1 killer
- **74% of deaths where weapon COULD have helped** if not degraded
- **63% of deaths from face cards and aces** (Jacks through Aces)
- Cascading degradation within rooms is a major problem

### Key Insights from Analysis
1. Using fresh weapons on small monsters degrades them permanently
2. Once degraded, using weapon on ALL monsters in a room causes further cascading degradation
3. The weapon preservation threshold (currently 8) is the most impactful parameter
4. Room skipping thresholds affect survival but are secondary to weapon management

## Implementation Plan

### Phase 1: Create Parameterized Player

**File:** `app/src/main/java/dev/mattbachmann/scoundroid/domain/solver/ParameterizedPlayer.kt`

Create a player that takes a `PlayerGenome` data class with tunable parameters:

```kotlin
data class PlayerGenome(
    // Room skip thresholds
    val skipIfDamageExceedsHealthMinus: Int,        // e.g., 2 = skip if damage >= health - 2
    val skipWithoutWeaponDamageFraction: Double,    // e.g., 0.5 = skip if no weapon help and damage > 50% health

    // Card leave evaluation
    val monsterLeavePenaltyMultiplier: Double,      // e.g., 1.0 = linear, 2.0 = quadratic penalty
    val weaponLeavePenaltyIfNeeded: Double,         // e.g., 10.0 = big penalty for leaving needed weapons

    // Weapon preservation (CRITICAL - most impactful parameters)
    val weaponPreservationThreshold: Int,           // e.g., 8 = only use fresh weapon on monsters >= 8
    val minDamageSavedToUseWeapon: Int,             // e.g., 3 = only use degraded weapon if it saves 3+ damage
    val emergencyHealthBuffer: Int,                 // e.g., 2 = use weapon if health <= monster.value + buffer

    // Weapon equip decisions
    val equipFreshWeaponIfDegradedBelow: Int,       // e.g., 6 = equip fresh weapon if current degraded below 6
)
```

**8 parameters total** - focused on the decisions that matter most based on analysis.

### Phase 2: Create GA Framework

**File:** `app/src/main/java/dev/mattbachmann/scoundroid/domain/solver/GeneticOptimizer.kt`

```kotlin
class GeneticOptimizer(
    val populationSize: Int = 50,
    val gamesPerEvaluation: Int = 5000,  // Higher for less noise with sparse wins
    val mutationRate: Double = 0.15,
    val crossoverRate: Double = 0.7,
    val eliteCount: Int = 5,
)
```

**Key functions:**
1. `randomGenome()` - Create random genome within valid bounds
2. `evaluate(genome: PlayerGenome): FitnessResult` - Play N games, return weighted fitness
3. `select(population: List<ScoredGenome>): List<PlayerGenome>` - Tournament selection
4. `crossover(parent1: PlayerGenome, parent2: PlayerGenome): PlayerGenome` - Blend parameters
5. `mutate(genome: PlayerGenome): PlayerGenome` - Random perturbation
6. `evolve(generations: Int): PlayerGenome` - Main loop

### Phase 3: Create Test/Runner

**File:** `app/src/test/java/dev/mattbachmann/scoundroid/domain/solver/GeneticOptimizerTest.kt`

```kotlin
@Test
fun `evolve optimal player`() {
    val optimizer = GeneticOptimizer(
        populationSize = 50,
        gamesPerEvaluation = 5000,
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
- Population of 50, 5000 games each = 250,000 games per generation
- At 50k games/sec = ~5 seconds per generation
- 50 generations = ~4-5 minutes total
- With parallelization: ~1-2 minutes total

### Fitness Function

Use weighted combination to get gradient even from losses:

```kotlin
fun fitness(results: SimulationResults): Double {
    // Wins are worth a lot, but losses still provide signal
    // Score in losses is negative (e.g., -50 to -150), wins are positive (1-20)
    return results.winRate * 10000 + results.averageScore
}
```

This gives partial credit for "better" losses (dying later with fewer cards remaining).

### Parameter Bounds
```kotlin
val GENOME_BOUNDS = mapOf(
    // Room skip
    "skipIfDamageExceedsHealthMinus" to 0..10,
    "skipWithoutWeaponDamageFraction" to 0.3..0.8,

    // Card leaving
    "monsterLeavePenaltyMultiplier" to 0.5..3.0,
    "weaponLeavePenaltyIfNeeded" to 0.0..20.0,

    // Weapon preservation (CRITICAL)
    "weaponPreservationThreshold" to 5..12,
    "minDamageSavedToUseWeapon" to 0..5,
    "emergencyHealthBuffer" to 0..5,

    // Weapon equip
    "equipFreshWeaponIfDegradedBelow" to 3..10,
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

1. `PlayerGenome.kt` - Data class with 8 tunable parameters
2. `ParameterizedPlayer.kt` - Player that uses genome parameters for decisions
3. `GeneticOptimizer.kt` - GA implementation with fitness evaluation
4. `GeneticOptimizerTest.kt` - Test runner that evolves and validates

**Existing files to keep:**
- `GameLog.kt` - Already created for logging/analysis
- `LoggingHeuristicPlayer.kt` - Useful for debugging evolved strategies

## Alternative Approaches (if GA doesn't work well)

1. **Bayesian Optimization** - More sample-efficient for continuous parameters
2. **Monte Carlo Tree Search** - For per-game decisions rather than fixed policy
3. **Reinforcement Learning** - Train a neural network policy (more complex)

## References

- Current heuristic player: `app/src/main/java/dev/mattbachmann/scoundroid/domain/solver/HeuristicPlayer.kt`
- Simulator: `HeuristicSimulator` class in same file
- Test file: `app/src/test/java/dev/mattbachmann/scoundroid/domain/solver/HeuristicPlayerTest.kt`
