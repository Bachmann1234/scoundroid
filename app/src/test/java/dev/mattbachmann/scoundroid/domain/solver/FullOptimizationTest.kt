package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import org.junit.Test
import kotlin.random.Random

class FullOptimizationTest {
    @Test
    fun `compare baseline vs fully optimized`() {
        val testSeeds = 0L..10000L

        println("=== FULL OPTIMIZATION COMPARISON ===")
        println()

        // Baseline (before all our changes)
        val baselineGenome = PlayerGenome(
            skipIfDamageExceedsHealthMinus = 5,
            skipWithoutWeaponDamageFraction = 0.444,
            skipDamageHealthFraction = 1.0, // disabled
            monsterLeavePenaltyMultiplier = 0.894,
            weaponLeavePenaltyIfNeeded = 2.505,
            potionLeavePenaltyPerRemaining = 0.0, // disabled
            weaponPreservationThreshold = 9,
            minDamageSavedToUseWeapon = 0,
            emergencyHealthBuffer = 0,
            equipFreshWeaponIfDegradedBelow = 10,
            alwaysSwapToFreshIfDegradedBelow = 0, // disabled
        )

        // Fully optimized
        val optimizedGenome = PlayerGenome(
            skipIfDamageExceedsHealthMinus = 5,
            skipWithoutWeaponDamageFraction = 0.444,
            skipDamageHealthFraction = 0.4, // skip if damage > 40% health
            monsterLeavePenaltyMultiplier = 0.894,
            weaponLeavePenaltyIfNeeded = 2.505,
            potionLeavePenaltyPerRemaining = 0.5, // avoid potion cascade
            weaponPreservationThreshold = 10, // more conservative with fresh weapons
            minDamageSavedToUseWeapon = 0,
            emergencyHealthBuffer = 0,
            equipFreshWeaponIfDegradedBelow = 10,
            alwaysSwapToFreshIfDegradedBelow = 8, // swap to fresh if degraded < 8
        )

        var baselineWins = 0
        var baselineScore = 0L
        var optimizedWins = 0
        var optimizedScore = 0L

        for (seed in testSeeds) {
            val game = GameState.newGame(Random(seed))

            val baselinePlayer = ParameterizedPlayer(baselineGenome)
            val baselineResult = baselinePlayer.playGame(game)
            if (baselineResult.isGameWon) baselineWins++
            baselineScore += baselineResult.calculateScore()

            val optimizedPlayer = ParameterizedPlayer(optimizedGenome)
            val optimizedResult = optimizedPlayer.playGame(GameState.newGame(Random(seed)))
            if (optimizedResult.isGameWon) optimizedWins++
            optimizedScore += optimizedResult.calculateScore()
        }

        val baselineWinRate = baselineWins * 100.0 / testSeeds.count()
        val optimizedWinRate = optimizedWins * 100.0 / testSeeds.count()
        val improvement = (optimizedWins - baselineWins) * 100.0 / baselineWins

        println("BASELINE (before changes):")
        println("  Wins: $baselineWins (${String.format("%.2f", baselineWinRate)}%)")
        println("  Avg Score: ${String.format("%.1f", baselineScore.toDouble() / testSeeds.count())}")
        println()
        println("OPTIMIZED (with user insights):")
        println("  Wins: $optimizedWins (${String.format("%.2f", optimizedWinRate)}%)")
        println("  Avg Score: ${String.format("%.1f", optimizedScore.toDouble() / testSeeds.count())}")
        println()
        println("IMPROVEMENT: ${String.format("%.0f", improvement)}% more wins ($baselineWins → $optimizedWins)")
    }

    @Test
    fun `test on user's winning seed`() {
        val seed = 1768180694612L

        println("=== SEED $seed ===")

        val optimizedGenome = PlayerGenome(
            skipIfDamageExceedsHealthMinus = 5,
            skipWithoutWeaponDamageFraction = 0.444,
            skipDamageHealthFraction = 0.4,
            monsterLeavePenaltyMultiplier = 0.894,
            weaponLeavePenaltyIfNeeded = 2.505,
            potionLeavePenaltyPerRemaining = 0.5,
            weaponPreservationThreshold = 10,
            minDamageSavedToUseWeapon = 0,
            emergencyHealthBuffer = 0,
            equipFreshWeaponIfDegradedBelow = 10,
            alwaysSwapToFreshIfDegradedBelow = 8,
        )

        val player = ParameterizedPlayer(optimizedGenome)
        val result = player.playGame(GameState.newGame(Random(seed)))

        println("Health: ${result.health}")
        println("Score: ${result.calculateScore()}")
        println("Won: ${result.isGameWon}")
        println("Deck remaining: ${result.deck.cards.size}")
    }
}
