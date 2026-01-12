package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import org.junit.Test
import kotlin.random.Random

class WeaponPreservationTest {
    @Test
    fun `test weapon preservation threshold`() {
        val testSeeds = 0L..10000L
        // Current is 9 - only use fresh weapon on monsters >= 9
        // Lower values mean using weapon on smaller monsters (less preservation)
        val thresholds = listOf(2, 4, 6, 7, 8, 9, 10, 12)

        println("=== Weapon Preservation Threshold Comparison ===")
        println("Current: 9 (only use fresh weapon on monsters >= 9)")
        println("Lower = use weapon more freely, Higher = preserve weapon more")
        println()

        for (threshold in thresholds) {
            var wins = 0
            var totalScore = 0L

            val genome = PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                skipDamageHealthFraction = 0.4,
                monsterLeavePenaltyMultiplier = 0.894,
                weaponLeavePenaltyIfNeeded = 2.505,
                potionLeavePenaltyPerRemaining = 0.5,
                weaponPreservationThreshold = threshold,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
                alwaysSwapToFreshIfDegradedBelow = 8,
            )
            val player = ParameterizedPlayer(genome)

            for (seed in testSeeds) {
                val game = GameState.newGame(Random(seed))
                val result = player.playGame(game)
                if (result.isGameWon) wins++
                totalScore += result.calculateScore()
            }

            val winRate = wins * 100.0 / testSeeds.count()
            val avgScore = totalScore.toDouble() / testSeeds.count()
            println(
                "Threshold $threshold: $wins wins (${String.format("%.2f", winRate)}%), avg score ${String.format("%.1f", avgScore)}"
            )
        }
    }

    @Test
    fun `test on seed 1768180694612 with different thresholds`() {
        val seed = 1768180694612L
        val thresholds = listOf(2, 6, 9, 12)

        println("=== Seed $seed with different weapon preservation ===")

        for (threshold in thresholds) {
            val genome = PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                skipDamageHealthFraction = 0.4,
                monsterLeavePenaltyMultiplier = 0.894,
                weaponLeavePenaltyIfNeeded = 2.505,
                potionLeavePenaltyPerRemaining = 0.5,
                weaponPreservationThreshold = threshold,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
                alwaysSwapToFreshIfDegradedBelow = 8,
            )
            val player = ParameterizedPlayer(genome)
            val result = player.playGame(GameState.newGame(Random(seed)))

            println(
                "Threshold $threshold: health=${result.health}, score=${result.calculateScore()}, won=${result.isGameWon}, deck=${result.deck.cards.size}"
            )
        }
    }
}
