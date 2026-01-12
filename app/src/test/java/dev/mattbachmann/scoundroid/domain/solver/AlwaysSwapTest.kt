package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import org.junit.Test
import kotlin.random.Random

class AlwaysSwapTest {
    @Test
    fun `test alwaysSwapToFresh parameter`() {
        val testSeeds = 0L..10000L
        val thresholds = listOf(0, 6, 7, 8, 9, 10, 12)

        println("=== alwaysSwapToFreshIfDegradedBelow Comparison ===")
        println("(0 = disabled, swap to any fresh weapon if degraded below this value)")

        for (threshold in thresholds) {
            var wins = 0
            var totalScore = 0L

            val genome =
                PlayerGenome(
                    skipIfDamageExceedsHealthMinus = 5,
                    skipWithoutWeaponDamageFraction = 0.444,
                    skipDamageHealthFraction = 0.4,
                    monsterLeavePenaltyMultiplier = 0.894,
                    weaponLeavePenaltyIfNeeded = 2.505,
                    potionLeavePenaltyPerRemaining = 0.5,
                    weaponPreservationThreshold = 9,
                    minDamageSavedToUseWeapon = 0,
                    emergencyHealthBuffer = 0,
                    equipFreshWeaponIfDegradedBelow = 10,
                    alwaysSwapToFreshIfDegradedBelow = threshold,
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
                "Threshold $threshold: $wins wins (${
                    String.format(
                        "%.2f",
                        winRate,
                    )
                }%), avg score ${String.format("%.1f", avgScore)}",
            )
        }
    }
}
