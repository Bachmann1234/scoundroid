package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import kotlin.math.pow
import org.junit.Test
import kotlin.random.Random

class MonsterPenaltyTest {
    @Test
    fun `test monster leave penalty multiplier`() {
        val testSeeds = 0L..10000L
        // Current is 0.894 (sublinear - big monsters penalized less proportionally)
        // Try superlinear values (big monsters penalized MORE)
        val multipliers = listOf(0.894, 1.0, 1.2, 1.5, 2.0, 2.5)

        println("=== Monster Leave Penalty Multiplier Comparison ===")
        println("Current: 0.894 (sublinear)")
        println("Testing: superlinear values penalize big monsters more")
        println()
        println("Example penalties for Ace (14):")
        for (m in multipliers) {
            val penalty = 14.0.pow(m)
            println("  multiplier $m: penalty = ${String.format("%.1f", penalty)}")
        }
        println()

        for (multiplier in multipliers) {
            var wins = 0
            var totalScore = 0L

            val genome = PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                skipDamageHealthFraction = 0.4,
                monsterLeavePenaltyMultiplier = multiplier,
                weaponLeavePenaltyIfNeeded = 2.505,
                potionLeavePenaltyPerRemaining = 0.5,
                weaponPreservationThreshold = 9,
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
                "Multiplier $multiplier: $wins wins (${String.format("%.2f", winRate)}%), avg score ${String.format("%.1f", avgScore)}"
            )
        }
    }

    @Test
    fun `test on seed 1768180694612 with different multipliers`() {
        val seed = 1768180694612L
        val multipliers = listOf(0.894, 1.5, 2.0, 2.5)

        println("=== Seed $seed with different monster penalties ===")

        for (multiplier in multipliers) {
            val genome = PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                skipDamageHealthFraction = 0.4,
                monsterLeavePenaltyMultiplier = multiplier,
                weaponLeavePenaltyIfNeeded = 2.505,
                potionLeavePenaltyPerRemaining = 0.5,
                weaponPreservationThreshold = 9,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
                alwaysSwapToFreshIfDegradedBelow = 8,
            )
            val player = ParameterizedPlayer(genome)
            val result = player.playGame(GameState.newGame(Random(seed)))

            println(
                "Multiplier $multiplier: health=${result.health}, score=${result.calculateScore()}, won=${result.isGameWon}, deck=${result.deck.cards.size}"
            )
        }
    }
}
