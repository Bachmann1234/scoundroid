package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.GameState
import org.junit.Test
import kotlin.random.Random

class SeedAnalysisTest {
    @Test
    fun `analyze seed 1768180694612`() {
        val seed = 1768180694612L
        val game = GameState.newGame(Random(seed))

        println("=== Analyzing Seed $seed ===")
        println()

        // Show all cards in deck order
        val deckCards = game.deck.cards.map { "${it.rank.displayName}${it.suit.symbol}" }
        println("Deck (top to bottom): $deckCards")
        println()

        // Try with HeuristicPlayer
        val player = HeuristicPlayer()
        val finalState = player.playGame(game)

        println("\n=== HeuristicPlayer Result ===")
        println("Final Health: ${finalState.health}")
        println("Score: ${finalState.calculateScore()}")
        println("Won: ${finalState.isGameWon}")
        println("Game Over: ${finalState.isGameOver}")
        println("Deck remaining: ${finalState.deck.cards.size}")
        println("Room remaining: ${finalState.currentRoom?.size ?: 0}")

        // Also try with ParameterizedPlayer using best evolved genome
        val bestGenome =
            PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                monsterLeavePenaltyMultiplier = 2.5,
                weaponLeavePenaltyIfNeeded = 9.0,
                weaponPreservationThreshold = 9,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
            )

        val paramPlayer = ParameterizedPlayer(bestGenome)
        val paramFinal = paramPlayer.playGame(GameState.newGame(Random(seed)))

        println("\n=== ParameterizedPlayer (Best GA) Result ===")
        println("Final Health: ${paramFinal.health}")
        println("Score: ${paramFinal.calculateScore()}")
        println("Won: ${paramFinal.isGameWon}")
    }
}
