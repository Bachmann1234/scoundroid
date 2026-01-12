package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Deck
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MonteCarloSimulatorTest {
    private val simulator = MonteCarloSimulator()

    // Helper to create a game with a specific deck
    private fun gameWithDeck(
        cards: List<Card>,
        health: Int = 20,
    ): GameState =
        GameState(
            deck = Deck(cards),
            health = health,
            currentRoom = null,
            weaponState = null,
            defeatedMonsters = emptyList(),
            discardPile = emptyList(),
            lastRoomAvoided = false,
            usedPotionThisTurn = false,
            lastCardProcessed = null,
        )

    @Test
    fun `guaranteed win scenario has 100% win rate`() {
        // All potions - can't lose
        val game =
            gameWithDeck(
                listOf(
                    Card(Suit.HEARTS, Rank.TWO),
                    Card(Suit.HEARTS, Rank.THREE),
                    Card(Suit.HEARTS, Rank.FOUR),
                    Card(Suit.HEARTS, Rank.FIVE),
                ),
                health = 20,
            )

        val result = simulator.simulate(game, samples = 100)

        assertEquals(100, result.wins)
        assertEquals(0, result.losses)
        assertEquals(1.0, result.winProbability)
    }

    @Test
    fun `guaranteed loss scenario has 0% win rate`() {
        // Lethal monsters with low health
        // Total damage: 14+14+13+13 = 54, health = 5
        val game =
            gameWithDeck(
                listOf(
                    Card(Suit.CLUBS, Rank.ACE), // 14
                    Card(Suit.SPADES, Rank.ACE), // 14
                    Card(Suit.CLUBS, Rank.KING), // 13
                    Card(Suit.SPADES, Rank.KING), // 13
                ),
                health = 5,
            )

        val result = simulator.simulate(game, samples = 100)

        println("Guaranteed loss: wins=${result.wins}, losses=${result.losses}")
        assertEquals(0.0, result.winProbability, 0.001)
    }

    @Test
    fun `mixed scenario runs without errors`() {
        // Survivable with optimal play, but random play is bad
        val game =
            gameWithDeck(
                listOf(
                    Card(Suit.DIAMONDS, Rank.TEN), // Weapon
                    Card(Suit.CLUBS, Rank.JACK), // 11 damage
                    Card(Suit.HEARTS, Rank.FIVE), // Potion
                    Card(Suit.HEARTS, Rank.SIX), // Potion
                ),
                health = 10,
            )

        val result = simulator.simulate(game, samples = 1000)

        println("Mixed scenario: wins=${result.wins}, losses=${result.losses}, winRate=${result.winProbability}")
        // Just verify simulation completes
        assertEquals(1000, result.wins + result.losses)
    }

    @Test
    fun `easy scenario with weak monsters has high win rate`() {
        // Very easy: weak monsters, lots of health
        val game =
            gameWithDeck(
                listOf(
                    Card(Suit.CLUBS, Rank.TWO), // 2 damage
                    Card(Suit.SPADES, Rank.TWO), // 2 damage
                    Card(Suit.CLUBS, Rank.THREE), // 3 damage
                    Card(Suit.SPADES, Rank.THREE), // 3 damage
                ),
                health = 20,
            )

        val result = simulator.simulate(game, samples = 100)

        // Total damage is 10, health is 20 - should always win
        println("Easy scenario: wins=${result.wins}, losses=${result.losses}")
        assertEquals(100, result.wins)
        assertEquals(1.0, result.winProbability)
    }

    @Test
    fun `simulate real seed 42`() {
        val game = GameState.newGame(Random(42))
        val result = simulator.simulate(game, samples = 10000, random = Random(123))

        println("\n=== Seed 42 Simulation Results ===")
        println("Samples: ${result.samples}")
        println("Wins: ${result.wins}")
        println("Losses: ${result.losses}")
        println("Win probability: ${String.format("%.2f%%", result.winProbability * 100)}")
        println("Max score: ${result.maxScore}")
        println("Min score: ${result.minScore}")
        result.averageWinScore?.let { println("Avg win score: ${String.format("%.1f", it)}") }
        result.averageLossScore?.let { println("Avg loss score: ${String.format("%.1f", it)}") }
        println("==================================\n")

        assertTrue(result.samples == 10000)
    }

    @Test
    fun `simulate multiple seeds`() {
        val results =
            simulator.simulateMultipleSeeds(
                seedRange = 1L..10L,
                samplesPerSeed = 1000,
                random = Random(456),
            )

        println("\n=== Multi-Seed Simulation Results ===")
        results.forEach { (seed, result) ->
            println(
                "Seed $seed: ${String.format(
                    "%.1f%%",
                    result.winProbability * 100,
                )} win rate, max score ${result.maxScore}",
            )
        }

        val avgWinRate = results.values.map { it.winProbability }.average()
        println("Average win rate across seeds: ${String.format("%.1f%%", avgWinRate * 100)}")
        println("=====================================\n")

        assertEquals(10, results.size)
    }

    @Test
    fun `deterministic results with fixed random seed`() {
        val game = GameState.newGame(Random(42))

        val result1 = simulator.simulate(game, samples = 100, random = Random(999))
        val result2 = simulator.simulate(game, samples = 100, random = Random(999))

        // Same random seed should give same results
        assertEquals(result1.wins, result2.wins)
        assertEquals(result1.losses, result2.losses)
    }
}
