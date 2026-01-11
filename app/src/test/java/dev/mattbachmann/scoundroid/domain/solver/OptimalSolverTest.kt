package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Deck
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class OptimalSolverTest {

    private val solver = OptimalSolver()

    private fun gameWithDeck(cards: List<Card>, health: Int = 20): GameState {
        return GameState(
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
    }

    @Test
    fun `empty deck is immediate win`() {
        val game = gameWithDeck(emptyList(), health = 15)
        val result = solver.solve(game)

        assertTrue(result.isWinnable)
        assertEquals(15, result.bestScore)
    }

    @Test
    fun `guaranteed win with all potions`() {
        val game = gameWithDeck(
            listOf(
                Card(Suit.HEARTS, Rank.TWO),
                Card(Suit.HEARTS, Rank.THREE),
                Card(Suit.HEARTS, Rank.FOUR),
                Card(Suit.HEARTS, Rank.FIVE),
            ),
            health = 20
        )
        val result = solver.solve(game, findBestScore = true)

        assertTrue(result.isWinnable)
        assertNotNull(result.bestScore)
        println("All potions: best score = ${result.bestScore}, nodes = ${result.nodesExplored}")
    }

    @Test
    fun `guaranteed loss with lethal monsters`() {
        val game = gameWithDeck(
            listOf(
                Card(Suit.CLUBS, Rank.ACE),   // 14
                Card(Suit.SPADES, Rank.ACE),  // 14
                Card(Suit.CLUBS, Rank.KING),  // 13
                Card(Suit.SPADES, Rank.KING), // 13
            ),
            health = 5
        )
        val result = solver.solve(game)

        assertFalse(result.isWinnable)
        println("Lethal monsters: nodes explored = ${result.nodesExplored}")
    }

    @Test
    fun `winnable with weapon before monster`() {
        val game = gameWithDeck(
            listOf(
                Card(Suit.DIAMONDS, Rank.TEN), // Weapon value 10
                Card(Suit.CLUBS, Rank.JACK),   // Monster value 11
                Card(Suit.HEARTS, Rank.FIVE),  // Potion
                Card(Suit.HEARTS, Rank.SIX),   // Potion
            ),
            health = 5
        )
        val result = solver.solve(game, findBestScore = true)

        // With weapon: 11 - 10 = 1 damage, then heal
        assertTrue(result.isWinnable)
        println("Weapon scenario: best score = ${result.bestScore}, nodes = ${result.nodesExplored}")
    }

    @Test
    fun `finds win even when order matters`() {
        // Only winnable if weapon is equipped before fighting the ace
        val game = gameWithDeck(
            listOf(
                Card(Suit.CLUBS, Rank.ACE),    // 14 damage barehanded
                Card(Suit.DIAMONDS, Rank.TEN), // Weapon value 10
                Card(Suit.HEARTS, Rank.TWO),   // Potion
                Card(Suit.HEARTS, Rank.THREE), // Potion
            ),
            health = 10
        )
        val result = solver.solve(game, findBestScore = true)

        // Optimal: leave ace, equip weapon, heal, then fight ace (14-10=4 damage)
        // Or: equip weapon, fight ace (4 damage), heal
        assertTrue(result.isWinnable)
        println("Order matters: best score = ${result.bestScore}, nodes = ${result.nodesExplored}")
    }

    @Test
    fun `respects node limit`() {
        val game = GameState.newGame(Random(42))
        val result = solver.solve(game, maxNodes = 1000)

        println("Node-limited (1000): winnable = ${result.isWinnable}, nodes = ${result.nodesExplored}")
        assertTrue(result.nodesExplored <= 1000)
    }

    @Test
    fun `solve real seed 42 with limit`() {
        val game = GameState.newGame(Random(42))
        val result = solver.solve(game, maxNodes = 100_000)

        println("\n=== Seed 42 Optimal Solve (100k node limit) ===")
        println("Winnable: ${result.isWinnable}")
        println("Best score: ${result.bestScore}")
        println("Nodes explored: ${result.nodesExplored}")
        println("===============================================\n")
    }

    @Test
    fun `solve multiple seeds with limit`() {
        println("\n=== Multi-Seed Optimal Solve (1M nodes each) ===")
        var winnableCount = 0
        var unknownCount = 0

        for (seed in 1L..10L) {
            val game = GameState.newGame(Random(seed))
            val result = solver.solve(game, maxNodes = 1_000_000)

            val status = when {
                result.isWinnable -> "WIN"
                result.nodesExplored >= 1_000_000 -> "???"
                else -> "LOSS"
            }

            when (status) {
                "WIN" -> winnableCount++
                "???" -> unknownCount++
            }

            println("Seed $seed: $status (nodes: ${result.nodesExplored}, score: ${result.bestScore ?: "N/A"})")
        }

        println("Confirmed winnable: $winnableCount / 10")
        println("Unknown (hit limit): $unknownCount / 10")
        println("=================================================\n")
    }
}
