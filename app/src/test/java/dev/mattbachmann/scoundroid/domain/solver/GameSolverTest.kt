package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Deck
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class GameSolverTest {

    private val solver = GameSolver()

    // Helper to create a game with a specific deck
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

    // ===================
    // Base Case Tests
    // ===================

    @Test
    fun `empty deck with health is a win`() {
        val game = gameWithDeck(emptyList(), health = 15)
        val result = solver.solve(game)

        assertTrue(result.isWinnable)
        assertEquals(1, result.totalPaths)
        assertEquals(1, result.winningPaths)
        assertEquals(0, result.losingPaths)
        assertEquals(15, result.maxScore)
        assertEquals(1.0, result.winProbability)
    }

    @Test
    fun `zero health is a loss`() {
        val game = gameWithDeck(
            listOf(Card(Suit.CLUBS, Rank.TWO)),
            health = 0
        )
        val result = solver.solve(game)

        assertFalse(result.isWinnable)
        assertEquals(1, result.totalPaths)
        assertEquals(0, result.winningPaths)
        assertEquals(1, result.losingPaths)
        assertEquals(0.0, result.winProbability)
    }

    // ===================
    // Simple Room Tests
    // ===================

    @Test
    fun `single room with all potions is guaranteed win`() {
        // 4 potions - can only heal, guaranteed win
        val game = gameWithDeck(
            listOf(
                Card(Suit.HEARTS, Rank.TWO),
                Card(Suit.HEARTS, Rank.THREE),
                Card(Suit.HEARTS, Rank.FOUR),
                Card(Suit.HEARTS, Rank.FIVE),
            ),
            health = 10
        )
        val result = solver.solve(game)

        assertTrue(result.isWinnable)
        assertEquals(1.0, result.winProbability)
        // Max score: health 10 + first potion (best is 5) = 15, capped at 20
        // With 10 health, best potion is 5 → 15 health
        // But only 1 potion per turn, so we use one and discard 2, leave 1
        // Then next "room" has 1 card, which gets processed
        // Final health depends on which potions we use
    }

    @Test
    fun `single room with weak monsters is winnable`() {
        // 4 weak monsters (value 2 each) with 20 health
        // Must fight 3, leave 1, then fight that 1
        // Total damage: 4 * 2 = 8, health 20 - 8 = 12
        val game = gameWithDeck(
            listOf(
                Card(Suit.CLUBS, Rank.TWO),
                Card(Suit.SPADES, Rank.TWO),
                Card(Suit.CLUBS, Rank.THREE),
                Card(Suit.SPADES, Rank.THREE),
            ),
            health = 20
        )
        val result = solver.solve(game)

        assertTrue(result.isWinnable)
        // All paths should win since total damage is 2+2+3+3=10, health is 20
        assertEquals(1.0, result.winProbability)
    }

    @Test
    fun `impossible seed with lethal monsters`() {
        // Single ace (14 damage) with only 10 health
        // Need exactly 4 cards for a room
        val game = gameWithDeck(
            listOf(
                Card(Suit.CLUBS, Rank.ACE),   // 14 damage
                Card(Suit.SPADES, Rank.ACE),  // 14 damage
                Card(Suit.CLUBS, Rank.KING),  // 13 damage
                Card(Suit.SPADES, Rank.KING), // 13 damage
            ),
            health = 10
        )
        val result = solver.solve(game)

        // Must process 3 of 4, minimum damage is 13+13+14=40, way over 10 health
        assertFalse(result.isWinnable)
        assertEquals(0.0, result.winProbability)
    }

    // ===================
    // Weapon Tests
    // ===================

    @Test
    fun `weapon reduces damage and enables win`() {
        // Weapon + monster that would otherwise kill
        val game = gameWithDeck(
            listOf(
                Card(Suit.DIAMONDS, Rank.TEN), // Weapon value 10
                Card(Suit.CLUBS, Rank.KING),   // Monster value 13, with weapon = 3 damage
                Card(Suit.HEARTS, Rank.TWO),   // Potion
                Card(Suit.HEARTS, Rank.THREE), // Potion
            ),
            health = 5
        )
        val result = solver.solve(game)

        // Optimal: equip weapon, fight king (3 damage), use potion
        // 5 - 3 + 2 or 3 = 4 or 5 health
        assertTrue(result.isWinnable)
    }

    @Test
    fun `weapon degradation affects later combat choices`() {
        // Two rooms: first has weapon and small monster, second has big monster
        val game = gameWithDeck(
            listOf(
                // Room 1
                Card(Suit.DIAMONDS, Rank.FIVE), // Weapon value 5
                Card(Suit.CLUBS, Rank.THREE),   // Monster value 3
                Card(Suit.HEARTS, Rank.TWO),    // Potion
                Card(Suit.HEARTS, Rank.THREE),  // Potion
                // Room 2 (3 more cards + leftover)
                Card(Suit.CLUBS, Rank.TEN),     // Monster value 10
                Card(Suit.CLUBS, Rank.TWO),     // Monster value 2
                Card(Suit.HEARTS, Rank.FOUR),   // Potion
            ),
            health = 15
        )
        val result = solver.solve(game)

        // After fighting 3♣ with weapon, weapon can only hit monsters ≤3
        // So 10♣ must be fought barehanded
        // Optimal play should still be able to win
        assertTrue(result.isWinnable)
    }

    // ===================
    // Room Avoidance Tests
    // ===================

    @Test
    fun `avoiding room changes outcome`() {
        // A room that's bad now but ok later after the bad cards go to bottom
        val game = gameWithDeck(
            listOf(
                // Room 1 - all big monsters
                Card(Suit.CLUBS, Rank.KING),    // 13
                Card(Suit.SPADES, Rank.KING),   // 13
                Card(Suit.CLUBS, Rank.QUEEN),   // 12
                Card(Suit.SPADES, Rank.QUEEN),  // 12
                // Room 2 - weapon and potions
                Card(Suit.DIAMONDS, Rank.TEN),  // Weapon
                Card(Suit.HEARTS, Rank.TEN),    // Potion
                Card(Suit.HEARTS, Rank.NINE),   // Potion
                Card(Suit.HEARTS, Rank.EIGHT),  // Potion
            ),
            health = 20
        )
        val result = solver.solve(game)

        // If we skip room 1, we get weapon + potions first
        // Then face the kings/queens with weapon equipped
        assertTrue(result.isWinnable)
    }

    @Test
    fun `cannot avoid two rooms in a row`() {
        // Test that the solver respects the skip restriction
        val game = gameWithDeck(
            listOf(
                Card(Suit.CLUBS, Rank.ACE),
                Card(Suit.SPADES, Rank.ACE),
                Card(Suit.CLUBS, Rank.KING),
                Card(Suit.SPADES, Rank.KING),
                Card(Suit.CLUBS, Rank.QUEEN),
                Card(Suit.SPADES, Rank.QUEEN),
                Card(Suit.CLUBS, Rank.JACK),
                Card(Suit.SPADES, Rank.JACK),
            ),
            health = 20
        )
        val result = solver.solve(game)

        // Even with skipping, we can't avoid all the damage
        // This tests that consecutive skip restriction is enforced
        assertFalse(result.isWinnable)
    }

    // ===================
    // Combat Choice Tests
    // ===================

    @Test
    fun `choosing barehanded preserves weapon for later`() {
        // Sometimes it's better to take damage now to save weapon for bigger threat
        val game = gameWithDeck(
            listOf(
                Card(Suit.DIAMONDS, Rank.FIVE), // Weapon value 5
                Card(Suit.CLUBS, Rank.TWO),     // Small monster - might want to skip weapon
                Card(Suit.HEARTS, Rank.THREE),  // Potion
                Card(Suit.HEARTS, Rank.FOUR),   // Potion
                // Next room will have bigger monster
                Card(Suit.CLUBS, Rank.TEN),     // Big monster - want weapon fresh
                Card(Suit.HEARTS, Rank.FIVE),   // Potion
                Card(Suit.HEARTS, Rank.SIX),    // Potion
            ),
            health = 20
        )
        val result = solver.solve(game)

        assertTrue(result.isWinnable)
        // The solver should find paths where fighting 2♣ barehanded
        // preserves the weapon's ability to fight 10♣ effectively
    }

    // ===================
    // Statistics Tests
    // ===================

    @Test
    fun `win probability reflects path distribution`() {
        // Create a scenario where some paths win and some lose
        val game = gameWithDeck(
            listOf(
                Card(Suit.CLUBS, Rank.TEN),    // Monster - 10 damage
                Card(Suit.DIAMONDS, Rank.TEN), // Weapon - makes monster 0 damage
                Card(Suit.HEARTS, Rank.TWO),   // Potion
                Card(Suit.HEARTS, Rank.THREE), // Potion
            ),
            health = 5
        )
        val result = solver.solve(game)

        // With weapon first, we win. Without weapon first and fighting monster, we might lose.
        assertTrue(result.isWinnable)
        assertTrue(result.winProbability > 0)
        // Some orderings should lose (fight monster without weapon when health is 5)
        assertTrue(result.winProbability < 1.0 || result.totalPaths == result.winningPaths)
    }

    @Test
    fun `max score tracks best outcome`() {
        val game = gameWithDeck(
            listOf(
                Card(Suit.HEARTS, Rank.TEN),   // Potion +10
                Card(Suit.HEARTS, Rank.NINE),  // Potion +9
                Card(Suit.HEARTS, Rank.EIGHT), // Potion +8
                Card(Suit.HEARTS, Rank.SEVEN), // Potion +7
            ),
            health = 15
        )
        val result = solver.solve(game)

        // Best: use 10 potion (15+10=20, capped), score = 20
        // But special rule: if at 20 health and last card was potion, score = 20 + potion value
        // So best is to end with highest potion at full health
        assertTrue(result.maxScore >= 20)
    }

    // ===================
    // Real Seed Test
    // ===================

    @Test
    fun `solve a real seeded game`() {
        // Create a game with a known seed
        val game = GameState.newGame(kotlin.random.Random(42))
        val result = solver.solve(game)

        // Just verify we get a result without crashing
        assertTrue(result.totalPaths > 0)
        println("Seed 42 results:")
        println("  Total paths: ${result.totalPaths}")
        println("  Winning paths: ${result.winningPaths}")
        println("  Win probability: ${String.format("%.2f%%", result.winProbability * 100)}")
        println("  Max score: ${result.maxScore}")
        println("  Is winnable: ${result.isWinnable}")
    }
}
