package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for potion mechanics.
 *
 * Potion rules:
 * - Potions restore health by their value
 * - Health is capped at MAX_HEALTH (20)
 * - Only the FIRST potion per turn can be used
 * - Second potion in same turn is discarded without effect
 * - Potion flag resets when starting a new turn (drawing room)
 */
class PotionTest {
    @Test
    fun `using potion restores health`() {
        val game = GameState.newGame().copy(health = 10)
        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥ (restore 5)

        val afterPotion = game.usePotion(potion)

        assertEquals(15, afterPotion.health, "Should restore 5 health: 10 + 5 = 15")
    }

    @Test
    fun `using potion caps health at max`() {
        val game = GameState.newGame().copy(health = 18)
        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥ (restore 5)

        val afterPotion = game.usePotion(potion)

        assertEquals(20, afterPotion.health, "Health should cap at 20, not 23")
    }

    @Test
    fun `using potion at full health keeps health at 20`() {
        val game = GameState.newGame().copy(health = 20)
        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥

        val afterPotion = game.usePotion(potion)

        assertEquals(20, afterPotion.health, "Health should stay at 20")
    }

    @Test
    fun `first potion in turn sets usedPotionThisTurn flag`() {
        val game =
            GameState.newGame().copy(
                health = 10,
                usedPotionThisTurn = false,
            )
        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥

        val afterPotion = game.usePotion(potion)

        assertTrue(afterPotion.usedPotionThisTurn, "Should mark potion as used this turn")
    }

    @Test
    fun `second potion in turn is discarded without effect`() {
        val game =
            GameState.newGame().copy(
                health = 10,
                // Already used a potion this turn
                usedPotionThisTurn = true,
            )
        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥

        val afterPotion = game.usePotion(potion)

        assertEquals(10, afterPotion.health, "Second potion should NOT restore health")
        assertTrue(afterPotion.usedPotionThisTurn, "Flag should still be true")
    }

    @Test
    fun `second potion with different value still has no effect`() {
        val game =
            GameState.newGame().copy(
                health = 5,
                usedPotionThisTurn = true,
            )
        val bigPotion = Card(Suit.HEARTS, Rank.TEN) // 10♥ (would restore 10)

        val afterPotion = game.usePotion(bigPotion)

        assertEquals(5, afterPotion.health, "Even a large second potion has no effect")
    }

    @Test
    fun `new turn resets potion flag`() {
        val game =
            GameState.newGame().copy(
                health = 10,
                // Used potion in previous turn
                usedPotionThisTurn = true,
            )

        val newTurn = game.drawRoom()

        assertFalse(newTurn.usedPotionThisTurn, "Drawing new room should reset potion flag")
    }

    @Test
    fun `can use potion again after new turn`() {
        var game = GameState.newGame().copy(health = 5)

        // Turn 1: Use first potion
        val potion1 = Card(Suit.HEARTS, Rank.FIVE) // 5♥
        game = game.usePotion(potion1)
        assertEquals(10, game.health)
        assertTrue(game.usedPotionThisTurn)

        // Start new turn
        game = game.drawRoom()
        assertFalse(game.usedPotionThisTurn)

        // Turn 2: Can use another potion
        val potion2 = Card(Suit.HEARTS, Rank.SEVEN) // 7♥
        game = game.usePotion(potion2)
        assertEquals(17, game.health, "Should restore health in new turn")
        assertTrue(game.usedPotionThisTurn)
    }

    @Test
    fun `using max value potion`() {
        val game = GameState.newGame().copy(health = 1)
        val maxPotion = Card(Suit.HEARTS, Rank.TEN) // 10♥ (max potion value)

        val afterPotion = game.usePotion(maxPotion)

        assertEquals(11, afterPotion.health, "Should restore 10 health: 1 + 10 = 11")
    }

    @Test
    fun `using min value potion`() {
        val game = GameState.newGame().copy(health = 15)
        val minPotion = Card(Suit.HEARTS, Rank.TWO) // 2♥ (min potion value)

        val afterPotion = game.usePotion(minPotion)

        assertEquals(17, afterPotion.health, "Should restore 2 health: 15 + 2 = 17")
    }

    @Test
    fun `cannot use non-potion card as potion`() {
        val game = GameState.newGame().copy(health = 10)
        val monster = Card(Suit.SPADES, Rank.FIVE) // 5♠ (monster)
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦ (weapon)

        assertFailsWith<IllegalArgumentException> {
            game.usePotion(monster)
        }

        assertFailsWith<IllegalArgumentException> {
            game.usePotion(weapon)
        }
    }

    @Test
    fun `new game starts with potion available`() {
        val game = GameState.newGame()

        assertFalse(game.usedPotionThisTurn, "New game should allow potion use")
    }

    @Test
    fun `avoiding room does not reset potion flag`() {
        var game = GameState.newGame()
        game = game.drawRoom() // Draw a room first
        game = game.copy(usedPotionThisTurn = true) // Simulate using potion in this turn
        game = game.avoidRoom() // Avoid the room

        // Avoiding a room is NOT starting a new turn, so flag should stay true
        // The potion flag only resets when we DRAW a new room (new turn)
        assertTrue(game.usedPotionThisTurn, "Avoiding room should not reset potion flag")
    }

    @Test
    fun `potion flag resets after avoiding and drawing new room`() {
        var game = GameState.newGame().copy(usedPotionThisTurn = true)
        game = game.drawRoom()
        game = game.avoidRoom()
        game = game.drawRoom()

        assertFalse(game.usedPotionThisTurn, "Drawing new room after avoid should reset flag")
    }

    @Test
    fun `realistic scenario - two potions in same room`() {
        var game = GameState.newGame().copy(health = 5)

        // Room has 2 potions among other cards
        val potion1 = Card(Suit.HEARTS, Rank.FIVE) // 5♥
        val potion2 = Card(Suit.HEARTS, Rank.SEVEN) // 7♥

        // Use first potion
        game = game.usePotion(potion1)
        assertEquals(10, game.health, "First potion works")

        // Try to use second potion in SAME turn
        game = game.usePotion(potion2)
        assertEquals(10, game.health, "Second potion has no effect")
    }
}
