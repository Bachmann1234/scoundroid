package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class GameStateTest {
    @Test
    fun `new game should start with full deck`() {
        val gameState = GameState.newGame()
        assertEquals(44, gameState.deck.size)
    }

    @Test
    fun `new game should start with 20 health`() {
        val gameState = GameState.newGame()
        assertEquals(20, gameState.health)
    }

    @Test
    fun `new game should have no current room`() {
        val gameState = GameState.newGame()
        assertNull(gameState.currentRoom)
    }

    @Test
    fun `new game should have no equipped weapon`() {
        val gameState = GameState.newGame()
        assertNull(gameState.weaponState)
    }

    @Test
    fun `new game should allow room avoidance`() {
        val gameState = GameState.newGame()
        assertTrue(gameState.canAvoidRoom)
    }

    @Test
    fun `drawing room should create room with 4 cards`() {
        val gameState = GameState.newGame()
        val newState = gameState.drawRoom()

        assertEquals(4, newState.currentRoom?.size ?: 0)
        assertEquals(40, newState.deck.size)
    }

    @Test
    fun `drawing room from deck with less than 4 cards should use all remaining cards`() {
        val smallDeck =
            Deck(
                listOf(
                    Card(Suit.CLUBS, Rank.TWO),
                    Card(Suit.SPADES, Rank.THREE),
                ),
            )
        val gameState =
            GameState(
                deck = smallDeck,
                health = 20,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                canAvoidRoom = true,
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
            )
        val newState = gameState.drawRoom()

        assertEquals(2, newState.currentRoom?.size ?: 0)
        assertEquals(0, newState.deck.size)
    }

    @Test
    fun `avoiding room should move all 4 cards to bottom of deck`() {
        val gameState = GameState.newGame().drawRoom()
        val originalRoomCards = gameState.currentRoom!!

        val newState = gameState.avoidRoom()

        assertNull(newState.currentRoom)
        assertEquals(44, newState.deck.size)

        // The avoided cards should be at the bottom
        val bottomFour = newState.deck.cards.takeLast(4)
        assertEquals(originalRoomCards.toSet(), bottomFour.toSet())
    }

    @Test
    fun `avoiding room should set canAvoidRoom to false`() {
        val gameState = GameState.newGame().drawRoom()
        val newState = gameState.avoidRoom()

        assertFalse(newState.canAvoidRoom)
    }

    @Test
    fun `avoiding room should set lastRoomAvoided to true`() {
        val gameState = GameState.newGame().drawRoom()
        val newState = gameState.avoidRoom()

        assertTrue(newState.lastRoomAvoided)
    }

    @Test
    fun `cannot avoid room twice in a row`() {
        val gameState =
            GameState.newGame()
                .drawRoom()
                .avoidRoom()

        assertFalse(gameState.canAvoidRoom)
    }

    @Test
    fun `processing room should restore room avoidance ability`() {
        val gameState =
            GameState.newGame()
                .drawRoom()
                .avoidRoom()
                .drawRoom()

        // After drawing a new room (which means we didn't avoid), we should be able to avoid again
        assertTrue(gameState.canAvoidRoom)
    }

    @Test
    fun `selecting 3 cards should leave 1 for next room`() {
        val gameState = GameState.newGame().drawRoom()
        val roomCards = gameState.currentRoom!!

        val selectedCards = roomCards.take(3)
        val newState = gameState.selectCards(selectedCards)

        // The unselected card should remain
        val expectedRemaining = roomCards[3]
        assertEquals(1, newState.currentRoom?.size ?: 0)
        assertEquals(expectedRemaining, newState.currentRoom?.first())
    }

    @Test
    fun `health should decrease when taking damage`() {
        val gameState = GameState.newGame()
        val newState = gameState.takeDamage(5)

        assertEquals(15, newState.health)
    }

    @Test
    fun `health should not go below zero`() {
        val gameState = GameState.newGame()
        val newState = gameState.takeDamage(25)

        assertEquals(0, newState.health)
    }

    @Test
    fun `healing should increase health`() {
        val gameState = GameState.newGame().takeDamage(10)
        val newState = gameState.heal(5)

        assertEquals(15, newState.health)
    }

    @Test
    fun `healing should not exceed max health of 20`() {
        val gameState = GameState.newGame()
        val newState = gameState.heal(10)

        assertEquals(20, newState.health)
    }

    @Test
    fun `healing from 18 with potion 5 should cap at 20`() {
        val gameState = GameState.newGame().takeDamage(2)
        val newState = gameState.heal(5)

        assertEquals(20, newState.health)
    }

    @Test
    fun `game should be over when health reaches 0`() {
        val gameState = GameState.newGame().takeDamage(20)

        assertTrue(gameState.isGameOver)
    }

    @Test
    fun `game should not be over with health above 0`() {
        val gameState = GameState.newGame().takeDamage(10)

        assertFalse(gameState.isGameOver)
    }

    @Test
    fun `game should be won when deck is empty and health above 0`() {
        val emptyDeck = Deck(emptyList())
        val gameState =
            GameState(
                deck = emptyDeck,
                health = 10,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                canAvoidRoom = true,
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
            )

        assertTrue(gameState.isGameWon)
    }

    @Test
    fun `game should not be won with cards remaining in deck`() {
        val gameState = GameState.newGame()

        assertFalse(gameState.isGameWon)
    }

    @Test
    fun `equipping weapon should replace current weapon`() {
        val weapon1 = Card(Suit.DIAMONDS, Rank.FIVE)
        val weapon2 = Card(Suit.DIAMONDS, Rank.SEVEN)

        val gameState =
            GameState.newGame()
                .equipWeapon(weapon1)
                .equipWeapon(weapon2)

        assertEquals(weapon2, gameState.weaponState?.weapon)
    }
}
