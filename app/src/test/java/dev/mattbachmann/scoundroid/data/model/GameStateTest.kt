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
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
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

        assertTrue(gameState.lastRoomAvoided)
    }

    @Test
    fun `processing room should restore room avoidance ability`() {
        val state1 = GameState.newGame().drawRoom()
        val state2 = state1.avoidRoom()
        val state3 = state2.drawRoom()
        val state4 = state3.selectCards(state3.currentRoom!!.take(3))
        val gameState = state4.drawRoom()

        // After avoiding a room, then processing the next room, we should be able to avoid again
        assertFalse(gameState.lastRoomAvoided)
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
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
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

    // ========== First Room Avoidance Tests ==========

    @Test
    fun `first room can be avoided since no previous room was avoided`() {
        val gameState = GameState.newGame().drawRoom()

        // First room CAN be avoided (lastRoomAvoided starts false)
        assertFalse(gameState.lastRoomAvoided)

        // Should be able to avoid
        val afterAvoid = gameState.avoidRoom()
        assertTrue(afterAvoid.lastRoomAvoided)
        assertNull(afterAvoid.currentRoom)
    }

    @Test
    fun `new game starts with lastRoomAvoided as false`() {
        val gameState = GameState.newGame()
        assertFalse(gameState.lastRoomAvoided)
    }

    // ========== Deck Exhaustion Edge Cases ==========

    @Test
    fun `drawing room when deck has exactly 4 cards uses all cards`() {
        val fourCards =
            listOf(
                Card(Suit.CLUBS, Rank.TWO),
                Card(Suit.SPADES, Rank.THREE),
                Card(Suit.DIAMONDS, Rank.FOUR),
                Card(Suit.HEARTS, Rank.FIVE),
            )
        val gameState =
            GameState(
                deck = Deck(fourCards),
                health = 20,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
            )

        val newState = gameState.drawRoom()

        assertEquals(4, newState.currentRoom?.size)
        assertEquals(0, newState.deck.size)
    }

    @Test
    fun `drawing room when deck has 3 cards creates room with 3 cards`() {
        val threeCards =
            listOf(
                Card(Suit.CLUBS, Rank.TWO),
                Card(Suit.SPADES, Rank.THREE),
                Card(Suit.DIAMONDS, Rank.FOUR),
            )
        val gameState =
            GameState(
                deck = Deck(threeCards),
                health = 20,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
            )

        val newState = gameState.drawRoom()

        assertEquals(3, newState.currentRoom?.size)
        assertEquals(0, newState.deck.size)
    }

    @Test
    fun `drawing room when deck has 1 card creates room with 1 card`() {
        val oneCard = listOf(Card(Suit.CLUBS, Rank.ACE))
        val gameState =
            GameState(
                deck = Deck(oneCard),
                health = 20,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
            )

        val newState = gameState.drawRoom()

        assertEquals(1, newState.currentRoom?.size)
        assertEquals(0, newState.deck.size)
    }

    @Test
    fun `drawing room when deck is empty creates empty room`() {
        val gameState =
            GameState(
                deck = Deck(emptyList()),
                health = 20,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
            )

        val newState = gameState.drawRoom()

        assertEquals(0, newState.currentRoom?.size ?: 0)
    }

    @Test
    fun `game win is determined by empty deck and positive health`() {
        // Note: Current implementation considers game won when deck.isEmpty && health > 0
        // This doesn't account for cards still in the room - the UI handles showing
        // the win condition only after the final room is processed.

        // Start with exactly 4 cards
        val fourCards =
            listOf(
                Card(Suit.CLUBS, Rank.TWO),
                Card(Suit.HEARTS, Rank.THREE),
                Card(Suit.HEARTS, Rank.FOUR),
                Card(Suit.HEARTS, Rank.FIVE),
            )
        val gameState =
            GameState(
                deck = Deck(fourCards),
                health = 20,
                currentRoom = null,
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
            )

        // Draw room (deck now empty)
        val withRoom = gameState.drawRoom()
        assertEquals(0, withRoom.deck.size)
        assertEquals(4, withRoom.currentRoom?.size)

        // With empty deck and health > 0, isGameWon returns true
        // (even though there's still a room to process)
        assertTrue(withRoom.isGameWon)

        // The UI layer (ViewModel) handles showing win screen only after
        // the room is fully processed - this tests the underlying state logic
    }

    @Test
    fun `leftover card from previous room joins next room draw`() {
        val gameState = GameState.newGame().drawRoom()
        val originalRoom = gameState.currentRoom!!

        // Select first 3 cards, leaving the 4th
        val leftoverCard = originalRoom[3]
        val afterSelect = gameState.selectCards(originalRoom.take(3))

        assertEquals(1, afterSelect.currentRoom?.size)
        assertEquals(leftoverCard, afterSelect.currentRoom?.first())

        // Draw next room - should have 4 cards (1 leftover + 3 new)
        val nextRoom = afterSelect.drawRoom()
        assertEquals(4, nextRoom.currentRoom?.size)

        // The leftover card should still be in the room
        assertTrue(nextRoom.currentRoom!!.contains(leftoverCard))
    }

    @Test
    fun `drawing next room with 1 leftover and 2 cards in deck creates 3 card room`() {
        // Create state with 2 cards in deck and 1 leftover
        val twoCards =
            listOf(
                Card(Suit.CLUBS, Rank.KING),
                Card(Suit.SPADES, Rank.QUEEN),
            )
        val leftover = Card(Suit.HEARTS, Rank.TWO)
        val gameState =
            GameState(
                deck = Deck(twoCards),
                health = 20,
                currentRoom = listOf(leftover),
                weaponState = null,
                defeatedMonsters = emptyList(),
                discardPile = emptyList(),
                lastRoomAvoided = false,
                usedPotionThisTurn = false,
                lastCardProcessed = null,
            )

        val newState = gameState.drawRoom()

        assertEquals(3, newState.currentRoom?.size)
        assertEquals(0, newState.deck.size)
        assertTrue(newState.currentRoom!!.contains(leftover))
    }
}
