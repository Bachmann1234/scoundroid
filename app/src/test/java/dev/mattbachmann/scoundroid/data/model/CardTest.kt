package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals

class CardTest {

    @Test
    fun `card with Clubs suit should be Monster type`() {
        val card = Card(Suit.CLUBS, Rank.FIVE)
        assertEquals(CardType.MONSTER, card.type)
    }

    @Test
    fun `card with Spades suit should be Monster type`() {
        val card = Card(Suit.SPADES, Rank.QUEEN)
        assertEquals(CardType.MONSTER, card.type)
    }

    @Test
    fun `card with Diamonds suit should be Weapon type`() {
        val card = Card(Suit.DIAMONDS, Rank.SEVEN)
        assertEquals(CardType.WEAPON, card.type)
    }

    @Test
    fun `card with Hearts suit should be Potion type`() {
        val card = Card(Suit.HEARTS, Rank.THREE)
        assertEquals(CardType.POTION, card.type)
    }

    @Test
    fun `numbered card 2-10 should have face value`() {
        assertEquals(2, Card(Suit.CLUBS, Rank.TWO).value)
        assertEquals(5, Card(Suit.SPADES, Rank.FIVE).value)
        assertEquals(10, Card(Suit.DIAMONDS, Rank.TEN).value)
    }

    @Test
    fun `Jack should have value 11`() {
        val card = Card(Suit.CLUBS, Rank.JACK)
        assertEquals(11, card.value)
    }

    @Test
    fun `Queen should have value 12`() {
        val card = Card(Suit.SPADES, Rank.QUEEN)
        assertEquals(12, card.value)
    }

    @Test
    fun `King should have value 13`() {
        val card = Card(Suit.CLUBS, Rank.KING)
        assertEquals(13, card.value)
    }

    @Test
    fun `Ace should have value 14`() {
        val card = Card(Suit.SPADES, Rank.ACE)
        assertEquals(14, card.value)
    }

    @Test
    fun `card should have correct display name`() {
        assertEquals("2♣", Card(Suit.CLUBS, Rank.TWO).displayName)
        assertEquals("K♠", Card(Suit.SPADES, Rank.KING).displayName)
        assertEquals("5♦", Card(Suit.DIAMONDS, Rank.FIVE).displayName)
        assertEquals("A♥", Card(Suit.HEARTS, Rank.ACE).displayName)
    }
}
