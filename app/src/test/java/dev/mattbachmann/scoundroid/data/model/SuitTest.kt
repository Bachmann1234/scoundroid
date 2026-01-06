package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals

class SuitTest {

    @Test
    fun `Clubs should have correct symbol`() {
        assertEquals("♣", Suit.CLUBS.symbol)
    }

    @Test
    fun `Spades should have correct symbol`() {
        assertEquals("♠", Suit.SPADES.symbol)
    }

    @Test
    fun `Diamonds should have correct symbol`() {
        assertEquals("♦", Suit.DIAMONDS.symbol)
    }

    @Test
    fun `Hearts should have correct symbol`() {
        assertEquals("♥", Suit.HEARTS.symbol)
    }
}
