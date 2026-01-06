package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals

class RankTest {
    @Test
    fun `numbered ranks should have correct values`() {
        assertEquals(2, Rank.TWO.value)
        assertEquals(3, Rank.THREE.value)
        assertEquals(4, Rank.FOUR.value)
        assertEquals(5, Rank.FIVE.value)
        assertEquals(6, Rank.SIX.value)
        assertEquals(7, Rank.SEVEN.value)
        assertEquals(8, Rank.EIGHT.value)
        assertEquals(9, Rank.NINE.value)
        assertEquals(10, Rank.TEN.value)
    }

    @Test
    fun `face cards should have correct values`() {
        assertEquals(11, Rank.JACK.value)
        assertEquals(12, Rank.QUEEN.value)
        assertEquals(13, Rank.KING.value)
        assertEquals(14, Rank.ACE.value)
    }

    @Test
    fun `numbered ranks should have correct display names`() {
        assertEquals("2", Rank.TWO.displayName)
        assertEquals("5", Rank.FIVE.displayName)
        assertEquals("10", Rank.TEN.displayName)
    }

    @Test
    fun `face cards should have correct display names`() {
        assertEquals("J", Rank.JACK.displayName)
        assertEquals("Q", Rank.QUEEN.displayName)
        assertEquals("K", Rank.KING.displayName)
        assertEquals("A", Rank.ACE.displayName)
    }
}
