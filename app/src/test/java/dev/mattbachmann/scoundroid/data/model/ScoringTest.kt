package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Tests for scoring mechanics.
 *
 * Scoring rules:
 * - **Win**: Survive entire dungeon → score = remaining health
 * - **Win with full health**: health = 20 → score = 20 (special case)
 * - **Lose**: Health reaches 0 → score = 0 - sum of remaining monsters (negative)
 */
class ScoringTest {
    @Test
    fun `winning score is remaining health`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 15,
            )

        val score = game.calculateScore()

        assertEquals(15, score, "Win score should equal remaining health")
    }

    @Test
    fun `winning with full health scores 20`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 20,
            )

        val score = game.calculateScore()

        assertEquals(20, score, "Win with full health scores 20")
    }

    @Test
    fun `winning with 1 health scores 1`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 1,
            )

        val score = game.calculateScore()

        assertEquals(1, score, "Win with 1 health scores 1")
    }

    @Test
    fun `losing score is negative sum of remaining monsters`() {
        val remainingCards =
            listOf(
                // 7
                Card(Suit.SPADES, Rank.SEVEN),
                // 5
                Card(Suit.CLUBS, Rank.FIVE),
                // 10
                Card(Suit.SPADES, Rank.TEN),
            )
        val deck = Deck(remainingCards)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // Score = 0 - (7 + 5 + 10) = -22
        assertEquals(-22, score, "Lose score should be negative sum of remaining monsters")
    }

    @Test
    fun `losing with many monsters is very negative`() {
        val monsters =
            listOf(
                // 13
                Card(Suit.SPADES, Rank.KING),
                // 12
                Card(Suit.CLUBS, Rank.QUEEN),
                // 11
                Card(Suit.SPADES, Rank.JACK),
                // 10
                Card(Suit.CLUBS, Rank.TEN),
            )
        val deck = Deck(monsters)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // Score = 0 - (13 + 12 + 11 + 10) = -46
        assertEquals(-46, score, "Many remaining monsters = very negative score")
    }

    @Test
    fun `losing score ignores non-monster cards`() {
        val remainingCards =
            listOf(
                // 7 (monster)
                Card(Suit.SPADES, Rank.SEVEN),
                // 5 (weapon - ignore)
                Card(Suit.DIAMONDS, Rank.FIVE),
                // 3 (potion - ignore)
                Card(Suit.HEARTS, Rank.THREE),
                // 10 (monster)
                Card(Suit.CLUBS, Rank.TEN),
            )
        val deck = Deck(remainingCards)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // Score = 0 - (7 + 10) = -17 (weapons and potions don't count)
        assertEquals(-17, score, "Only monsters count toward losing score")
    }

    @Test
    fun `losing with no remaining cards scores 0`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 0,
            )

        val score = game.calculateScore()

        assertEquals(0, score, "Losing with no cards left scores 0")
    }

    @Test
    fun `can calculate score at any time`() {
        // Mid-game score
        val partialDeck =
            Deck(
                listOf(
                    Card(Suit.SPADES, Rank.FIVE),
                    Card(Suit.CLUBS, Rank.SEVEN),
                ),
            )
        val game =
            GameState.newGame().copy(
                deck = partialDeck,
                health = 10,
            )

        // Can calculate score even if game isn't over
        val score = game.calculateScore()

        // If we stopped here and won: score = 10
        // But deck isn't empty, so this is a hypothetical win score
        assertEquals(10, score, "Score calculation works mid-game")
    }

    @Test
    fun `losing score with one Ace monster`() {
        val aceMonster = listOf(Card(Suit.SPADES, Rank.ACE)) // 14
        val deck = Deck(aceMonster)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        assertEquals(-14, score, "Single Ace monster = -14")
    }

    @Test
    fun `winning with various health values`() {
        val emptyDeck = Deck(emptyList())

        for (health in 1..20) {
            val game =
                GameState.newGame().copy(
                    deck = emptyDeck,
                    health = health,
                )

            assertEquals(health, game.calculateScore(), "Health $health should score $health")
        }
    }

    @Test
    fun `losing score includes all remaining monsters`() {
        // Create a deck with only monsters
        val monsters =
            listOf(
                // 2
                Card(Suit.SPADES, Rank.TWO),
                // 3
                Card(Suit.CLUBS, Rank.THREE),
                // 4
                Card(Suit.SPADES, Rank.FOUR),
                // 5
                Card(Suit.CLUBS, Rank.FIVE),
            )
        val deck = Deck(monsters)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        // 2 + 3 + 4 + 5 = 14, so score = -14
        assertEquals(-14, score)
    }

    @Test
    fun `score is negative when losing`() {
        val monsters = listOf(Card(Suit.SPADES, Rank.TWO))
        val deck = Deck(monsters)
        val game =
            GameState.newGame().copy(
                deck = deck,
                health = 0,
            )

        val score = game.calculateScore()

        assertTrue(score <= 0, "Losing score should be ≤ 0")
    }

    @Test
    fun `score is positive when winning`() {
        val emptyDeck = Deck(emptyList())
        val game =
            GameState.newGame().copy(
                deck = emptyDeck,
                health = 5,
            )

        val score = game.calculateScore()

        assertTrue(score > 0, "Winning score should be > 0")
    }
}
