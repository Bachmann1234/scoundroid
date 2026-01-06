package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

class DeckTest {
    @Test
    fun `deck should have exactly 44 cards`() {
        val deck = Deck.create()
        assertEquals(44, deck.cards.size)
    }

    @Test
    fun `deck should have 26 monsters`() {
        val deck = Deck.create()
        val monsters = deck.cards.filter { it.type == CardType.MONSTER }
        assertEquals(26, monsters.size)
    }

    @Test
    fun `deck should have 9 weapons`() {
        val deck = Deck.create()
        val weapons = deck.cards.filter { it.type == CardType.WEAPON }
        assertEquals(9, weapons.size)
    }

    @Test
    fun `deck should have 9 potions`() {
        val deck = Deck.create()
        val potions = deck.cards.filter { it.type == CardType.POTION }
        assertEquals(9, potions.size)
    }

    @Test
    fun `deck should contain all clubs 2-A`() {
        val deck = Deck.create()
        val clubs = deck.cards.filter { it.suit == Suit.CLUBS }
        assertEquals(13, clubs.size)

        // Verify all ranks present
        Rank.entries.forEach { rank ->
            assertTrue(
                clubs.any { it.rank == rank },
                "Deck should contain $rank of Clubs",
            )
        }
    }

    @Test
    fun `deck should contain all spades 2-A`() {
        val deck = Deck.create()
        val spades = deck.cards.filter { it.suit == Suit.SPADES }
        assertEquals(13, spades.size)

        // Verify all ranks present
        Rank.entries.forEach { rank ->
            assertTrue(
                spades.any { it.rank == rank },
                "Deck should contain $rank of Spades",
            )
        }
    }

    @Test
    fun `deck should contain only diamonds 2-10`() {
        val deck = Deck.create()
        val diamonds = deck.cards.filter { it.suit == Suit.DIAMONDS }
        assertEquals(9, diamonds.size)

        // Should have 2-10
        val expectedRanks =
            listOf(
                Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
                Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
            )
        expectedRanks.forEach { rank ->
            assertTrue(
                diamonds.any { it.rank == rank },
                "Deck should contain $rank of Diamonds",
            )
        }

        // Should NOT have J, Q, K, A
        assertFalse(diamonds.any { it.rank == Rank.JACK })
        assertFalse(diamonds.any { it.rank == Rank.QUEEN })
        assertFalse(diamonds.any { it.rank == Rank.KING })
        assertFalse(diamonds.any { it.rank == Rank.ACE })
    }

    @Test
    fun `deck should contain only hearts 2-10`() {
        val deck = Deck.create()
        val hearts = deck.cards.filter { it.suit == Suit.HEARTS }
        assertEquals(9, hearts.size)

        // Should have 2-10
        val expectedRanks =
            listOf(
                Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
                Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN,
            )
        expectedRanks.forEach { rank ->
            assertTrue(
                hearts.any { it.rank == rank },
                "Deck should contain $rank of Hearts",
            )
        }

        // Should NOT have J, Q, K, A
        assertFalse(hearts.any { it.rank == Rank.JACK })
        assertFalse(hearts.any { it.rank == Rank.QUEEN })
        assertFalse(hearts.any { it.rank == Rank.KING })
        assertFalse(hearts.any { it.rank == Rank.ACE })
    }

    @Test
    fun `deck should not have duplicate cards`() {
        val deck = Deck.create()
        val uniqueCards = deck.cards.distinct()
        assertEquals(deck.cards.size, uniqueCards.size, "Deck should not contain duplicates")
    }

    @Test
    fun `shuffled deck should have same number of cards`() {
        val deck = Deck.create()
        val shuffled = deck.shuffle()
        assertEquals(44, shuffled.cards.size)
    }

    @Test
    fun `shuffled deck should contain same cards`() {
        val deck = Deck.create()
        val shuffled = deck.shuffle()

        // Both decks should have same cards (regardless of order)
        assertEquals(
            deck.cards.sorted().map { it.displayName },
            shuffled.cards.sorted().map { it.displayName },
        )
    }

    @Test
    fun `shuffle should change card order`() {
        val deck = Deck.create()
        val shuffled = deck.shuffle()

        // Very unlikely (1 in 44! chance) that order is identical after shuffle
        // Check if at least one card is in a different position
        val orderChanged =
            deck.cards.indices.any { i ->
                deck.cards[i] != shuffled.cards[i]
            }
        assertTrue(orderChanged, "Shuffle should change card order")
    }

    @Test
    fun `drawing from empty deck should return empty list`() {
        val deck = Deck(emptyList())
        val (drawn, _) = deck.draw(4)
        assertEquals(0, drawn.size)
    }

    @Test
    fun `drawing 4 cards should return 4 cards and remaining deck`() {
        val deck = Deck.create()
        val (drawn, remaining) = deck.draw(4)

        assertEquals(4, drawn.size)
        assertEquals(40, remaining.cards.size)
    }

    @Test
    fun `drawing more cards than available should return all cards`() {
        val deck = Deck.create()
        val (drawn, remaining) = deck.draw(50)

        assertEquals(44, drawn.size)
        assertEquals(0, remaining.cards.size)
    }

    @Test
    fun `drawn cards should be from top of deck`() {
        val originalCards =
            listOf(
                Card(Suit.CLUBS, Rank.TWO),
                Card(Suit.SPADES, Rank.FIVE),
                Card(Suit.HEARTS, Rank.TEN),
                Card(Suit.DIAMONDS, Rank.SEVEN),
            )
        val deck = Deck(originalCards)
        val (drawn, _) = deck.draw(2)

        assertEquals(originalCards[0], drawn[0])
        assertEquals(originalCards[1], drawn[1])
    }
}
