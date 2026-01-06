package dev.mattbachmann.scoundroid.data.model

/**
 * Represents a deck of Scoundrel cards.
 * The Scoundrel deck has 44 cards (standard 52-card deck with specific removals):
 * - Remove all Jokers
 * - Remove red face cards (J♥, Q♥, K♥, J♦, Q♦, K♦)
 * - Remove red Aces (A♥, A♦)
 *
 * Resulting composition:
 * - 26 Monsters: All Clubs (13) + All Spades (13)
 * - 9 Weapons: Diamonds 2-10
 * - 9 Potions: Hearts 2-10
 *
 * @property cards The list of cards in the deck
 */
data class Deck(val cards: List<Card>) {

    companion object {
        /**
         * Creates a new standard Scoundrel deck with 44 cards.
         */
        fun create(): Deck {
            val cards = mutableListOf<Card>()

            // Add all Clubs (2-A) - Monsters
            Rank.entries.forEach { rank ->
                cards.add(Card(Suit.CLUBS, rank))
            }

            // Add all Spades (2-A) - Monsters
            Rank.entries.forEach { rank ->
                cards.add(Card(Suit.SPADES, rank))
            }

            // Add Diamonds 2-10 only - Weapons
            val numberedRanks = listOf(
                Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX,
                Rank.SEVEN, Rank.EIGHT, Rank.NINE, Rank.TEN
            )
            numberedRanks.forEach { rank ->
                cards.add(Card(Suit.DIAMONDS, rank))
            }

            // Add Hearts 2-10 only - Potions
            numberedRanks.forEach { rank ->
                cards.add(Card(Suit.HEARTS, rank))
            }

            return Deck(cards)
        }
    }

    /**
     * Returns a new shuffled deck with the same cards in random order.
     */
    fun shuffle(): Deck {
        return Deck(cards.shuffled())
    }

    /**
     * Draws the specified number of cards from the top of the deck.
     *
     * @param count The number of cards to draw
     * @return Pair of (drawn cards, remaining deck)
     */
    fun draw(count: Int): Pair<List<Card>, Deck> {
        val actualCount = count.coerceAtMost(cards.size)
        val drawn = cards.take(actualCount)
        val remaining = cards.drop(actualCount)
        return Pair(drawn, Deck(remaining))
    }

    /**
     * Returns the number of cards remaining in the deck.
     */
    val size: Int
        get() = cards.size

    /**
     * Returns true if the deck is empty.
     */
    val isEmpty: Boolean
        get() = cards.isEmpty()
}
