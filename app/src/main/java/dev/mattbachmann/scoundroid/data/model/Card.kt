package dev.mattbachmann.scoundroid.data.model

/**
 * Represents a single playing card in the Scoundrel deck.
 *
 * @property suit The suit of the card (Clubs, Spades, Diamonds, Hearts)
 * @property rank The rank of the card (2-10, J, Q, K, A)
 */
data class Card(
    val suit: Suit,
    val rank: Rank,
) {
    /**
     * The type of card determined by its suit:
     * - Clubs & Spades: Monster
     * - Diamonds: Weapon
     * - Hearts: Potion
     */
    val type: CardType
        get() =
            when (suit) {
                Suit.CLUBS, Suit.SPADES -> CardType.MONSTER
                Suit.DIAMONDS -> CardType.WEAPON
                Suit.HEARTS -> CardType.POTION
            }

    /**
     * The numeric value of the card.
     * For all card types, the value equals the rank value (2-10=face, J=11, Q=12, K=13, A=14)
     */
    val value: Int
        get() = rank.value

    /**
     * Display name combining rank and suit symbol (e.g., "5♠", "K♣")
     */
    val displayName: String
        get() = "${rank.displayName}${suit.symbol}"

    override fun toString(): String = displayName
}
