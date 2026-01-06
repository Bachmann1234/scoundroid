package dev.mattbachmann.scoundroid.data.model

/**
 * Represents the four suits in the Scoundrel deck.
 * Each suit corresponds to a card type:
 * - Clubs & Spades: Monsters
 * - Diamonds: Weapons
 * - Hearts: Potions
 */
enum class Suit(val symbol: String) {
    CLUBS("♣"),
    SPADES("♠"),
    DIAMONDS("♦"),
    HEARTS("♥"),
}
