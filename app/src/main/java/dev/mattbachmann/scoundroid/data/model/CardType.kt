package dev.mattbachmann.scoundroid.data.model

/**
 * The three types of cards in Scoundrel.
 * - MONSTER (♣ ♠): Deals damage equal to card value
 * - WEAPON (♦): Reduces monster damage
 * - POTION (♥): Restores health
 */
enum class CardType {
    MONSTER,
    WEAPON,
    POTION,
}
