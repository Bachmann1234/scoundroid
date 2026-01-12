package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.Deck

/**
 * Tracks knowledge about the Scoundrel deck throughout a game.
 *
 * In Scoundrel, the exact 44-card deck composition is known, and we can track
 * what cards have been played. This "card counting" enables smarter decisions.
 *
 * @property processedCards Cards that have been permanently removed from play
 *                          (fought monsters, used potions, equipped/discarded weapons)
 * @property skippedCards Cards sent to bottom of deck via room skip, in order.
 *                        These will return when the deck cycles.
 */
data class DeckKnowledge(
    val processedCards: Set<Card>,
    val skippedCards: List<Card>,
    private val fullDeck: List<Card>,
) {
    companion object {
        /**
         * Creates initial knowledge with the full Scoundrel deck.
         */
        fun initial(): DeckKnowledge =
            DeckKnowledge(
                processedCards = emptySet(),
                skippedCards = emptyList(),
                fullDeck = Deck.create().cards,
            )
    }

    /**
     * All cards still in the game (not yet processed).
     */
    val remainingCards: List<Card>
        get() = fullDeck.filter { it !in processedCards }

    /**
     * Remaining monster cards.
     */
    val remainingMonsters: List<Card>
        get() = remainingCards.filter { it.type == CardType.MONSTER }

    /**
     * Remaining weapon cards.
     */
    val remainingWeapons: List<Card>
        get() = remainingCards.filter { it.type == CardType.WEAPON }

    /**
     * Remaining potion cards.
     */
    val remainingPotions: List<Card>
        get() = remainingCards.filter { it.type == CardType.POTION }

    /**
     * The highest monster value still in the deck, or 0 if none remain.
     */
    val maxMonsterRemaining: Int
        get() = remainingMonsters.maxOfOrNull { it.value } ?: 0

    /**
     * The highest weapon value still in the deck, or 0 if none remain.
     */
    val maxWeaponRemaining: Int
        get() = remainingWeapons.maxOfOrNull { it.value } ?: 0

    /**
     * Total damage from all remaining monsters.
     */
    val totalDamageRemaining: Int
        get() = remainingMonsters.sumOf { it.value }

    /**
     * Total healing from all remaining potions.
     */
    val totalHealingRemaining: Int
        get() = remainingPotions.sumOf { it.value }

    /**
     * Number of cards that will be drawn before skipped cards return.
     * This is the remaining deck size minus the skipped cards.
     */
    val cardsBeforeSkipped: Int
        get() = remainingCards.size - skippedCards.size

    /**
     * Records that a card has been processed (removed from play).
     */
    fun cardProcessed(card: Card): DeckKnowledge =
        copy(
            processedCards = processedCards + card,
            skippedCards = skippedCards.filter { it != card },
        )

    /**
     * Records that a room was skipped, sending cards to bottom of deck.
     * Cards are added to skippedCards in order (first skipped = first to return).
     */
    fun roomSkipped(room: List<Card>): DeckKnowledge =
        copy(
            skippedCards = skippedCards + room,
        )

    /**
     * Expected damage per room based on remaining cards.
     * Assumes 4 cards drawn, calculates expected monster damage.
     */
    fun expectedDamagePerRoom(): Double {
        if (remainingCards.isEmpty()) return 0.0

        val monsterCount = remainingMonsters.size.toDouble()
        val totalCards = remainingCards.size.toDouble()
        val avgMonsterDamage =
            if (remainingMonsters.isEmpty()) 0.0 else totalDamageRemaining.toDouble() / monsterCount

        // Expected monsters in a room of 4
        val roomSize = minOf(4.0, totalCards)
        val expectedMonsters = roomSize * (monsterCount / totalCards)

        return expectedMonsters * avgMonsterDamage
    }

    /**
     * Probability of at least one weapon appearing in the next room.
     * Calculated as 1 - P(no weapons in 4 draws without replacement).
     */
    fun chanceOfWeaponInNextRoom(): Double {
        val weaponCount = remainingWeapons.size
        val totalCards = remainingCards.size

        if (totalCards == 0) return 0.0
        if (weaponCount == 0) return 0.0
        if (weaponCount >= totalCards) return 1.0

        // P(no weapon in 4 cards) = product of (remaining non-weapons / remaining total)
        val roomSize = minOf(4, totalCards)
        var probNoWeapon = 1.0
        var nonWeapons = totalCards - weaponCount
        var remaining = totalCards

        for (i in 0 until roomSize) {
            probNoWeapon *= nonWeapons.toDouble() / remaining.toDouble()
            nonWeapons--
            remaining--
        }

        return 1.0 - probNoWeapon
    }

    /**
     * Probability of at least one potion appearing in the next room.
     */
    fun chanceOfPotionInNextRoom(): Double {
        val potionCount = remainingPotions.size
        val totalCards = remainingCards.size

        if (totalCards == 0) return 0.0
        if (potionCount == 0) return 0.0
        if (potionCount >= totalCards) return 1.0

        val roomSize = minOf(4, totalCards)
        var probNoPotion = 1.0
        var nonPotions = totalCards - potionCount
        var remaining = totalCards

        for (i in 0 until roomSize) {
            probNoPotion *= nonPotions.toDouble() / remaining.toDouble()
            nonPotions--
            remaining--
        }

        return 1.0 - probNoPotion
    }

    /**
     * Calculates survival margin: health + potential healing - remaining damage.
     * Positive = can theoretically survive, Negative = will die without perfect play.
     */
    fun survivalMargin(currentHealth: Int): Int = currentHealth + totalHealingRemaining - totalDamageRemaining
}
