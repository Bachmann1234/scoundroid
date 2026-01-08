package dev.mattbachmann.scoundroid.data.model

/**
 * Represents a single entry in the game action log.
 * Each sealed class variant captures specific data for different action types.
 */
sealed class LogEntry {
    abstract val timestamp: Long

    /**
     * Monster was fought.
     * @param monster The monster card fought
     * @param weaponUsed The weapon used (null if barehanded)
     * @param damageBlocked Damage reduced by weapon (0 if barehanded)
     * @param damageTaken Actual damage taken
     * @param healthBefore Player health before combat
     * @param healthAfter Player health after combat
     */
    data class MonsterFought(
        override val timestamp: Long,
        val monster: Card,
        val weaponUsed: Card?,
        val damageBlocked: Int,
        val damageTaken: Int,
        val healthBefore: Int,
        val healthAfter: Int,
    ) : LogEntry()

    /**
     * Weapon was equipped.
     * @param weapon The weapon card equipped
     * @param replacedWeapon The weapon that was replaced (null if none)
     */
    data class WeaponEquipped(
        override val timestamp: Long,
        val weapon: Card,
        val replacedWeapon: Card?,
    ) : LogEntry()

    /**
     * Potion was used.
     * @param potion The potion card
     * @param healthRestored Actual health restored (may be less than value due to cap)
     * @param healthBefore Player health before
     * @param healthAfter Player health after
     * @param wasDiscarded True if this was a second potion (discarded without effect)
     */
    data class PotionUsed(
        override val timestamp: Long,
        val potion: Card,
        val healthRestored: Int,
        val healthBefore: Int,
        val healthAfter: Int,
        val wasDiscarded: Boolean,
    ) : LogEntry()

    /**
     * Room was drawn from deck.
     * @param cardsDrawn Number of cards drawn (usually 3 or 4)
     * @param deckSizeAfter Deck size after drawing
     */
    data class RoomDrawn(
        override val timestamp: Long,
        val cardsDrawn: Int,
        val deckSizeAfter: Int,
    ) : LogEntry()

    /**
     * Room was avoided.
     * @param cardsReturned Number of cards returned to bottom of deck
     */
    data class RoomAvoided(
        override val timestamp: Long,
        val cardsReturned: Int,
    ) : LogEntry()

    /**
     * New game started.
     */
    data class GameStarted(
        override val timestamp: Long,
    ) : LogEntry()
}
