package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

class LogEntryTest {
    private val monster = Card(Suit.SPADES, Rank.KING)
    private val weapon = Card(Suit.DIAMONDS, Rank.FIVE)
    private val potion = Card(Suit.HEARTS, Rank.SEVEN)

    @Test
    fun `MonsterFought captures all combat details for barehanded fight`() {
        val entry =
            LogEntry.MonsterFought(
                timestamp = 1000L,
                monster = monster,
                weaponUsed = null,
                damageBlocked = 0,
                damageTaken = 13,
                healthBefore = 20,
                healthAfter = 7,
            )

        assertEquals(1000L, entry.timestamp)
        assertEquals(monster, entry.monster)
        assertNull(entry.weaponUsed)
        assertEquals(0, entry.damageBlocked)
        assertEquals(13, entry.damageTaken)
        assertEquals(20, entry.healthBefore)
        assertEquals(7, entry.healthAfter)
    }

    @Test
    fun `MonsterFought captures weapon details when weapon used`() {
        val entry =
            LogEntry.MonsterFought(
                timestamp = 1000L,
                monster = monster,
                weaponUsed = weapon,
                damageBlocked = 5,
                damageTaken = 8,
                healthBefore = 20,
                healthAfter = 12,
            )

        assertEquals(weapon, entry.weaponUsed)
        assertEquals(5, entry.damageBlocked)
        assertEquals(8, entry.damageTaken)
    }

    @Test
    fun `WeaponEquipped captures weapon without replacement`() {
        val entry =
            LogEntry.WeaponEquipped(
                timestamp = 2000L,
                weapon = weapon,
                replacedWeapon = null,
            )

        assertEquals(2000L, entry.timestamp)
        assertEquals(weapon, entry.weapon)
        assertNull(entry.replacedWeapon)
    }

    @Test
    fun `WeaponEquipped captures weapon with replacement`() {
        val oldWeapon = Card(Suit.DIAMONDS, Rank.THREE)
        val entry =
            LogEntry.WeaponEquipped(
                timestamp = 2000L,
                weapon = weapon,
                replacedWeapon = oldWeapon,
            )

        assertEquals(weapon, entry.weapon)
        assertEquals(oldWeapon, entry.replacedWeapon)
    }

    @Test
    fun `PotionUsed captures health restoration`() {
        val entry =
            LogEntry.PotionUsed(
                timestamp = 3000L,
                potion = potion,
                healthRestored = 7,
                healthBefore = 10,
                healthAfter = 17,
                wasDiscarded = false,
            )

        assertEquals(3000L, entry.timestamp)
        assertEquals(potion, entry.potion)
        assertEquals(7, entry.healthRestored)
        assertEquals(10, entry.healthBefore)
        assertEquals(17, entry.healthAfter)
        assertFalse(entry.wasDiscarded)
    }

    @Test
    fun `PotionUsed captures discarded state for second potion`() {
        val entry =
            LogEntry.PotionUsed(
                timestamp = 3000L,
                potion = potion,
                healthRestored = 0,
                healthBefore = 15,
                healthAfter = 15,
                wasDiscarded = true,
            )

        assertTrue(entry.wasDiscarded)
        assertEquals(0, entry.healthRestored)
        assertEquals(entry.healthBefore, entry.healthAfter)
    }

    @Test
    fun `RoomDrawn captures cards drawn and deck size`() {
        val entry =
            LogEntry.RoomDrawn(
                timestamp = 4000L,
                cardsDrawn = 4,
                deckSizeAfter = 36,
            )

        assertEquals(4000L, entry.timestamp)
        assertEquals(4, entry.cardsDrawn)
        assertEquals(36, entry.deckSizeAfter)
    }

    @Test
    fun `RoomDrawn captures 3 cards when 1 leftover`() {
        val entry =
            LogEntry.RoomDrawn(
                timestamp = 4000L,
                cardsDrawn = 3,
                deckSizeAfter = 0,
            )

        assertEquals(3, entry.cardsDrawn)
    }

    @Test
    fun `RoomAvoided captures cards returned count`() {
        val entry =
            LogEntry.RoomAvoided(
                timestamp = 5000L,
                cardsReturned = 4,
            )

        assertEquals(5000L, entry.timestamp)
        assertEquals(4, entry.cardsReturned)
    }

    @Test
    fun `GameStarted marks start of new game`() {
        val entry =
            LogEntry.GameStarted(
                timestamp = 0L,
            )

        assertEquals(0L, entry.timestamp)
    }

    @Test
    fun `log entries can be sorted by timestamp`() {
        val entries =
            listOf(
                LogEntry.GameStarted(timestamp = 0L),
                LogEntry.RoomDrawn(timestamp = 100L, cardsDrawn = 4, deckSizeAfter = 36),
                LogEntry.MonsterFought(
                    timestamp = 200L,
                    monster = monster,
                    weaponUsed = null,
                    damageBlocked = 0,
                    damageTaken = 13,
                    healthBefore = 20,
                    healthAfter = 7,
                ),
                LogEntry.WeaponEquipped(timestamp = 300L, weapon = weapon, replacedWeapon = null),
            )

        val sorted = entries.sortedBy { it.timestamp }

        assertEquals(0L, sorted[0].timestamp)
        assertEquals(100L, sorted[1].timestamp)
        assertEquals(200L, sorted[2].timestamp)
        assertEquals(300L, sorted[3].timestamp)
    }

    @Test
    fun `log entries can be sorted in reverse order for newest first display`() {
        val entries =
            listOf(
                LogEntry.GameStarted(timestamp = 0L),
                LogEntry.RoomDrawn(timestamp = 100L, cardsDrawn = 4, deckSizeAfter = 36),
                LogEntry.WeaponEquipped(timestamp = 300L, weapon = weapon, replacedWeapon = null),
            )

        val reversed = entries.sortedByDescending { it.timestamp }

        assertEquals(300L, reversed[0].timestamp)
        assertEquals(100L, reversed[1].timestamp)
        assertEquals(0L, reversed[2].timestamp)
    }
}
