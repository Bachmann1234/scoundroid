package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * Tests for weapon degradation mechanics.
 *
 * Critical rule: Once a weapon is used on a monster, it can ONLY be used on monsters
 * with value ≤ the last monster it defeated. The weapon degrades to the value of
 * the last monster defeated.
 */
class WeaponStateTest {
    @Test
    fun `new weapon has no max monster value restriction`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon)

        assertNull(weaponState.maxMonsterValue, "New weapon should have no restrictions")
    }

    @Test
    fun `new weapon can defeat any monster`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦ (value 5)
        val weaponState = WeaponState(weapon)

        val queenMonster = Card(Suit.SPADES, Rank.QUEEN) // Q♠ (value 12)

        assertTrue(
            weaponState.canDefeat(queenMonster),
            "New weapon should be able to defeat any monster",
        )
    }

    @Test
    fun `weapon tracks max monster value after first use`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon)

        val queenMonster = Card(Suit.SPADES, Rank.QUEEN) // Q♠ (value 12)
        val newState = weaponState.useOn(queenMonster)

        assertEquals(
            12,
            newState.maxMonsterValue,
            "Weapon should track the value of the first monster defeated",
        )
    }

    @Test
    fun `weapon can defeat monster equal to max value`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon)

        // First use on Queen (12)
        val usedWeapon = weaponState.useOn(Card(Suit.SPADES, Rank.QUEEN))

        // Should be able to defeat another 12
        val anotherQueen = Card(Suit.CLUBS, Rank.QUEEN) // Q♣ (value 12)
        assertTrue(
            usedWeapon.canDefeat(anotherQueen),
            "Weapon should be able to defeat monster with exact max value",
        )
    }

    @Test
    fun `weapon can defeat monster less than max value`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon)

        // First use on Queen (12)
        val usedWeapon = weaponState.useOn(Card(Suit.SPADES, Rank.QUEEN))

        // Should be able to defeat a 6
        val sixMonster = Card(Suit.CLUBS, Rank.SIX) // 6♣ (value 6)
        assertTrue(
            usedWeapon.canDefeat(sixMonster),
            "Weapon should be able to defeat monster less than max value",
        )
    }

    @Test
    fun `weapon cannot defeat monster greater than max value`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon)

        // First use on SIX (6)
        val usedWeapon = weaponState.useOn(Card(Suit.SPADES, Rank.SIX))

        // Should NOT be able to defeat a 7
        val sevenMonster = Card(Suit.CLUBS, Rank.SEVEN) // 7♣ (value 7)
        assertFalse(
            usedWeapon.canDefeat(sevenMonster),
            "Weapon should NOT be able to defeat monster greater than max value",
        )
    }

    @Test
    fun `weapon degrades when used on lower value monster`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon)

        // Use on Queen (12)
        val afterQueen = weaponState.useOn(Card(Suit.SPADES, Rank.QUEEN))
        assertEquals(12, afterQueen.maxMonsterValue, "Should track Queen value")

        // Use on Six (6) - degrades!
        val afterSix = afterQueen.useOn(Card(Suit.CLUBS, Rank.SIX))
        assertEquals(6, afterSix.maxMonsterValue, "Should degrade to Six value")
    }

    @Test
    fun `weapon degradation example from rules`() {
        // Example from CLAUDE.md:
        // Weapon: 5♦
        // Defeat Queen (12) → maxMonsterValue = 12
        // Can use on any monster ≤ 12
        // Defeat 6 → maxMonsterValue = 6
        // Can NOW only use on monsters ≤ 6
        // Cannot use on 7+

        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        var weaponState = WeaponState(weapon)

        // Defeat Queen (12)
        weaponState = weaponState.useOn(Card(Suit.SPADES, Rank.QUEEN))
        assertEquals(12, weaponState.maxMonsterValue)

        // Can use on monsters ≤ 12
        assertTrue(weaponState.canDefeat(Card(Suit.CLUBS, Rank.TEN))) // 10
        assertTrue(weaponState.canDefeat(Card(Suit.SPADES, Rank.QUEEN))) // 12

        // Defeat 6
        weaponState = weaponState.useOn(Card(Suit.CLUBS, Rank.SIX))
        assertEquals(6, weaponState.maxMonsterValue)

        // Can NOW only use on monsters ≤ 6
        assertTrue(weaponState.canDefeat(Card(Suit.SPADES, Rank.SIX))) // 6
        assertTrue(weaponState.canDefeat(Card(Suit.CLUBS, Rank.TWO))) // 2
        assertFalse(weaponState.canDefeat(Card(Suit.SPADES, Rank.SEVEN))) // 7
        assertFalse(weaponState.canDefeat(Card(Suit.CLUBS, Rank.TEN))) // 10
    }

    @Test
    fun `weapon degradation is sequential and cumulative`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        var weaponState = WeaponState(weapon)

        // Progressive degradation
        weaponState = weaponState.useOn(Card(Suit.SPADES, Rank.KING)) // 13
        assertEquals(13, weaponState.maxMonsterValue)

        weaponState = weaponState.useOn(Card(Suit.CLUBS, Rank.TEN)) // 10
        assertEquals(10, weaponState.maxMonsterValue)

        weaponState = weaponState.useOn(Card(Suit.SPADES, Rank.SEVEN)) // 7
        assertEquals(7, weaponState.maxMonsterValue)

        weaponState = weaponState.useOn(Card(Suit.CLUBS, Rank.THREE)) // 3
        assertEquals(3, weaponState.maxMonsterValue)

        // Now can only defeat 2 or 3
        assertTrue(weaponState.canDefeat(Card(Suit.SPADES, Rank.TWO)))
        assertTrue(weaponState.canDefeat(Card(Suit.CLUBS, Rank.THREE)))
        assertFalse(weaponState.canDefeat(Card(Suit.SPADES, Rank.FOUR)))
    }

    @Test
    fun `using weapon on equal value monster maintains max value`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        var weaponState = WeaponState(weapon)

        // Use on 10
        weaponState = weaponState.useOn(Card(Suit.SPADES, Rank.TEN))
        assertEquals(10, weaponState.maxMonsterValue)

        // Use on another 10 - should stay at 10
        weaponState = weaponState.useOn(Card(Suit.CLUBS, Rank.TEN))
        assertEquals(10, weaponState.maxMonsterValue)
    }

    @Test
    fun `weapon card itself remains constant`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        var weaponState = WeaponState(weapon)

        weaponState = weaponState.useOn(Card(Suit.SPADES, Rank.QUEEN))
        weaponState = weaponState.useOn(Card(Suit.CLUBS, Rank.THREE))

        assertEquals(
            weapon,
            weaponState.weapon,
            "Weapon card should not change, only its state",
        )
    }

    @Test
    fun `cannot use weapon on non-monster card`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon)

        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥ (potion)
        val anotherWeapon = Card(Suit.DIAMONDS, Rank.TEN) // 10♦ (weapon)

        assertFalse(
            weaponState.canDefeat(potion),
            "Cannot use weapon on potion",
        )
        assertFalse(
            weaponState.canDefeat(anotherWeapon),
            "Cannot use weapon on another weapon",
        )
    }

    @Test
    fun `weapon with Ace value starts unrestricted`() {
        val aceWeapon = Card(Suit.DIAMONDS, Rank.TEN) // 10♦ (highest weapon)
        val weaponState = WeaponState(aceWeapon)

        // New weapon can defeat even an Ace monster
        val aceMonster = Card(Suit.SPADES, Rank.ACE) // A♠ (value 14)
        assertTrue(
            weaponState.canDefeat(aceMonster),
            "New weapon should defeat Ace monster",
        )
    }

    @Test
    fun `weapon value does not affect degradation tracking`() {
        // Weapon value (for damage reduction) is separate from degradation tracking
        val lowWeapon = Card(Suit.DIAMONDS, Rank.TWO) // 2♦ (low damage reduction)
        var weaponState = WeaponState(lowWeapon)

        // Even a 2♦ can defeat a King when new
        val kingMonster = Card(Suit.SPADES, Rank.KING) // K♠ (value 13)
        assertTrue(
            weaponState.canDefeat(kingMonster),
            "Weapon damage value doesn't affect what it can defeat when new",
        )

        // And degrades the same way
        weaponState = weaponState.useOn(kingMonster)
        assertEquals(13, weaponState.maxMonsterValue)

        weaponState = weaponState.useOn(Card(Suit.CLUBS, Rank.FIVE))
        assertEquals(5, weaponState.maxMonsterValue)
    }
}
