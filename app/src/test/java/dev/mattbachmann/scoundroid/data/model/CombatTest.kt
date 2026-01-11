package dev.mattbachmann.scoundroid.data.model

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertNull

/**
 * Tests for combat mechanics.
 *
 * Combat rules:
 * - Barehanded: Take full monster damage
 * - With weapon: damage = max(0, monster - weapon)
 * - Weapon can only be used if canDefeat(monster) is true
 * - Defeated monsters go on weapon stack
 * - Weapon degrades after each use
 */
class CombatTest {
    @Test
    fun `barehanded combat deals full monster damage`() {
        val game = GameState.newGame().copy(health = 20)
        val monster = Card(Suit.SPADES, Rank.SEVEN) // 7♠ (damage 7)

        val afterCombat = game.fightMonster(monster)

        assertEquals(13, afterCombat.health, "Should take full 7 damage barehanded")
    }

    @Test
    fun `barehanded combat with high value monster`() {
        val game = GameState.newGame().copy(health = 20)
        val kingMonster = Card(Suit.CLUBS, Rank.KING) // K♣ (damage 13)

        val afterCombat = game.fightMonster(kingMonster)

        assertEquals(7, afterCombat.health, "Should take full 13 damage")
    }

    @Test
    fun `barehanded combat can kill player`() {
        val game = GameState.newGame().copy(health = 5)
        val monster = Card(Suit.SPADES, Rank.TEN) // 10♠ (damage 10)

        val afterCombat = game.fightMonster(monster)

        assertEquals(0, afterCombat.health, "Health should not go below 0")
    }

    @Test
    fun `weapon combat reduces damage`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
            )

        val monster = Card(Suit.SPADES, Rank.SEVEN) // 7♠ (damage 7)
        val afterCombat = game.fightMonster(monster)

        // Damage = max(0, 7 - 5) = 2
        assertEquals(18, afterCombat.health, "Should take reduced damage: 7 - 5 = 2")
    }

    @Test
    fun `weapon can completely negate damage`() {
        val weapon = Card(Suit.DIAMONDS, Rank.TEN) // 10♦
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
            )

        val monster = Card(Suit.SPADES, Rank.SEVEN) // 7♠ (damage 7)
        val afterCombat = game.fightMonster(monster)

        // Damage = max(0, 7 - 10) = 0
        assertEquals(20, afterCombat.health, "Weapon should negate all damage")
    }

    @Test
    fun `weapon exactly neutralizes monster deals zero damage`() {
        val weapon = Card(Suit.DIAMONDS, Rank.SEVEN) // 7♦
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
            )

        val monster = Card(Suit.CLUBS, Rank.SEVEN) // 7♣ (damage 7)
        val afterCombat = game.fightMonster(monster)

        // Damage = max(0, 7 - 7) = 0
        assertEquals(20, afterCombat.health, "Equal weapon and monster = 0 damage")
    }

    @Test
    fun `weapon degrades after combat`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
            )

        val monster = Card(Suit.SPADES, Rank.TEN) // 10♠
        val afterCombat = game.fightMonster(monster)

        val weaponStateAfter = requireNotNull(afterCombat.weaponState) { "Weapon should still be equipped" }
        assertEquals(10, weaponStateAfter.maxMonsterValue, "Weapon should degrade to monster value")
    }

    @Test
    fun `degraded weapon cannot defeat stronger monster - barehanded damage`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon, maxMonsterValue = 6)
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = weaponState,
            )

        val monster = Card(Suit.SPADES, Rank.TEN) // 10♠ (stronger than weapon can handle)
        val afterCombat = game.fightMonster(monster)

        // Weapon can't defeat this monster, so barehanded combat
        assertEquals(10, afterCombat.health, "Should take full 10 damage (weapon can't help)")
        // Weapon state should NOT change
        assertEquals(6, afterCombat.weaponState?.maxMonsterValue, "Weapon should not degrade if not used")
    }

    @Test
    fun `degraded weapon can defeat weaker monster - reduced damage`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon, maxMonsterValue = 8)
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = weaponState,
            )

        val monster = Card(Suit.SPADES, Rank.SIX) // 6♠ (within weapon range)
        val afterCombat = game.fightMonster(monster)

        // Weapon can defeat this monster: damage = max(0, 6 - 5) = 1
        assertEquals(19, afterCombat.health, "Should take reduced damage: 6 - 5 = 1")
        assertEquals(6, afterCombat.weaponState?.maxMonsterValue, "Weapon should degrade to 6")
    }

    @Test
    fun `monster is added to defeated monsters pile`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
                defeatedMonsters = emptyList(),
            )

        val monster = Card(Suit.SPADES, Rank.SEVEN) // 7♠
        val afterCombat = game.fightMonster(monster)

        assertEquals(1, afterCombat.defeatedMonsters.size, "Should have 1 defeated monster")
        assertEquals(monster, afterCombat.defeatedMonsters.first(), "Should be the monster we fought")
    }

    @Test
    fun `multiple monsters accumulate in defeated pile`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        var game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
                defeatedMonsters = emptyList(),
            )

        val monster1 = Card(Suit.SPADES, Rank.SEVEN) // 7♠
        game = game.fightMonster(monster1)

        val monster2 = Card(Suit.CLUBS, Rank.FIVE) // 5♣
        game = game.fightMonster(monster2)

        val monster3 = Card(Suit.SPADES, Rank.THREE) // 3♠
        game = game.fightMonster(monster3)

        assertEquals(3, game.defeatedMonsters.size, "Should have 3 defeated monsters")
        assertEquals(listOf(monster1, monster2, monster3), game.defeatedMonsters)
    }

    @Test
    fun `barehanded monster goes to defeated pile`() {
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = null,
                defeatedMonsters = emptyList(),
            )

        val monster = Card(Suit.SPADES, Rank.SEVEN) // 7♠
        val afterCombat = game.fightMonster(monster)

        assertEquals(1, afterCombat.defeatedMonsters.size, "Barehanded monster should go to pile")
        assertEquals(monster, afterCombat.defeatedMonsters.first())
    }

    @Test
    fun `combat with no weapon equipped is barehanded`() {
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = null,
            )

        val monster = Card(Suit.SPADES, Rank.NINE) // 9♠
        val afterCombat = game.fightMonster(monster)

        assertEquals(11, afterCombat.health, "Should take full 9 damage barehanded")
        assertNull(afterCombat.weaponState, "Should still have no weapon")
    }

    @Test
    fun `weapon combat with Ace monster`() {
        val weapon = Card(Suit.DIAMONDS, Rank.TEN) // 10♦
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
            )

        val aceMonster = Card(Suit.SPADES, Rank.ACE) // A♠ (damage 14)
        val afterCombat = game.fightMonster(aceMonster)

        // Damage = max(0, 14 - 10) = 4
        assertEquals(16, afterCombat.health, "Should take 14 - 10 = 4 damage")
        assertEquals(14, afterCombat.weaponState?.maxMonsterValue, "Weapon degrades to 14")
    }

    @Test
    fun `sequential combat with weapon degradation`() {
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        var game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
            )

        // Fight Queen (12) - weapon works
        game = game.fightMonster(Card(Suit.SPADES, Rank.QUEEN))
        assertEquals(13, game.health, "First fight: 20 - (12-5) = 13")
        assertEquals(12, game.weaponState?.maxMonsterValue)

        // Fight Seven (7) - weapon works and degrades
        game = game.fightMonster(Card(Suit.CLUBS, Rank.SEVEN))
        assertEquals(11, game.health, "Second fight: 13 - (7-5) = 11")
        assertEquals(7, game.weaponState?.maxMonsterValue)

        // Fight Ten (10) - weapon can't help (max is 7)
        game = game.fightMonster(Card(Suit.SPADES, Rank.TEN))
        assertEquals(1, game.health, "Third fight: 11 - 10 = 1 (barehanded)")
        assertEquals(7, game.weaponState?.maxMonsterValue, "Weapon unchanged")
    }

    @Test
    fun `cannot fight non-monster card`() {
        val game = GameState.newGame().copy(health = 20)
        val potion = Card(Suit.HEARTS, Rank.FIVE) // 5♥ (potion)

        assertFailsWith<IllegalArgumentException> {
            game.fightMonster(potion)
        }
    }

    // ========== Combat Choice Behavior Tests ==========

    @Test
    fun `fightMonster convenience method auto-uses weapon when it can defeat monster`() {
        // Note: fightMonster() is a convenience method that auto-uses weapon if canDefeat() is true
        // In actual gameplay, the ViewModel uses fightMonsterWithWeapon() or fightMonsterBarehanded()
        // based on player choice
        val weapon = Card(Suit.DIAMONDS, Rank.SEVEN) // 7♦
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = WeaponState(weapon),
            )

        val monster = Card(Suit.SPADES, Rank.FIVE) // 5♠
        val afterCombat = game.fightMonster(monster)

        // Weapon was automatically used: damage = max(0, 5 - 7) = 0
        assertEquals(20, afterCombat.health, "Weapon auto-used, no damage taken")
        // Weapon degraded
        assertEquals(5, afterCombat.weaponState?.maxMonsterValue, "Weapon degraded to 5")
    }

    @Test
    fun `weapon remains equipped but unused when it cannot defeat monster`() {
        // When weapon can't defeat monster, it stays equipped but player fights barehanded
        val weapon = Card(Suit.DIAMONDS, Rank.FIVE) // 5♦
        val weaponState = WeaponState(weapon, maxMonsterValue = 4) // Can only defeat 4 or less
        val game =
            GameState.newGame().copy(
                health = 20,
                weaponState = weaponState,
            )

        val monster = Card(Suit.SPADES, Rank.EIGHT) // 8♠ (too strong for weapon)
        val afterCombat = game.fightMonster(monster)

        // Fight was barehanded: full damage
        assertEquals(12, afterCombat.health, "Barehanded: took full 8 damage")
        // Weapon still equipped with same max value
        val weaponStateAfter = requireNotNull(afterCombat.weaponState)
        assertEquals(4, weaponStateAfter.maxMonsterValue, "Weapon unchanged")
        assertEquals(weapon, weaponStateAfter.weapon, "Same weapon equipped")
    }
}
