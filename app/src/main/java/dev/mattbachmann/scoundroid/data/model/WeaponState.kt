package dev.mattbachmann.scoundroid.data.model

/**
 * Tracks the state of an equipped weapon, including degradation.
 *
 * Critical rule: Once a weapon is used on a monster, it can ONLY be used on monsters
 * with value ≤ the last monster it defeated. The weapon degrades to the value of
 * the last monster defeated.
 *
 * @property weapon The weapon card
 * @property maxMonsterValue The maximum monster value this weapon can defeat.
 *                          Null means unrestricted (new weapon).
 */
data class WeaponState(
    val weapon: Card,
    val maxMonsterValue: Int? = null,
) {
    init {
        require(weapon.type == CardType.WEAPON) {
            "WeaponState can only be created with a weapon card"
        }
    }

    /**
     * Checks if this weapon can defeat the given monster.
     *
     * A weapon can defeat a monster if:
     * - The target is a monster card
     * - The weapon is unrestricted (maxMonsterValue is null), OR
     * - The monster's value is ≤ the weapon's maxMonsterValue
     *
     * @param monster The monster card to check
     * @return true if this weapon can defeat the monster
     */
    fun canDefeat(monster: Card): Boolean {
        // Can only use weapons on monsters
        if (monster.type != CardType.MONSTER) {
            return false
        }

        // Unrestricted weapon can defeat any monster
        if (maxMonsterValue == null) {
            return true
        }

        // Otherwise, monster value must be ≤ max
        return monster.value <= maxMonsterValue
    }

    /**
     * Uses this weapon on a monster, returning a new WeaponState with updated degradation.
     *
     * When a weapon is used on a monster, the weapon's maxMonsterValue is set to the
     * monster's value. If the weapon already had a maxMonsterValue, it will degrade
     * if the new monster has a lower value.
     *
     * @param monster The monster being defeated
     * @return A new WeaponState with updated maxMonsterValue
     */
    fun useOn(monster: Card): WeaponState {
        require(monster.type == CardType.MONSTER) {
            "Weapon can only be used on monsters"
        }

        return copy(maxMonsterValue = monster.value)
    }
}
