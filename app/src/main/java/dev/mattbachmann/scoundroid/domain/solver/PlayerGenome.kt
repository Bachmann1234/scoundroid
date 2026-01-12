package dev.mattbachmann.scoundroid.domain.solver

import kotlin.random.Random

/**
 * Tunable parameters for the heuristic player.
 * These values can be evolved by a genetic algorithm to find optimal play.
 */
data class PlayerGenome(
    // Room skip thresholds
    val skipIfDamageExceedsHealthMinus: Int, // Skip if damage >= health - this value
    val skipWithoutWeaponDamageFraction: Double, // Skip if no weapon help and damage > this fraction of health
    // Card leave evaluation
    val monsterLeavePenaltyMultiplier: Double, // Penalty multiplier for leaving monsters (1.0 = linear)
    val weaponLeavePenaltyIfNeeded: Double, // Penalty for leaving a weapon we need
    // Weapon preservation (CRITICAL - most impactful parameters)
    val weaponPreservationThreshold: Int, // Only use fresh weapon on monsters >= this value
    val minDamageSavedToUseWeapon: Int, // Only use degraded weapon if it saves >= this damage
    val emergencyHealthBuffer: Int, // Use weapon if health <= monster.value + this buffer
    // Weapon equip decisions
    val equipFreshWeaponIfDegradedBelow: Int, // Equip fresh weapon if current is degraded below this
) {
    companion object {
        /** Parameter bounds for random generation and mutation */
        val BOUNDS =
            GenomeBounds(
                skipIfDamageExceedsHealthMinus = 0..10,
                skipWithoutWeaponDamageFraction = 0.3..0.8,
                monsterLeavePenaltyMultiplier = 0.5..3.0,
                weaponLeavePenaltyIfNeeded = 0.0..20.0,
                weaponPreservationThreshold = 5..12,
                minDamageSavedToUseWeapon = 0..5,
                emergencyHealthBuffer = 0..5,
                equipFreshWeaponIfDegradedBelow = 3..10,
            )

        /** Default genome - evolved via genetic algorithm */
        val DEFAULT =
            PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                monsterLeavePenaltyMultiplier = 0.894,
                weaponLeavePenaltyIfNeeded = 2.505,
                weaponPreservationThreshold = 9,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
            )

        /** Generate a random genome within bounds */
        fun random(rng: Random = Random): PlayerGenome =
            PlayerGenome(
                skipIfDamageExceedsHealthMinus = rng.nextInt(BOUNDS.skipIfDamageExceedsHealthMinus),
                skipWithoutWeaponDamageFraction = rng.nextDouble(BOUNDS.skipWithoutWeaponDamageFraction),
                monsterLeavePenaltyMultiplier = rng.nextDouble(BOUNDS.monsterLeavePenaltyMultiplier),
                weaponLeavePenaltyIfNeeded = rng.nextDouble(BOUNDS.weaponLeavePenaltyIfNeeded),
                weaponPreservationThreshold = rng.nextInt(BOUNDS.weaponPreservationThreshold),
                minDamageSavedToUseWeapon = rng.nextInt(BOUNDS.minDamageSavedToUseWeapon),
                emergencyHealthBuffer = rng.nextInt(BOUNDS.emergencyHealthBuffer),
                equipFreshWeaponIfDegradedBelow = rng.nextInt(BOUNDS.equipFreshWeaponIfDegradedBelow),
            )
    }

    /** Crossover with another genome (blend parameters) */
    fun crossover(
        other: PlayerGenome,
        rng: Random = Random,
    ): PlayerGenome =
        PlayerGenome(
            skipIfDamageExceedsHealthMinus =
                if (rng.nextBoolean()) skipIfDamageExceedsHealthMinus else other.skipIfDamageExceedsHealthMinus,
            skipWithoutWeaponDamageFraction =
                if (rng.nextBoolean()) skipWithoutWeaponDamageFraction else other.skipWithoutWeaponDamageFraction,
            monsterLeavePenaltyMultiplier =
                if (rng.nextBoolean()) monsterLeavePenaltyMultiplier else other.monsterLeavePenaltyMultiplier,
            weaponLeavePenaltyIfNeeded =
                if (rng.nextBoolean()) weaponLeavePenaltyIfNeeded else other.weaponLeavePenaltyIfNeeded,
            weaponPreservationThreshold =
                if (rng.nextBoolean()) weaponPreservationThreshold else other.weaponPreservationThreshold,
            minDamageSavedToUseWeapon =
                if (rng.nextBoolean()) minDamageSavedToUseWeapon else other.minDamageSavedToUseWeapon,
            emergencyHealthBuffer =
                if (rng.nextBoolean()) emergencyHealthBuffer else other.emergencyHealthBuffer,
            equipFreshWeaponIfDegradedBelow =
                if (rng.nextBoolean()) equipFreshWeaponIfDegradedBelow else other.equipFreshWeaponIfDegradedBelow,
        )

    /** Mutate this genome with given probability per parameter */
    fun mutate(
        mutationRate: Double,
        rng: Random = Random,
    ): PlayerGenome =
        PlayerGenome(
            skipIfDamageExceedsHealthMinus =
                if (rng.nextDouble() < mutationRate) {
                    mutateInt(skipIfDamageExceedsHealthMinus, BOUNDS.skipIfDamageExceedsHealthMinus, rng)
                } else {
                    skipIfDamageExceedsHealthMinus
                },
            skipWithoutWeaponDamageFraction =
                if (rng.nextDouble() < mutationRate) {
                    mutateDouble(skipWithoutWeaponDamageFraction, BOUNDS.skipWithoutWeaponDamageFraction, rng)
                } else {
                    skipWithoutWeaponDamageFraction
                },
            monsterLeavePenaltyMultiplier =
                if (rng.nextDouble() < mutationRate) {
                    mutateDouble(monsterLeavePenaltyMultiplier, BOUNDS.monsterLeavePenaltyMultiplier, rng)
                } else {
                    monsterLeavePenaltyMultiplier
                },
            weaponLeavePenaltyIfNeeded =
                if (rng.nextDouble() < mutationRate) {
                    mutateDouble(weaponLeavePenaltyIfNeeded, BOUNDS.weaponLeavePenaltyIfNeeded, rng)
                } else {
                    weaponLeavePenaltyIfNeeded
                },
            weaponPreservationThreshold =
                if (rng.nextDouble() < mutationRate) {
                    mutateInt(weaponPreservationThreshold, BOUNDS.weaponPreservationThreshold, rng)
                } else {
                    weaponPreservationThreshold
                },
            minDamageSavedToUseWeapon =
                if (rng.nextDouble() < mutationRate) {
                    mutateInt(minDamageSavedToUseWeapon, BOUNDS.minDamageSavedToUseWeapon, rng)
                } else {
                    minDamageSavedToUseWeapon
                },
            emergencyHealthBuffer =
                if (rng.nextDouble() < mutationRate) {
                    mutateInt(emergencyHealthBuffer, BOUNDS.emergencyHealthBuffer, rng)
                } else {
                    emergencyHealthBuffer
                },
            equipFreshWeaponIfDegradedBelow =
                if (rng.nextDouble() < mutationRate) {
                    mutateInt(
                        equipFreshWeaponIfDegradedBelow,
                        BOUNDS.equipFreshWeaponIfDegradedBelow,
                        rng,
                    )
                } else {
                    equipFreshWeaponIfDegradedBelow
                },
        )

    private fun mutateInt(
        value: Int,
        bounds: IntRange,
        rng: Random,
    ): Int {
        val delta = rng.nextInt(-2, 3) // -2, -1, 0, 1, 2
        return (value + delta).coerceIn(bounds)
    }

    private fun mutateDouble(
        value: Double,
        bounds: ClosedFloatingPointRange<Double>,
        rng: Random,
    ): Double {
        val range = bounds.endInclusive - bounds.start
        val delta = (rng.nextDouble() - 0.5) * range * 0.3 // ±15% of range
        return (value + delta).coerceIn(bounds)
    }
}

/** Bounds for each genome parameter */
data class GenomeBounds(
    val skipIfDamageExceedsHealthMinus: IntRange,
    val skipWithoutWeaponDamageFraction: ClosedFloatingPointRange<Double>,
    val monsterLeavePenaltyMultiplier: ClosedFloatingPointRange<Double>,
    val weaponLeavePenaltyIfNeeded: ClosedFloatingPointRange<Double>,
    val weaponPreservationThreshold: IntRange,
    val minDamageSavedToUseWeapon: IntRange,
    val emergencyHealthBuffer: IntRange,
    val equipFreshWeaponIfDegradedBelow: IntRange,
)

/** Helper to get random int in range */
private fun Random.nextInt(range: IntRange): Int = nextInt(range.first, range.last + 1)

/** Helper to get random double in range */
private fun Random.nextDouble(range: ClosedFloatingPointRange<Double>): Double =
    nextDouble(range.start, range.endInclusive)
