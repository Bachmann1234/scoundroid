package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.WeaponState
import kotlin.math.pow

/**
 * A player that uses tunable genome parameters for all decisions.
 * Can be used with a genetic algorithm to evolve optimal play.
 */
class ParameterizedPlayer(
    private val genome: PlayerGenome,
) {
    fun playGame(initialState: GameState): GameState {
        var state = initialState

        while (!state.isGameOver && !isActuallyWon(state)) {
            state = playOneStep(state)
        }

        return state
    }

    private fun playOneStep(state: GameState): GameState {
        if (state.currentRoom == null || state.currentRoom.isEmpty()) {
            return state.drawRoom()
        }

        val room = state.currentRoom

        if (room.size < GameState.ROOM_SIZE && !state.deck.isEmpty) {
            return state.drawRoom()
        }

        if (room.size == GameState.ROOM_SIZE) {
            if (!state.lastRoomAvoided && shouldSkipRoom(state, room)) {
                return state.avoidRoom()
            }
            return processRoom(state, room)
        }

        return processEndGame(state, room)
    }

    private fun shouldSkipRoom(
        state: GameState,
        room: List<Card>,
    ): Boolean {
        val monsters = room.filter { it.type == CardType.MONSTER }
        if (monsters.isEmpty()) return false

        val bestCardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != bestCardToLeave }
        val estimatedNetDamage = simulateNetDamage(state, cardsToProcess)

        // Skip if damage would kill us or leave us near death
        if (estimatedNetDamage >= state.health - genome.skipIfDamageExceedsHealthMinus) {
            return true
        }

        // NEW: Skip if damage exceeds threshold fraction of current health ("too much damage")
        if (estimatedNetDamage > state.health * genome.skipDamageHealthFraction) {
            return true
        }

        // Check if we have weapon help
        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful =
            state.weaponState?.let { ws ->
                monsters.any { ws.canDefeat(it) }
            } ?: false

        // More aggressive skip without weapon
        if (!currentWeaponUseful && !hasWeaponInRoom) {
            if (estimatedNetDamage > state.health * genome.skipWithoutWeaponDamageFraction) {
                return true
            }
        }

        return false
    }

    private fun processRoom(
        state: GameState,
        room: List<Card>,
    ): GameState {
        val cardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != cardToLeave }
        val orderedCards = orderCardsForProcessing(state, cardsToProcess)

        var currentState =
            state.copy(
                currentRoom = listOf(cardToLeave),
                usedPotionThisTurn = false,
            )

        for (card in orderedCards) {
            currentState = processCard(currentState, card)
            if (currentState.isGameOver) {
                return currentState
            }
        }

        return currentState
    }

    private fun chooseCardToLeave(
        state: GameState,
        room: List<Card>,
    ): Card {
        var bestCardToLeave = room.first()
        var bestScore = Double.MAX_VALUE

        for (candidate in room) {
            val cardsToProcess = room.filter { it != candidate }
            val score = evaluateLeaveChoice(state, candidate, cardsToProcess)

            if (score < bestScore) {
                bestScore = score
                bestCardToLeave = candidate
            }
        }

        return bestCardToLeave
    }

    private fun simulateNetDamage(
        state: GameState,
        cards: List<Card>,
    ): Int {
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }
        val weapons = cards.filter { it.type == CardType.WEAPON }
        val potions = cards.filter { it.type == CardType.POTION }

        val currentWeapon = state.weaponState
        val bestNewWeapon = weapons.maxByOrNull { it.value }

        val effectiveWeaponValue: Int
        var weaponIsFresh: Boolean

        if (bestNewWeapon != null && shouldEquipWeapon(currentWeapon, bestNewWeapon)) {
            effectiveWeaponValue = bestNewWeapon.value
            weaponIsFresh = true
        } else if (currentWeapon != null) {
            effectiveWeaponValue = currentWeapon.weapon.value
            weaponIsFresh = currentWeapon.maxMonsterValue == null
        } else {
            effectiveWeaponValue = 0
            weaponIsFresh = false
        }

        var totalDamage = 0
        var simulatedHealth = state.health
        var weaponMaxMonster: Int? = if (weaponIsFresh) null else currentWeapon?.maxMonsterValue

        for (monster in monsters) {
            val canUseWeapon =
                effectiveWeaponValue > 0 &&
                    (weaponMaxMonster == null || monster.value <= weaponMaxMonster)

            if (canUseWeapon) {
                val damageSaved = minOf(effectiveWeaponValue, monster.value)
                val shouldUseWeapon =
                    shouldUseWeaponOnMonster(
                        monsterValue = monster.value,
                        weaponIsFresh = weaponIsFresh,
                        damageSaved = damageSaved,
                        currentHealth = simulatedHealth,
                    )

                if (shouldUseWeapon) {
                    val damage = maxOf(0, monster.value - effectiveWeaponValue)
                    totalDamage += damage
                    simulatedHealth -= damage
                    weaponMaxMonster = monster.value
                    weaponIsFresh = false
                } else {
                    totalDamage += monster.value
                    simulatedHealth -= monster.value
                }
            } else {
                totalDamage += monster.value
                simulatedHealth -= monster.value
            }
        }

        val healthDeficit = GameState.MAX_HEALTH - state.health + totalDamage
        val totalPotionValue = potions.sumOf { it.value }
        val effectiveHealing = minOf(totalPotionValue, healthDeficit.coerceAtLeast(0))

        return totalDamage - effectiveHealing
    }

    private fun evaluateLeaveChoice(
        state: GameState,
        cardToLeave: Card,
        cardsToProcess: List<Card>,
    ): Double {
        val netDamage = simulateNetDamage(state, cardsToProcess)

        val leftoverPenalty =
            when (cardToLeave.type) {
                CardType.MONSTER -> {
                    // Use multiplier for non-linear penalty (e.g., 2.0 = quadratic)
                    cardToLeave.value.toDouble().pow(genome.monsterLeavePenaltyMultiplier)
                }
                CardType.POTION -> {
                    // Penalty for leaving a potion based on how many potions remain in deck
                    // More potions remaining = higher chance of potion cascade
                    val potionsInDeck = state.deck.cards.count { it.type == CardType.POTION }
                    potionsInDeck * genome.potionLeavePenaltyPerRemaining
                }
                CardType.WEAPON -> {
                    val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                    if (cardToLeave.value > currentWeaponValue) {
                        genome.weaponLeavePenaltyIfNeeded
                    } else {
                        0.0
                    }
                }
            }

        return netDamage + leftoverPenalty
    }

    private fun shouldEquipWeapon(
        current: WeaponState?,
        newWeapon: Card,
    ): Boolean {
        if (current == null) return true

        val currentValue = current.weapon.value
        val currentMaxMonster = current.maxMonsterValue

        if (newWeapon.value > currentValue) return true

        // Always swap to ANY fresh weapon if current is severely degraded
        // A fresh 5 that can hit anything beats a degraded 9 that can only hit small monsters
        if (currentMaxMonster != null && currentMaxMonster < genome.alwaysSwapToFreshIfDegradedBelow) {
            return true
        }

        // Equip fresh weapon if current is degraded below threshold and new weapon is decent
        if (currentMaxMonster != null && currentMaxMonster < genome.equipFreshWeaponIfDegradedBelow) {
            return newWeapon.value >= currentMaxMonster
        }

        return false
    }

    private fun shouldUseWeaponOnMonster(
        monsterValue: Int,
        weaponIsFresh: Boolean,
        damageSaved: Int,
        currentHealth: Int,
    ): Boolean {
        // Emergency: always use weapon if we might die
        if (currentHealth <= monsterValue + genome.emergencyHealthBuffer) {
            return true
        }

        // Fresh weapon: only use on big monsters
        if (weaponIsFresh) {
            return monsterValue >= genome.weaponPreservationThreshold
        }

        // Degraded weapon: use if it saves enough damage
        return damageSaved >= genome.minDamageSavedToUseWeapon
    }

    private fun orderCardsForProcessing(
        state: GameState,
        cards: List<Card>,
    ): List<Card> {
        val weapons = cards.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
        val potions = cards.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }

        val result = mutableListOf<Card>()

        val bestNewWeapon = weapons.firstOrNull { shouldEquipWeapon(state.weaponState, it) }

        val effectiveWeaponValue =
            if (bestNewWeapon != null) {
                bestNewWeapon.value
            } else {
                state.weaponState?.weapon?.value ?: 0
            }

        if (bestNewWeapon != null) {
            result.add(bestNewWeapon)
        }

        result.addAll(weapons.filter { it !in result })

        // Heal first if health is critically low
        val estimatedDamage =
            monsters.sumOf {
                if (effectiveWeaponValue > 0) maxOf(0, it.value - effectiveWeaponValue) else it.value
            }
        if (state.health <= estimatedDamage / 2 && potions.isNotEmpty()) {
            result.add(potions.first())
        }

        result.addAll(monsters)
        result.addAll(potions.filter { it !in result })

        return result
    }

    private fun processCard(
        state: GameState,
        card: Card,
    ): GameState =
        when (card.type) {
            CardType.MONSTER -> processCombat(state, card)
            CardType.WEAPON -> {
                if (shouldEquipWeapon(state.weaponState, card)) {
                    state.equipWeapon(card)
                } else {
                    state
                }
            }
            CardType.POTION -> state.usePotion(card)
        }

    private fun processCombat(
        state: GameState,
        monster: Card,
    ): GameState {
        val weapon = state.weaponState

        if (weapon == null || !weapon.canDefeat(monster)) {
            return state.fightMonsterBarehanded(monster)
        }

        val weaponIsFresh = weapon.maxMonsterValue == null
        val damageSaved = minOf(weapon.weapon.value, monster.value)

        val shouldUse =
            shouldUseWeaponOnMonster(
                monsterValue = monster.value,
                weaponIsFresh = weaponIsFresh,
                damageSaved = damageSaved,
                currentHealth = state.health,
            )

        return if (shouldUse) {
            state.fightMonsterWithWeapon(monster)
        } else {
            state.fightMonsterBarehanded(monster)
        }
    }

    private fun processEndGame(
        state: GameState,
        room: List<Card>,
    ): GameState {
        val orderedCards = orderCardsForProcessing(state, room)
        var currentState = state.copy(currentRoom = null)

        for (card in orderedCards) {
            currentState = processCard(currentState, card)
            if (currentState.isGameOver) {
                return currentState
            }
        }

        return currentState
    }

    private fun isActuallyWon(state: GameState): Boolean =
        state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0
}
