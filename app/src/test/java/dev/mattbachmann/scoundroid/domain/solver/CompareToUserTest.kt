package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.WeaponState
import kotlin.math.pow
import org.junit.Test
import kotlin.random.Random

class CompareToUserTest {
    @Test
    fun `trace optimized bot on seed 1768180694612`() {
        val seed = 1768180694612L
        println("=== OPTIMIZED BOT TRACE: Seed $seed ===\n")

        val game = GameState.newGame(Random(seed))
        val trace = OptimizedTracingPlayer()
        trace.playGameWithTrace(game)
    }
}

/**
 * Tracing player with all current optimizations.
 */
class OptimizedTracingPlayer {
    private var roomNumber = 0
    private val genome = PlayerGenome(
        skipIfDamageExceedsHealthMinus = 5,
        skipWithoutWeaponDamageFraction = 0.444,
        skipDamageHealthFraction = 0.4,
        monsterLeavePenaltyMultiplier = 0.894,
        weaponLeavePenaltyIfNeeded = 2.505,
        potionLeavePenaltyPerRemaining = 0.5,
        weaponPreservationThreshold = 10,
        minDamageSavedToUseWeapon = 0,
        emergencyHealthBuffer = 0,
        equipFreshWeaponIfDegradedBelow = 10,
        alwaysSwapToFreshIfDegradedBelow = 8,
    )

    fun playGameWithTrace(initialState: GameState) {
        var state = initialState

        println("Initial deck (${state.deck.cards.size} cards):")
        state.deck.cards.chunked(4).forEachIndexed { i, chunk ->
            println("  ${i * 4 + 1}-${i * 4 + chunk.size}: ${chunk.map { formatCard(it) }}")
        }
        println()

        while (!state.isGameOver && !isActuallyWon(state)) {
            state = playOneStepWithTrace(state)
        }

        println("\n=== FINAL RESULT ===")
        println("Health: ${state.health}")
        println("Score: ${state.calculateScore()}")
        println("Won: ${state.isGameWon}")
    }

    private fun playOneStepWithTrace(state: GameState): GameState {
        if (state.currentRoom == null || state.currentRoom.isEmpty()) {
            return state.drawRoom()
        }

        val room = state.currentRoom

        if (room.size < GameState.ROOM_SIZE && !state.deck.isEmpty) {
            return state.drawRoom()
        }

        if (room.size == GameState.ROOM_SIZE) {
            roomNumber++
            println("--- Room $roomNumber ---")
            println(
                "Health: ${state.health}, Weapon: ${
                    state.weaponState?.let {
                        "${formatCard(it.weapon)} (max: ${it.maxMonsterValue ?: "fresh"})"
                    } ?: "none"
                }"
            )
            println("Cards: ${room.map { formatCard(it) }}")

            val shouldSkip = !state.lastRoomAvoided && shouldSkipRoom(state, room)
            if (shouldSkip) {
                println("DECISION: SKIP ROOM")
                return state.avoidRoom()
            }

            return processRoomWithTrace(state, room)
        }

        // End game
        roomNumber++
        println("--- Final Room $roomNumber (${room.size} cards) ---")
        println(
            "Health: ${state.health}, Weapon: ${
                state.weaponState?.let {
                    "${formatCard(it.weapon)} (max: ${it.maxMonsterValue ?: "fresh"})"
                } ?: "none"
            }"
        )
        println("Cards: ${room.map { formatCard(it) }}")
        return processEndGameWithTrace(state, room)
    }

    private fun shouldSkipRoom(state: GameState, room: List<Card>): Boolean {
        val monsters = room.filter { it.type == CardType.MONSTER }
        if (monsters.isEmpty()) return false

        val bestCardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != bestCardToLeave }
        val estimatedNetDamage = simulateNetDamage(state, cardsToProcess)

        // Skip if damage exceeds health fraction
        if (estimatedNetDamage > state.health * genome.skipDamageHealthFraction) {
            println("  Skip: damage $estimatedNetDamage > ${(genome.skipDamageHealthFraction * 100).toInt()}% of health ${state.health}")
            return true
        }

        if (estimatedNetDamage >= state.health - genome.skipIfDamageExceedsHealthMinus) {
            println("  Skip: damage $estimatedNetDamage >= health ${state.health} - ${genome.skipIfDamageExceedsHealthMinus}")
            return true
        }

        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful = state.weaponState?.let { ws ->
            monsters.any { ws.canDefeat(it) }
        } ?: false

        if (!currentWeaponUseful && !hasWeaponInRoom) {
            if (estimatedNetDamage > state.health * genome.skipWithoutWeaponDamageFraction) {
                println("  Skip: no weapon help, damage $estimatedNetDamage > ${(genome.skipWithoutWeaponDamageFraction * 100).toInt()}% health")
                return true
            }
        }

        println("  Process: estimated damage $estimatedNetDamage, health ${state.health}")
        return false
    }

    private fun chooseCardToLeave(state: GameState, room: List<Card>): Card {
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

    private fun evaluateLeaveChoice(state: GameState, cardToLeave: Card, cardsToProcess: List<Card>): Double {
        val netDamage = simulateNetDamage(state, cardsToProcess)
        val leftoverPenalty = when (cardToLeave.type) {
            CardType.MONSTER -> cardToLeave.value.toDouble().pow(genome.monsterLeavePenaltyMultiplier)
            CardType.POTION -> {
                val potionsInDeck = state.deck.cards.count { it.type == CardType.POTION }
                potionsInDeck * genome.potionLeavePenaltyPerRemaining
            }
            CardType.WEAPON -> {
                val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                if (cardToLeave.value > currentWeaponValue) genome.weaponLeavePenaltyIfNeeded else 0.0
            }
        }
        return netDamage + leftoverPenalty
    }

    private fun simulateNetDamage(state: GameState, cards: List<Card>): Int {
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
        var weaponMaxMonster: Int? = if (weaponIsFresh) null else currentWeapon?.maxMonsterValue

        for (monster in monsters) {
            val canUseWeapon = effectiveWeaponValue > 0 &&
                (weaponMaxMonster == null || monster.value <= weaponMaxMonster)

            if (canUseWeapon) {
                val shouldUseWeapon = !weaponIsFresh || monster.value >= genome.weaponPreservationThreshold

                if (shouldUseWeapon) {
                    val damage = maxOf(0, monster.value - effectiveWeaponValue)
                    totalDamage += damage
                    weaponMaxMonster = monster.value
                    weaponIsFresh = false
                } else {
                    totalDamage += monster.value
                }
            } else {
                totalDamage += monster.value
            }
        }

        val healthDeficit = GameState.MAX_HEALTH - state.health + totalDamage
        val totalPotionValue = potions.sumOf { it.value }
        val effectiveHealing = minOf(totalPotionValue, healthDeficit.coerceAtLeast(0))

        return totalDamage - effectiveHealing
    }

    private fun shouldEquipWeapon(current: WeaponState?, newWeapon: Card): Boolean {
        if (current == null) return true

        val currentValue = current.weapon.value
        val currentMaxMonster = current.maxMonsterValue

        if (newWeapon.value > currentValue) return true

        // Always swap to ANY fresh weapon if current is severely degraded
        if (currentMaxMonster != null && currentMaxMonster < genome.alwaysSwapToFreshIfDegradedBelow) {
            return true
        }

        // Equip fresh weapon if current is degraded below threshold and new weapon is decent
        if (currentMaxMonster != null && currentMaxMonster < genome.equipFreshWeaponIfDegradedBelow) {
            return newWeapon.value >= currentMaxMonster
        }

        return false
    }

    private fun processRoomWithTrace(state: GameState, room: List<Card>): GameState {
        val cardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != cardToLeave }

        println("DECISION: Leave ${formatCard(cardToLeave)}, process ${cardsToProcess.map { formatCard(it) }}")

        val orderedCards = orderCardsForProcessing(state, cardsToProcess)

        var currentState = state.copy(
            currentRoom = listOf(cardToLeave),
            usedPotionThisTurn = false,
        )

        for (card in orderedCards) {
            val beforeHealth = currentState.health
            currentState = processCardWithTrace(currentState, card)
            val afterHealth = currentState.health
            if (beforeHealth != afterHealth) {
                println("  Health: $beforeHealth → $afterHealth")
            }
            if (currentState.isGameOver) {
                println("  *** GAME OVER ***")
                return currentState
            }
        }

        return currentState
    }

    private fun orderCardsForProcessing(state: GameState, cards: List<Card>): List<Card> {
        val weapons = cards.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
        val potions = cards.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }

        val result = mutableListOf<Card>()
        val bestNewWeapon = weapons.firstOrNull { shouldEquipWeapon(state.weaponState, it) }
        val effectiveWeaponValue = bestNewWeapon?.value ?: state.weaponState?.weapon?.value ?: 0

        if (bestNewWeapon != null) result.add(bestNewWeapon)
        result.addAll(weapons.filter { it !in result })

        val estimatedDamage = monsters.sumOf {
            if (effectiveWeaponValue > 0) maxOf(0, it.value - effectiveWeaponValue) else it.value
        }
        val needsHealingFirst = state.health <= estimatedDamage / 2

        if (needsHealingFirst && potions.isNotEmpty()) {
            result.add(potions.first())
        }

        result.addAll(monsters)
        result.addAll(potions.filter { it !in result })

        return result
    }

    private fun processCardWithTrace(state: GameState, card: Card): GameState = when (card.type) {
        CardType.MONSTER -> processCombatWithTrace(state, card)
        CardType.WEAPON -> {
            if (shouldEquipWeapon(state.weaponState, card)) {
                val oldWeapon = state.weaponState
                println("  Equip ${formatCard(card)}${oldWeapon?.let { " (replacing ${formatCard(it.weapon)} max=${it.maxMonsterValue})" } ?: ""}")
                state.equipWeapon(card)
            } else {
                println("  Skip weapon ${formatCard(card)}")
                state
            }
        }
        CardType.POTION -> {
            println("  Use ${formatCard(card)}")
            state.usePotion(card)
        }
    }

    private fun processCombatWithTrace(state: GameState, monster: Card): GameState {
        val weapon = state.weaponState

        if (weapon == null || !weapon.canDefeat(monster)) {
            println("  Fight ${formatCard(monster)} BAREHANDED (${if (weapon == null) "no weapon" else "weapon max=${weapon.maxMonsterValue}, can't hit ${monster.value}"})")
            return state.fightMonsterBarehanded(monster)
        }

        val weaponIsFresh = weapon.maxMonsterValue == null
        val shouldUse = when {
            state.health <= monster.value + genome.emergencyHealthBuffer -> true
            weaponIsFresh -> monster.value >= genome.weaponPreservationThreshold
            else -> true
        }

        return if (shouldUse) {
            val damage = maxOf(0, monster.value - weapon.weapon.value)
            println("  Fight ${formatCard(monster)} with ${formatCard(weapon.weapon)} → $damage damage")
            state.fightMonsterWithWeapon(monster)
        } else {
            println("  Fight ${formatCard(monster)} BAREHANDED (preserve fresh weapon)")
            state.fightMonsterBarehanded(monster)
        }
    }

    private fun processEndGameWithTrace(state: GameState, room: List<Card>): GameState {
        println("DECISION: Process all ${room.map { formatCard(it) }}")
        val orderedCards = orderCardsForProcessing(state, room)

        var currentState = state.copy(currentRoom = null)

        for (card in orderedCards) {
            val beforeHealth = currentState.health
            currentState = processCardWithTrace(currentState, card)
            val afterHealth = currentState.health
            if (beforeHealth != afterHealth) {
                println("  Health: $beforeHealth → $afterHealth")
            }
            if (currentState.isGameOver) {
                println("  *** GAME OVER ***")
                return currentState
            }
        }

        return currentState
    }

    private fun formatCard(card: Card): String = "${card.rank.displayName}${card.suit.symbol}"

    private fun isActuallyWon(state: GameState): Boolean =
        state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0
}
