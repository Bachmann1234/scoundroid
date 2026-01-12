package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.WeaponState

/**
 * A version of HeuristicPlayer that logs all decisions for analysis.
 * Uses the same strategy as HeuristicPlayer but records events.
 */
class LoggingHeuristicPlayer {
    companion object {
        /** Minimum monster value to use a fresh weapon on. Evolved: 8 → 9 */
        const val WEAPON_PRESERVATION_THRESHOLD = 9

        /** Skip if damage >= health - this. Evolved: 2 → 5 */
        const val SKIP_DAMAGE_HEALTH_BUFFER = 5

        /** Skip without weapon if damage > this fraction of health. Evolved: 0.5 → 0.444 */
        const val SKIP_WITHOUT_WEAPON_FRACTION = 0.444

        /** Equip fresh weapon if degraded below this. Evolved: 6 → 10 */
        const val EQUIP_FRESH_IF_DEGRADED_BELOW = 10
    }

    private val events = mutableListOf<GameEvent>()
    private var roomNumber = 0
    private var roomsSkipped = 0
    private var potionsUsed = 0
    private var lastMonsterFought: Card? = null
    private var healthBeforeLastHit = 0
    private var damageFromLastHit = 0

    fun playGame(
        seed: Long,
        initialState: GameState,
    ): GameLog {
        events.clear()
        roomNumber = 0
        roomsSkipped = 0
        potionsUsed = 0
        lastMonsterFought = null

        var state = initialState

        while (!state.isGameOver && !isActuallyWon(state)) {
            state = playOneStep(state)
        }

        val won =
            state.health > 0 &&
                state.deck.isEmpty &&
                (state.currentRoom == null || state.currentRoom.isEmpty())
        val score = state.calculateScore()

        events.add(GameEvent.GameEnded(won, state.health, score))

        val deathInfo =
            if (!won && lastMonsterFought != null) {
                DeathInfo(
                    killerMonster = lastMonsterFought!!,
                    healthBeforeHit = healthBeforeLastHit,
                    damageTaken = damageFromLastHit,
                    hadWeapon = state.weaponState != null,
                    weaponValue = state.weaponState?.weapon?.value,
                    weaponMaxMonster = state.weaponState?.maxMonsterValue,
                    couldWeaponHaveHelped =
                        state.weaponState?.let { ws ->
                            // Could weapon have reduced damage if not degraded?
                            ws.weapon.value > 0 && !ws.canDefeat(lastMonsterFought!!)
                        } ?: false,
                    roomsSkipped = roomsSkipped,
                    potionsUsedThisGame = potionsUsed,
                )
            } else {
                null
            }

        return GameLog(
            seed = seed,
            won = won,
            finalHealth = state.health,
            finalScore = score,
            cardsRemainingInDeck = state.deck.size,
            events = events.toList(),
            deathInfo = deathInfo,
        )
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
            roomNumber++
            events.add(
                GameEvent.RoomDrawn(
                    roomNumber = roomNumber,
                    cards = room.toList(),
                    health = state.health,
                    deckRemaining = state.deck.size,
                    weaponState = state.weaponState,
                ),
            )

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

        if (estimatedNetDamage >= state.health - SKIP_DAMAGE_HEALTH_BUFFER) {
            roomsSkipped++
            events.add(
                GameEvent.RoomSkipped(
                    roomNumber = roomNumber,
                    cards = room.toList(),
                    estimatedDamage = estimatedNetDamage,
                    health = state.health,
                    reason = "Would kill or leave dangerously low (damage=$estimatedNetDamage, health=${state.health})",
                ),
            )
            return true
        }

        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful =
            state.weaponState?.let { ws ->
                monsters.any { ws.canDefeat(it) }
            } ?: false

        if (!currentWeaponUseful && !hasWeaponInRoom) {
            if (estimatedNetDamage > state.health * SKIP_WITHOUT_WEAPON_FRACTION) {
                roomsSkipped++
                events.add(
                    GameEvent.RoomSkipped(
                        roomNumber = roomNumber,
                        cards = room.toList(),
                        estimatedDamage = estimatedNetDamage,
                        health = state.health,
                        reason = "No weapon help and damage > ${(SKIP_WITHOUT_WEAPON_FRACTION * 100).toInt()}% health",
                    ),
                )
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

        val reason =
            when (cardToLeave.type) {
                CardType.MONSTER -> "Leaving monster (value=${cardToLeave.value}) to fight later"
                CardType.POTION -> "Leaving potion for next room"
                CardType.WEAPON -> "Leaving weapon (already have better or equal)"
            }

        events.add(
            GameEvent.CardLeftBehind(
                roomNumber = roomNumber,
                cardLeft = cardToLeave,
                cardsProcessed = cardsToProcess,
                reason = reason,
            ),
        )

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
        var bestScore = Int.MAX_VALUE

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
                // Apply weapon preservation logic (no emergency buffer per GA)
                val shouldUseWeapon =
                    !weaponIsFresh ||
                        monster.value >= WEAPON_PRESERVATION_THRESHOLD

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
    ): Int {
        val netDamage = simulateNetDamage(state, cardsToProcess)

        val leftoverPenalty =
            when (cardToLeave.type) {
                CardType.MONSTER -> cardToLeave.value
                CardType.POTION -> 0
                CardType.WEAPON -> {
                    val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                    if (cardToLeave.value > currentWeaponValue) cardToLeave.value else 0
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

        // Equip fresh weapon if current is degraded below threshold
        if (currentMaxMonster != null && currentMaxMonster < EQUIP_FRESH_IF_DEGRADED_BELOW) {
            return newWeapon.value >= currentMaxMonster
        }

        return false
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

        val estimatedDamage = monsters.sumOf { estimateDamage(it, effectiveWeaponValue) }
        val needsHealingFirst = state.health <= estimatedDamage / 2

        if (needsHealingFirst && potions.isNotEmpty()) {
            result.add(potions.first())
        }

        result.addAll(monsters)
        result.addAll(potions.filter { it !in result })

        return result
    }

    private fun estimateDamage(
        monster: Card,
        weaponValue: Int,
    ): Int =
        if (weaponValue > 0) {
            (monster.value - weaponValue).coerceAtLeast(0)
        } else {
            monster.value
        }

    private fun processCard(
        state: GameState,
        card: Card,
    ): GameState =
        when (card.type) {
            CardType.MONSTER -> processCombat(state, card)
            CardType.WEAPON -> processWeapon(state, card)
            CardType.POTION -> processPotion(state, card)
        }

    private fun processWeapon(
        state: GameState,
        weapon: Card,
    ): GameState =
        if (shouldEquipWeapon(state.weaponState, weapon)) {
            val previousWeapon = state.weaponState?.weapon
            val wasDegraded = state.weaponState?.maxMonsterValue != null

            events.add(
                GameEvent.WeaponEquipped(
                    weapon = weapon,
                    previousWeapon = previousWeapon,
                    previousWeaponDegraded = wasDegraded,
                ),
            )

            state.equipWeapon(weapon)
        } else {
            events.add(
                GameEvent.WeaponSkipped(
                    weapon = weapon,
                    currentWeapon = state.weaponState!!.weapon,
                    reason = "Current weapon is better or equal",
                ),
            )
            state
        }

    private fun processPotion(
        state: GameState,
        potion: Card,
    ): GameState {
        val healthBefore = state.health
        val newState = state.usePotion(potion)

        if (state.usedPotionThisTurn) {
            events.add(
                GameEvent.PotionWasted(
                    potion = potion,
                    reason = "Already used potion this turn",
                ),
            )
        } else {
            potionsUsed++
            val healingWasted = (healthBefore + potion.value) - minOf(healthBefore + potion.value, GameState.MAX_HEALTH)
            events.add(
                GameEvent.PotionUsed(
                    potion = potion,
                    healthBefore = healthBefore,
                    healthAfter = newState.health,
                    healingWasted = healingWasted,
                ),
            )
        }

        return newState
    }

    private fun processCombat(
        state: GameState,
        monster: Card,
    ): GameState {
        val weapon = state.weaponState
        val healthBefore = state.health

        lastMonsterFought = monster
        healthBeforeLastHit = healthBefore

        // No weapon or weapon can't hit this monster = barehanded
        if (weapon == null || !weapon.canDefeat(monster)) {
            val newState = state.fightMonsterBarehanded(monster)
            val damage = monster.value

            damageFromLastHit = damage

            events.add(
                GameEvent.CombatResolved(
                    monster = monster,
                    usedWeapon = false,
                    weaponValue = weapon?.weapon?.value,
                    damageTaken = damage,
                    healthBefore = healthBefore,
                    healthAfter = newState.health,
                ),
            )

            return newState
        }

        // Apply weapon preservation strategy (GA evolved emergencyHealthBuffer to 0)
        val weaponIsDegraded = weapon.maxMonsterValue != null
        val monsterIsBigEnough = monster.value >= WEAPON_PRESERVATION_THRESHOLD

        // Use weapon if already degraded (use it or lose it) or monster is big enough
        val shouldUseWeapon = weaponIsDegraded || monsterIsBigEnough

        if (shouldUseWeapon) {
            val newState = state.fightMonsterWithWeapon(monster)
            val damage = maxOf(0, monster.value - weapon.weapon.value)

            damageFromLastHit = damage

            events.add(
                GameEvent.CombatResolved(
                    monster = monster,
                    usedWeapon = true,
                    weaponValue = weapon.weapon.value,
                    damageTaken = damage,
                    healthBefore = healthBefore,
                    healthAfter = newState.health,
                ),
            )

            return newState
        }

        // Fight barehanded to preserve weapon
        val newState = state.fightMonsterBarehanded(monster)
        val damage = monster.value

        damageFromLastHit = damage

        events.add(
            GameEvent.CombatResolved(
                monster = monster,
                usedWeapon = false,
                weaponValue = weapon.weapon.value,
                damageTaken = damage,
                healthBefore = healthBefore,
                healthAfter = newState.health,
            ),
        )

        return newState
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

/**
 * Simulator that uses the logging player and collects statistics.
 */
class LoggingSimulator {
    fun simulate(
        seedRange: LongRange,
        collectDetailedLogs: Int = 100,
    ): SimulationAnalysis {
        val player = LoggingHeuristicPlayer()
        val statsBuilder = AggregateStatsBuilder()
        val detailedLogs = mutableListOf<GameLog>()
        var detailedLossCount = 0

        for (seed in seedRange) {
            val game = GameState.newGame(kotlin.random.Random(seed))
            val log = player.playGame(seed, game)

            statsBuilder.addGame(log)

            // Collect detailed logs for a sample of losses
            if (!log.won && detailedLossCount < collectDetailedLogs) {
                detailedLogs.add(log)
                detailedLossCount++
            }
        }

        return SimulationAnalysis(
            stats = statsBuilder.build(),
            sampleLogs = detailedLogs,
        )
    }
}

data class SimulationAnalysis(
    val stats: AggregateStats,
    val sampleLogs: List<GameLog>,
)
