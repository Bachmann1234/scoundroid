package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.WeaponState

/**
 * A deck-knowledge-aware player that uses card counting to make smarter decisions.
 *
 * Key improvements over HeuristicPlayer:
 * 1. Dynamic weapon preservation threshold based on max monster remaining
 * 2. Smarter room skipping based on survival margin and probability of finding help
 * 3. Better leave-behind decisions considering card scarcity
 * 4. Weapon equip decisions that consider what threats remain
 */
class InformedPlayer {
    companion object {
        /**
         * Base threshold for weapon preservation (from GA-optimized HeuristicPlayer).
         * This gets dynamically adjusted based on what monsters remain.
         */
        const val BASE_WEAPON_PRESERVATION_THRESHOLD = 9

        /**
         * Skip room if estimated damage >= health - this value.
         */
        const val SKIP_DAMAGE_HEALTH_BUFFER = 5

        /**
         * Skip room without weapon help if damage > this fraction of health.
         */
        const val SKIP_WITHOUT_WEAPON_FRACTION = 0.444

        /**
         * Equip a fresh weapon if current is degraded below this value.
         */
        const val EQUIP_FRESH_IF_DEGRADED_BELOW = 10
    }

    /**
     * Plays a complete game using informed decisions with deck tracking.
     * Returns the final game state.
     */
    fun playGame(initialState: GameState): GameState {
        var state = initialState
        var knowledge = DeckKnowledge.initial()

        while (!state.isGameOver && !isActuallyWon(state)) {
            val (newState, newKnowledge) = playOneStep(state, knowledge)
            state = newState
            knowledge = newKnowledge
        }

        return state
    }

    private fun playOneStep(
        state: GameState,
        knowledge: DeckKnowledge,
    ): Pair<GameState, DeckKnowledge> {
        // If no room, draw one
        if (state.currentRoom == null || state.currentRoom.isEmpty()) {
            return Pair(state.drawRoom(), knowledge)
        }

        val room = state.currentRoom

        // If room has < 4 cards and deck has cards, draw more
        if (room.size < GameState.ROOM_SIZE && !state.deck.isEmpty) {
            return Pair(state.drawRoom(), knowledge)
        }

        // If room has 4 cards, decide: avoid or process
        if (room.size == GameState.ROOM_SIZE) {
            // Decide whether to skip this room
            if (!state.lastRoomAvoided && shouldSkipRoom(state, room, knowledge)) {
                val newKnowledge = knowledge.roomSkipped(room)
                return Pair(state.avoidRoom(), newKnowledge)
            }

            // Process the room
            return processRoom(state, room, knowledge)
        }

        // End game: room has < 4 cards and deck is empty
        return processEndGame(state, room, knowledge)
    }

    /**
     * Dynamic weapon preservation threshold based on what monsters remain.
     * If all Aces and Kings are gone, no need to preserve for 14.
     */
    fun getWeaponPreservationThreshold(knowledge: DeckKnowledge): Int =
        minOf(BASE_WEAPON_PRESERVATION_THRESHOLD, knowledge.maxMonsterRemaining)

    /**
     * Decides whether to skip the current room using deck knowledge.
     *
     * Key insight: Skipping only helps if we're likely to find help before
     * facing these cards again. If few weapons/potions remain, skipping
     * just delays the inevitable.
     */
    private fun shouldSkipRoom(
        state: GameState,
        room: List<Card>,
        knowledge: DeckKnowledge,
    ): Boolean {
        val monsters = room.filter { it.type == CardType.MONSTER }

        if (monsters.isEmpty()) return false

        // Calculate estimated damage if we process this room
        val bestCardToLeave = chooseCardToLeave(state, room, knowledge)
        val cardsToProcess = room.filter { it != bestCardToLeave }
        val estimatedNetDamage = simulateNetDamage(state, cardsToProcess, knowledge)

        // Would this room kill us or leave us dangerously low?
        if (estimatedNetDamage >= state.health - SKIP_DAMAGE_HEALTH_BUFFER) {
            return true
        }

        // Would this room take too much health with no weapon help?
        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful =
            state.weaponState?.let { ws ->
                monsters.any { ws.canDefeat(it) }
            } ?: false

        if (!currentWeaponUseful && !hasWeaponInRoom) {
            if (estimatedNetDamage > state.health * SKIP_WITHOUT_WEAPON_FRACTION) {
                return true
            }
        }

        return false
    }

    /**
     * Processes a room of 4 cards with knowledge-informed decisions.
     */
    private fun processRoom(
        state: GameState,
        room: List<Card>,
        knowledge: DeckKnowledge,
    ): Pair<GameState, DeckKnowledge> {
        val cardToLeave = chooseCardToLeave(state, room, knowledge)
        val cardsToProcess = room.filter { it != cardToLeave }

        val orderedCards = orderCardsForProcessing(state, cardsToProcess, knowledge)

        var currentState =
            state.copy(
                currentRoom = listOf(cardToLeave),
                usedPotionThisTurn = false,
            )

        var currentKnowledge = knowledge

        for (card in orderedCards) {
            currentState = processCard(currentState, card, currentKnowledge)
            currentKnowledge = currentKnowledge.cardProcessed(card)
            if (currentState.isGameOver) {
                return Pair(currentState, currentKnowledge)
            }
        }

        return Pair(currentState, currentKnowledge)
    }

    /**
     * Chooses which card to leave for the next room, considering deck knowledge.
     */
    private fun chooseCardToLeave(
        state: GameState,
        room: List<Card>,
        knowledge: DeckKnowledge,
    ): Card {
        var bestCardToLeave = room.first()
        var bestScore = Int.MAX_VALUE

        for (candidate in room) {
            val cardsToProcess = room.filter { it != candidate }
            val score = evaluateLeaveChoice(state, candidate, cardsToProcess, knowledge)

            if (score < bestScore) {
                bestScore = score
                bestCardToLeave = candidate
            }
        }

        return bestCardToLeave
    }

    /**
     * Simulates net damage using knowledge-informed weapon decisions.
     */
    private fun simulateNetDamage(
        state: GameState,
        cards: List<Card>,
        knowledge: DeckKnowledge,
    ): Int {
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }
        val weapons = cards.filter { it.type == CardType.WEAPON }
        val potions = cards.filter { it.type == CardType.POTION }

        val currentWeapon = state.weaponState
        val bestNewWeapon = weapons.maxByOrNull { it.value }

        val effectiveWeaponValue: Int
        var weaponIsFresh: Boolean

        if (bestNewWeapon != null && shouldEquipWeaponWithKnowledge(currentWeapon, bestNewWeapon, knowledge)) {
            effectiveWeaponValue = bestNewWeapon.value
            weaponIsFresh = true
        } else if (currentWeapon != null) {
            effectiveWeaponValue = currentWeapon.weapon.value
            weaponIsFresh = currentWeapon.maxMonsterValue == null
        } else {
            effectiveWeaponValue = 0
            weaponIsFresh = false
        }

        // Use dynamic threshold based on knowledge
        val preservationThreshold = getWeaponPreservationThreshold(knowledge)

        var totalDamage = 0
        var weaponMaxMonster: Int? = if (weaponIsFresh) null else currentWeapon?.maxMonsterValue

        for (monster in monsters) {
            val canUseWeapon =
                effectiveWeaponValue > 0 &&
                    (weaponMaxMonster == null || monster.value <= weaponMaxMonster)

            if (canUseWeapon) {
                val shouldUseWeapon =
                    !weaponIsFresh ||
                        monster.value >= preservationThreshold

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

    /**
     * Evaluates which card to leave, considering deck knowledge for scarcity.
     */
    private fun evaluateLeaveChoice(
        state: GameState,
        cardToLeave: Card,
        cardsToProcess: List<Card>,
        knowledge: DeckKnowledge,
    ): Int {
        val netDamage = simulateNetDamage(state, cardsToProcess, knowledge)

        // Simple leave penalty - same as HeuristicPlayer baseline
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

    /**
     * Determines if we should equip a new weapon.
     * Uses deck knowledge to make smarter decisions.
     */
    fun shouldEquipWeaponWithKnowledge(
        current: WeaponState?,
        newWeapon: Card,
        knowledge: DeckKnowledge,
    ): Boolean {
        if (current == null) return true

        val currentValue = current.weapon.value
        val currentMaxMonster = current.maxMonsterValue

        // New weapon has higher value - definitely equip
        if (newWeapon.value > currentValue) return true

        // Knowledge-based: if current weapon can still handle ALL remaining monsters,
        // no need to swap to a fresh one (even if it's degraded)
        if (currentMaxMonster != null && currentMaxMonster >= knowledge.maxMonsterRemaining) {
            return false // Current weapon is sufficient for remaining threats
        }

        // Current weapon is degraded below threshold - equip a fresh one
        if (currentMaxMonster != null && currentMaxMonster < EQUIP_FRESH_IF_DEGRADED_BELOW) {
            return newWeapon.value >= currentMaxMonster
        }

        return false
    }

    /**
     * Orders cards for optimal processing with knowledge-informed decisions.
     */
    private fun orderCardsForProcessing(
        state: GameState,
        cards: List<Card>,
        knowledge: DeckKnowledge,
    ): List<Card> {
        val weapons = cards.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
        val potions = cards.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }

        val result = mutableListOf<Card>()

        val bestNewWeapon = weapons.firstOrNull { shouldEquipWeaponWithKnowledge(state.weaponState, it, knowledge) }

        val effectiveWeaponValue =
            if (bestNewWeapon != null) {
                bestNewWeapon.value
            } else {
                state.weaponState?.weapon?.value ?: 0
            }

        // 1. Equip best weapon first
        if (bestNewWeapon != null) {
            result.add(bestNewWeapon)
        }

        // 2. Add remaining weapons
        result.addAll(weapons.filter { it !in result })

        // 3. Decide potion timing
        val estimatedDamage = monsters.sumOf { estimateDamage(it, effectiveWeaponValue, knowledge) }
        val needsHealingFirst = state.health <= estimatedDamage / 2

        if (needsHealingFirst && potions.isNotEmpty()) {
            result.add(potions.first())
        }

        // 4. Fight monsters - big ones first
        result.addAll(monsters)

        // 5. Add remaining potions
        result.addAll(potions.filter { it !in result })

        return result
    }

    private fun estimateDamage(
        monster: Card,
        weaponValue: Int,
        knowledge: DeckKnowledge,
    ): Int {
        if (weaponValue <= 0) return monster.value

        // Use dynamic threshold
        val threshold = getWeaponPreservationThreshold(knowledge)
        return if (monster.value >= threshold) {
            (monster.value - weaponValue).coerceAtLeast(0)
        } else {
            monster.value // Fighting barehanded to preserve
        }
    }

    /**
     * Processes a single card with knowledge-informed decisions.
     */
    private fun processCard(
        state: GameState,
        card: Card,
        knowledge: DeckKnowledge,
    ): GameState =
        when (card.type) {
            CardType.MONSTER -> processCombat(state, card, knowledge)
            CardType.WEAPON -> {
                if (shouldEquipWeaponWithKnowledge(state.weaponState, card, knowledge)) {
                    state.equipWeapon(card)
                } else {
                    state
                }
            }
            CardType.POTION -> state.usePotion(card)
        }

    /**
     * Makes combat decisions using deck knowledge.
     */
    private fun processCombat(
        state: GameState,
        monster: Card,
        knowledge: DeckKnowledge,
    ): GameState {
        val weapon = state.weaponState

        if (weapon == null || !weapon.canDefeat(monster)) {
            return state.fightMonsterBarehanded(monster)
        }

        val weaponIsDegraded = weapon.maxMonsterValue != null
        if (weaponIsDegraded) {
            return state.fightMonsterWithWeapon(monster)
        }

        // Fresh weapon - use dynamic threshold
        val threshold = getWeaponPreservationThreshold(knowledge)
        if (monster.value >= threshold) {
            return state.fightMonsterWithWeapon(monster)
        }

        // One safe improvement: use weapon to avoid death
        if (monster.value >= state.health) {
            return state.fightMonsterWithWeapon(monster)
        }

        return state.fightMonsterBarehanded(monster)
    }

    /**
     * Processes end game cards.
     */
    private fun processEndGame(
        state: GameState,
        room: List<Card>,
        knowledge: DeckKnowledge,
    ): Pair<GameState, DeckKnowledge> {
        val orderedCards = orderCardsForProcessing(state, room, knowledge)

        var currentState = state.copy(currentRoom = null)
        var currentKnowledge = knowledge

        for (card in orderedCards) {
            currentState = processCard(currentState, card, currentKnowledge)
            currentKnowledge = currentKnowledge.cardProcessed(card)
            if (currentState.isGameOver) {
                return Pair(currentState, currentKnowledge)
            }
        }

        return Pair(currentState, currentKnowledge)
    }

    private fun isActuallyWon(state: GameState): Boolean =
        state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0
}

/**
 * Simulator using the informed player.
 */
class InformedSimulator {
    private val player = InformedPlayer()

    fun simulateSeeds(seedRange: LongRange): Map<Long, SimulationResult> =
        seedRange.associateWith { seed ->
            val game = GameState.newGame(kotlin.random.Random(seed))
            simulateSingleSeed(game)
        }

    private fun simulateSingleSeed(initialState: GameState): SimulationResult {
        val finalState = player.playGame(initialState)
        val score = finalState.calculateScore()
        val won =
            finalState.health > 0 &&
                finalState.deck.isEmpty &&
                (finalState.currentRoom == null || finalState.currentRoom.isEmpty())

        return SimulationResult(
            samples = 1,
            wins = if (won) 1 else 0,
            losses = if (won) 0 else 1,
            winProbability = if (won) 1.0 else 0.0,
            averageWinScore = if (won) score.toDouble() else null,
            averageLossScore = if (won) null else score.toDouble(),
            maxScore = score,
            minScore = score,
        )
    }
}
