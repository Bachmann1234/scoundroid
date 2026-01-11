package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState

/**
 * A heuristic-based player that makes intelligent decisions.
 *
 * Strategy:
 * 1. Equip weapons before fighting monsters
 * 2. Use weapon on big monsters, fight small ones barehanded to preserve weapon
 * 3. Use potions when health is low or to top off
 * 4. Leave the worst card for next room when possible
 * 5. Skip rooms that look dangerous without mitigation
 */
class HeuristicPlayer {

    /**
     * Plays a complete game using heuristic decisions.
     * Returns the final game state.
     */
    fun playGame(initialState: GameState): GameState {
        var state = initialState

        while (!state.isGameOver && !isActuallyWon(state)) {
            state = playOneStep(state)
        }

        return state
    }

    private fun playOneStep(state: GameState): GameState {
        // If no room, draw one
        if (state.currentRoom == null || state.currentRoom.isEmpty()) {
            return state.drawRoom()
        }

        val room = state.currentRoom

        // If room has < 4 cards and deck has cards, draw more
        if (room.size < GameState.ROOM_SIZE && !state.deck.isEmpty) {
            return state.drawRoom()
        }

        // If room has 4 cards, decide: avoid or process
        if (room.size == GameState.ROOM_SIZE) {
            // Decide whether to skip this room
            if (!state.lastRoomAvoided && shouldSkipRoom(state, room)) {
                return state.avoidRoom()
            }

            // Process the room
            return processRoom(state, room)
        }

        // End game: room has < 4 cards and deck is empty
        return processEndGame(state, room)
    }

    /**
     * Decides whether to skip the current room.
     *
     * Key insight: Skipping puts these 4 cards at the bottom of deck plus draws 4 new.
     * Skip when the room would deal fatal/near-fatal damage that we can't mitigate.
     */
    private fun shouldSkipRoom(state: GameState, room: List<Card>): Boolean {
        val monsters = room.filter { it.type == CardType.MONSTER }

        if (monsters.isEmpty()) return false // No reason to skip

        // Calculate estimated damage if we process this room
        val bestCardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != bestCardToLeave }
        val estimatedNetDamage = simulateNetDamage(state, cardsToProcess)

        // Would this room kill us or leave us near death?
        if (estimatedNetDamage >= state.health - 2) {
            return true // Skip to survive
        }

        // Would this room take more than half our health with no good reason?
        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful = state.weaponState?.let { ws ->
            // Weapon is useful if it can hit at least some monsters
            monsters.any { ws.canDefeat(it) }
        } ?: false

        if (!currentWeaponUseful && !hasWeaponInRoom) {
            // No weapon to help - check if damage is too high
            if (estimatedNetDamage > state.health / 2) {
                return true
            }
        }

        return false
    }

    /**
     * Processes a room of 4 cards with smart decisions.
     */
    private fun processRoom(state: GameState, room: List<Card>): GameState {
        // Decide which card to leave (leave the worst one)
        val cardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != cardToLeave }

        // Order the cards smartly
        val orderedCards = orderCardsForProcessing(state, cardsToProcess)

        // Process each card
        var currentState = state.copy(
            currentRoom = listOf(cardToLeave),
            usedPotionThisTurn = false
        )

        for (card in orderedCards) {
            currentState = processCard(currentState, card)
            if (currentState.isGameOver) {
                return currentState
            }
        }

        return currentState
    }

    /**
     * Chooses which card to leave for the next room.
     *
     * Evaluates each possible choice considering both immediate and future impact.
     */
    private fun chooseCardToLeave(state: GameState, room: List<Card>): Card {
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

    /**
     * Simulates processing a set of cards and returns net damage taken.
     * This considers weapon pickup and degradation.
     *
     * Key insight: Don't count healing that would be wasted (health already at max).
     * Also, prefer processing big monsters to degrade weapon less (better for future rooms).
     */
    private fun simulateNetDamage(state: GameState, cards: List<Card>): Int {
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }
        val weapons = cards.filter { it.type == CardType.WEAPON }
        val potions = cards.filter { it.type == CardType.POTION }

        // Determine best weapon situation after processing
        val currentWeapon = state.weaponState
        val bestNewWeapon = weapons.maxByOrNull { it.value }

        // Decide which weapon we'd actually use
        val effectiveWeaponValue: Int
        val weaponIsFresh: Boolean

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

        // Calculate damage from monsters (fight big ones first while weapon is fresh)
        var totalDamage = 0
        var weaponMaxMonster: Int? = if (weaponIsFresh) null else currentWeapon?.maxMonsterValue

        for (monster in monsters) {
            val canUseWeapon = effectiveWeaponValue > 0 &&
                (weaponMaxMonster == null || monster.value <= weaponMaxMonster)

            if (canUseWeapon) {
                totalDamage += maxOf(0, monster.value - effectiveWeaponValue)
                weaponMaxMonster = monster.value // Weapon degrades
            } else {
                totalDamage += monster.value // Barehanded
            }
        }

        // Calculate EFFECTIVE healing (capped by health deficit)
        val healthDeficit = GameState.MAX_HEALTH - state.health + totalDamage
        val totalPotionValue = potions.sumOf { it.value }
        val effectiveHealing = minOf(totalPotionValue, healthDeficit.coerceAtLeast(0))

        return totalDamage - effectiveHealing
    }

    /**
     * Evaluates which card to leave, considering future impact.
     * Returns a score where LOWER is BETTER.
     */
    private fun evaluateLeaveChoice(state: GameState, cardToLeave: Card, cardsToProcess: List<Card>): Int {
        val netDamage = simulateNetDamage(state, cardsToProcess)

        // Add penalty for leaving big monsters - they'll be harder to fight later
        // when our weapon has degraded from fighting this room's monsters
        val leftoverPenalty = when (cardToLeave.type) {
            CardType.MONSTER -> cardToLeave.value // Big monsters left = big penalty
            CardType.POTION -> 0 // Potions are fine to leave
            CardType.WEAPON -> {
                // Leaving a weapon is bad if we need it
                val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                if (cardToLeave.value > currentWeaponValue) cardToLeave.value else 0
            }
        }

        return netDamage + leftoverPenalty
    }

    /**
     * Determines if we should equip a new weapon over the current one.
     * Considers both raw value AND degradation state.
     */
    private fun shouldEquipWeapon(current: dev.mattbachmann.scoundroid.data.model.WeaponState?, newWeapon: Card): Boolean {
        if (current == null) return true

        val currentValue = current.weapon.value
        val currentMaxMonster = current.maxMonsterValue

        // New weapon has higher value - definitely equip
        if (newWeapon.value > currentValue) return true

        // Current weapon is degraded below new weapon's value - equip the fresh one
        if (currentMaxMonster != null && currentMaxMonster < newWeapon.value) return true

        return false
    }

    /**
     * Assigns a value to a card for "keep vs leave" decisions.
     * Higher = more valuable to process now.
     */
    private fun cardValue(card: Card, state: GameState): Int {
        return when (card.type) {
            CardType.WEAPON -> {
                // Weapons are valuable if we don't have one or this is better
                val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                if (card.value > currentWeaponValue) card.value + 10 else card.value
            }
            CardType.POTION -> {
                // Potions are more valuable when health is low
                val healthDeficit = GameState.MAX_HEALTH - state.health
                if (healthDeficit >= card.value) card.value + 5 else card.value
            }
            CardType.MONSTER -> {
                // Monsters are negative value (damage), but small ones are less bad
                -card.value
            }
        }
    }

    /**
     * Orders cards for optimal processing.
     * Order: Weapons first (if upgrade), then BIG monsters, then potions
     */
    private fun orderCardsForProcessing(state: GameState, cards: List<Card>): List<Card> {
        val weapons = cards.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
        val potions = cards.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value } // BIG first!

        val result = mutableListOf<Card>()

        // Find best weapon to equip (considering degradation)
        val bestNewWeapon = weapons.firstOrNull { shouldEquipWeapon(state.weaponState, it) }

        // Calculate effective weapon after potential equip
        val effectiveWeaponValue = if (bestNewWeapon != null) {
            bestNewWeapon.value
        } else {
            state.weaponState?.weapon?.value ?: 0
        }

        // 1. Equip best weapon first (if it's an upgrade)
        if (bestNewWeapon != null) {
            result.add(bestNewWeapon)
        }

        // 2. Add remaining weapons (they'll be skipped if worse)
        result.addAll(weapons.filter { it !in result })

        // 3. Decide potion timing based on health
        val estimatedDamage = monsters.sumOf { estimateDamage(it, effectiveWeaponValue) }
        val needsHealingFirst = state.health <= estimatedDamage / 2

        if (needsHealingFirst && potions.isNotEmpty()) {
            result.add(potions.first())
        }

        // 4. Fight monsters - BIG ONES FIRST (use weapon while fresh, then it can still hit smaller ones)
        result.addAll(monsters)

        // 5. Add remaining potions (heal after combat)
        result.addAll(potions.filter { it !in result })

        return result
    }

    private fun estimateDamage(monster: Card, weaponValue: Int): Int {
        return if (weaponValue > 0) {
            (monster.value - weaponValue).coerceAtLeast(0)
        } else {
            monster.value
        }
    }

    /**
     * Processes a single card with smart decisions.
     */
    private fun processCard(state: GameState, card: Card): GameState {
        return when (card.type) {
            CardType.MONSTER -> processCombat(state, card)
            CardType.WEAPON -> {
                // Equip if it's an upgrade (considering degradation)
                if (shouldEquipWeapon(state.weaponState, card)) {
                    state.equipWeapon(card)
                } else {
                    // Discard the worse weapon (just don't equip it)
                    state
                }
            }
            CardType.POTION -> state.usePotion(card)
        }
    }

    /**
     * Makes intelligent combat decisions.
     *
     * Simple rule: Use weapon whenever it saves ANY damage.
     * The degradation mechanic means using it on big monsters first is key,
     * and once degraded, we should still use it on anything it can hit.
     */
    private fun processCombat(state: GameState, monster: Card): GameState {
        val weapon = state.weaponState

        // No weapon or weapon can't hit this monster = barehanded
        if (weapon == null || !weapon.canDefeat(monster)) {
            return state.fightMonsterBarehanded(monster)
        }

        // Weapon can hit - always use it, every point of damage saved matters!
        return state.fightMonsterWithWeapon(monster)
    }

    /**
     * Processes end game cards.
     */
    private fun processEndGame(state: GameState, room: List<Card>): GameState {
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

    private fun isActuallyWon(state: GameState): Boolean {
        return state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0
    }
}

/**
 * Monte Carlo simulation using the heuristic player.
 */
class HeuristicSimulator {

    private val player = HeuristicPlayer()

    /**
     * Simulates games for multiple seeds using heuristic play.
     */
    fun simulateSeeds(
        seedRange: LongRange,
    ): Map<Long, SimulationResult> {
        return seedRange.associateWith { seed ->
            val game = GameState.newGame(kotlin.random.Random(seed))
            simulateSingleSeed(game)
        }
    }

    /**
     * Since heuristic play is deterministic for a given seed,
     * we only need to play once per seed.
     */
    private fun simulateSingleSeed(initialState: GameState): SimulationResult {
        val finalState = player.playGame(initialState)
        val score = finalState.calculateScore()
        val won = finalState.health > 0 && finalState.deck.isEmpty &&
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
