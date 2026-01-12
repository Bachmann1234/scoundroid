package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import org.junit.Test
import kotlin.random.Random

class SeedAnalysisTest {
    @Test
    fun `analyze seed 1768180694612`() {
        val seed = 1768180694612L
        val game = GameState.newGame(Random(seed))

        println("=== Analyzing Seed $seed ===")
        println()

        // Show all cards in deck order
        val deckCards = game.deck.cards.map { "${it.rank.displayName}${it.suit.symbol}" }
        println("Deck (top to bottom): $deckCards")
        println()

        // Try with HeuristicPlayer
        val player = HeuristicPlayer()
        val finalState = player.playGame(game)

        println("\n=== HeuristicPlayer Result ===")
        println("Final Health: ${finalState.health}")
        println("Score: ${finalState.calculateScore()}")
        println("Won: ${finalState.isGameWon}")
        println("Game Over: ${finalState.isGameOver}")
        println("Deck remaining: ${finalState.deck.cards.size}")
        println("Room remaining: ${finalState.currentRoom?.size ?: 0}")

        // Try with InformedPlayer
        val informedPlayer = InformedPlayer()
        val informedFinal = informedPlayer.playGame(GameState.newGame(Random(seed)))

        println("\n=== InformedPlayer Result ===")
        println("Final Health: ${informedFinal.health}")
        println("Score: ${informedFinal.calculateScore()}")
        println("Won: ${informedFinal.isGameWon}")
        println("Game Over: ${informedFinal.isGameOver}")
        println("Deck remaining: ${informedFinal.deck.cards.size}")
        println("Room remaining: ${informedFinal.currentRoom?.size ?: 0}")

        // Also try with ParameterizedPlayer using best evolved genome
        val bestGenome =
            PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                skipDamageHealthFraction = 1.0, // 1.0 = disabled
                monsterLeavePenaltyMultiplier = 2.5,
                weaponLeavePenaltyIfNeeded = 9.0,
                potionLeavePenaltyPerRemaining = 0.0,
                weaponPreservationThreshold = 9,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
                alwaysSwapToFreshIfDegradedBelow = 8,
            )

        val paramPlayer = ParameterizedPlayer(bestGenome)
        val paramFinal = paramPlayer.playGame(GameState.newGame(Random(seed)))

        println("\n=== ParameterizedPlayer (Best GA) Result ===")
        println("Final Health: ${paramFinal.health}")
        println("Score: ${paramFinal.calculateScore()}")
        println("Won: ${paramFinal.isGameWon}")
    }

    @Test
    fun `detailed trace of seed 1768180694612`() {
        val seed = 1768180694612L
        println("=== DETAILED TRACE: Seed $seed ===\n")

        // Run with logging
        val game = GameState.newGame(Random(seed))
        val trace = TracingPlayer()
        trace.playGameWithTrace(game)
    }

    @Test
    fun `test improved weapon strategy on multiple seeds`() {
        // Test if preferring fresh weapons over degraded weapons helps
        val seeds = 0L..10000L
        var originalWins = 0
        var improvedWins = 0

        for (seed in seeds) {
            val game = GameState.newGame(Random(seed))

            // Original heuristic player
            val originalPlayer = HeuristicPlayer()
            val originalResult = originalPlayer.playGame(game)
            if (originalResult.isGameWon) originalWins++

            // Player that prefers fresh weapons
            val improvedPlayer = FreshWeaponPlayer()
            val improvedResult = improvedPlayer.playGame(GameState.newGame(Random(seed)))
            if (improvedResult.isGameWon) improvedWins++
        }

        println("=== Fresh Weapon Strategy Comparison (${seeds.count()} seeds) ===")
        println("Original HeuristicPlayer wins: $originalWins (${originalWins * 100.0 / seeds.count()}%)")
        println("FreshWeaponPlayer wins: $improvedWins (${improvedWins * 100.0 / seeds.count()}%)")
    }

    @Test
    fun `test FreshWeaponPlayer on seed 1768180694612`() {
        val seed = 1768180694612L
        val game = GameState.newGame(Random(seed))
        val player = FreshWeaponPlayer()
        val result = player.playGame(game)

        println("=== FreshWeaponPlayer on seed $seed ===")
        println("Health: ${result.health}")
        println("Score: ${result.calculateScore()}")
        println("Won: ${result.isGameWon}")
        println("Deck remaining: ${result.deck.cards.size}")

        // Also try with new optimal skipDamageHealthFraction
        println("\n=== ParameterizedPlayer (0.4 skip threshold) on seed $seed ===")
        val optimizedGenome =
            PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                skipDamageHealthFraction = 0.4,
                monsterLeavePenaltyMultiplier = 0.894,
                weaponLeavePenaltyIfNeeded = 2.505,
                potionLeavePenaltyPerRemaining = 0.0,
                weaponPreservationThreshold = 9,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
                alwaysSwapToFreshIfDegradedBelow = 8,
            )
        val paramPlayer = ParameterizedPlayer(optimizedGenome)
        val paramResult = paramPlayer.playGame(GameState.newGame(Random(seed)))
        println("Health: ${paramResult.health}")
        println("Score: ${paramResult.calculateScore()}")
        println("Won: ${paramResult.isGameWon}")
        println("Deck remaining: ${paramResult.deck.cards.size}")

        // Try with ALL optimizations (skip + potion penalty)
        println("\n=== ParameterizedPlayer (ALL optimizations) on seed $seed ===")
        val fullyOptimizedGenome =
            PlayerGenome(
                skipIfDamageExceedsHealthMinus = 5,
                skipWithoutWeaponDamageFraction = 0.444,
                skipDamageHealthFraction = 0.4,
                monsterLeavePenaltyMultiplier = 0.894,
                weaponLeavePenaltyIfNeeded = 2.505,
                potionLeavePenaltyPerRemaining = 0.5,
                weaponPreservationThreshold = 9,
                minDamageSavedToUseWeapon = 0,
                emergencyHealthBuffer = 0,
                equipFreshWeaponIfDegradedBelow = 10,
                alwaysSwapToFreshIfDegradedBelow = 8,
            )
        val fullPlayer = ParameterizedPlayer(fullyOptimizedGenome)
        val fullResult = fullPlayer.playGame(GameState.newGame(Random(seed)))
        println("Health: ${fullResult.health}")
        println("Score: ${fullResult.calculateScore()}")
        println("Won: ${fullResult.isGameWon}")
        println("Deck remaining: ${fullResult.deck.cards.size}")
    }

    @Test
    fun `analyze what makes winnable seeds different`() {
        // Find seeds where heuristic wins and analyze them
        val seeds = 0L..50000L
        val winnableSeeds = mutableListOf<Long>()

        for (seed in seeds) {
            val game = GameState.newGame(Random(seed))
            val player = HeuristicPlayer()
            val result = player.playGame(game)
            if (result.isGameWon) winnableSeeds.add(seed)
        }

        println("=== Analysis of ${winnableSeeds.size} Winnable Seeds ===")
        println("Winnable seeds: $winnableSeeds")

        // Analyze the first few winnable seeds
        for (seed in winnableSeeds.take(5)) {
            val game = GameState.newGame(Random(seed))
            val deckCards =
                game.deck.cards
                    .take(12)
                    .map { "${it.rank.displayName}${it.suit.symbol}" }
            println("\nSeed $seed first 12 cards: $deckCards")
        }
    }

    @Test
    fun `test potion leave penalty`() {
        // Test different penalties for leaving potions behind
        val testSeeds = 0L..10000L
        val penalties = listOf(0.0, 0.5, 1.0, 1.5, 2.0)

        println("=== Potion Leave Penalty Comparison ===")
        println("(penalty = potionsInDeck * penaltyPerRemaining)")

        for (penalty in penalties) {
            var wins = 0
            var totalScore = 0L

            val genome =
                PlayerGenome(
                    skipIfDamageExceedsHealthMinus = 5,
                    skipWithoutWeaponDamageFraction = 0.444,
                    skipDamageHealthFraction = 0.4,
                    monsterLeavePenaltyMultiplier = 0.894,
                    weaponLeavePenaltyIfNeeded = 2.505,
                    potionLeavePenaltyPerRemaining = penalty,
                    weaponPreservationThreshold = 9,
                    minDamageSavedToUseWeapon = 0,
                    emergencyHealthBuffer = 0,
                    equipFreshWeaponIfDegradedBelow = 10,
                    alwaysSwapToFreshIfDegradedBelow = 8,
                )
            val player = ParameterizedPlayer(genome)

            for (seed in testSeeds) {
                val game = GameState.newGame(Random(seed))
                val result = player.playGame(game)
                if (result.isGameWon) wins++
                totalScore += result.calculateScore()
            }

            val winRate = wins * 100.0 / testSeeds.count()
            val avgScore = totalScore.toDouble() / testSeeds.count()
            println(
                "Penalty $penalty: $wins wins (${
                    String.format(
                        "%.2f",
                        winRate,
                    )
                }%), avg score ${String.format("%.1f", avgScore)}",
            )
        }
    }

    @Test
    fun `test heal first threshold`() {
        // Current AI logic: heal first if health <= estimatedDamage / 2
        // Test different thresholds
        val testSeeds = 0L..10000L

        println("=== Heal-First Threshold Comparison ===")
        println("(heal first if health <= estimatedDamage * threshold)")

        // We'll test by modifying the base genome and comparing
        val thresholds = listOf(0.5, 0.75, 1.0, 1.25, 1.5)

        for (threshold in thresholds) {
            var wins = 0
            var totalScore = 0L

            for (seed in testSeeds) {
                val game = GameState.newGame(Random(seed))
                val player = HealThresholdPlayer(threshold)
                val result = player.playGame(game)
                if (result.isGameWon) wins++
                totalScore += result.calculateScore()
            }

            val winRate = wins * 100.0 / testSeeds.count()
            val avgScore = totalScore.toDouble() / testSeeds.count()
            println(
                "Threshold $threshold: $wins wins (${
                    String.format(
                        "%.2f",
                        winRate,
                    )
                }%), avg score ${String.format("%.1f", avgScore)}",
            )
        }
    }

    @Test
    fun `test aggressive skip threshold`() {
        // Test different skip thresholds to see which works best
        val testSeeds = 0L..10000L
        val thresholds = listOf(1.0, 0.6, 0.5, 0.4, 0.3)

        println("=== Skip Threshold Comparison ===")

        for (threshold in thresholds) {
            var wins = 0
            var totalScore = 0L

            val genome =
                PlayerGenome(
                    skipIfDamageExceedsHealthMinus = 5,
                    skipWithoutWeaponDamageFraction = 0.444,
                    skipDamageHealthFraction = threshold,
                    monsterLeavePenaltyMultiplier = 0.894,
                    weaponLeavePenaltyIfNeeded = 2.505,
                    potionLeavePenaltyPerRemaining = 0.0,
                    weaponPreservationThreshold = 9,
                    minDamageSavedToUseWeapon = 0,
                    emergencyHealthBuffer = 0,
                    equipFreshWeaponIfDegradedBelow = 10,
                    alwaysSwapToFreshIfDegradedBelow = 8,
                )
            val player = ParameterizedPlayer(genome)

            for (seed in testSeeds) {
                val game = GameState.newGame(Random(seed))
                val result = player.playGame(game)
                if (result.isGameWon) wins++
                totalScore += result.calculateScore()
            }

            val winRate = wins * 100.0 / testSeeds.count()
            val avgScore = totalScore.toDouble() / testSeeds.count()
            println(
                "Threshold $threshold: $wins wins (${String.format(
                    "%.2f",
                    winRate,
                )}%), avg score ${String.format("%.1f", avgScore)}",
            )
        }
    }
}

/**
 * A player that more aggressively values fresh weapons over degraded ones.
 * Key insight: A fresh 5♦ can hit ANY monster, while a degraded 9♦ (max=8) can only hit ≤8.
 */
class FreshWeaponPlayer {
    companion object {
        const val WEAPON_PRESERVATION_THRESHOLD = 9
        const val SKIP_DAMAGE_HEALTH_BUFFER = 5
        const val SKIP_WITHOUT_WEAPON_FRACTION = 0.444
    }

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

    /**
     * IMPROVED: Consider skipping when weapon is degraded and room has big monsters
     */
    private fun shouldSkipRoom(
        state: GameState,
        room: List<Card>,
    ): Boolean {
        val monsters = room.filter { it.type == CardType.MONSTER }
        if (monsters.isEmpty()) return false

        val bestCardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != bestCardToLeave }
        val estimatedNetDamage = simulateNetDamage(state, cardsToProcess)

        // Standard skip logic
        if (estimatedNetDamage >= state.health - SKIP_DAMAGE_HEALTH_BUFFER) {
            return true
        }

        // NEW: Skip if we have a degraded weapon and face monsters we can't hit
        val weapon = state.weaponState
        if (weapon != null && weapon.maxMonsterValue != null) {
            val bigMonstersWeCannotHit = monsters.filter { it.value > weapon.maxMonsterValue!! }
            val barehandedDamage = bigMonstersWeCannotHit.sumOf { it.value }
            if (barehandedDamage >= state.health - 2) {
                return true // Skip to hopefully find a fresh weapon
            }
        }

        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful =
            weapon?.let { ws ->
                monsters.any { ws.canDefeat(it) }
            } ?: false

        if (!currentWeaponUseful && !hasWeaponInRoom) {
            if (estimatedNetDamage > state.health * SKIP_WITHOUT_WEAPON_FRACTION) {
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
            if (currentState.isGameOver) return currentState
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

    private fun evaluateLeaveChoice(
        state: GameState,
        cardToLeave: Card,
        cardsToProcess: List<Card>,
    ): Int {
        val netDamage = simulateNetDamage(state, cardsToProcess)

        // IMPROVED: Heavier penalty for leaving big monsters if our weapon is degraded
        val leftoverPenalty =
            when (cardToLeave.type) {
                CardType.MONSTER -> {
                    val weapon = state.weaponState
                    val baseValue = cardToLeave.value
                    // If weapon can't hit this monster, leaving it is less bad (we'd take full damage anyway)
                    if (weapon != null &&
                        weapon.maxMonsterValue != null &&
                        cardToLeave.value > weapon.maxMonsterValue!!
                    ) {
                        baseValue / 2 // Reduced penalty for leaving monsters we can't hit
                    } else {
                        baseValue
                    }
                }
                CardType.POTION -> 0
                CardType.WEAPON -> {
                    val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                    if (cardToLeave.value > currentWeaponValue) cardToLeave.value else 0
                }
            }

        return netDamage + leftoverPenalty
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
        var weaponMaxMonster: Int? = if (weaponIsFresh) null else currentWeapon?.maxMonsterValue

        for (monster in monsters) {
            val canUseWeapon =
                effectiveWeaponValue > 0 &&
                    (weaponMaxMonster == null || monster.value <= weaponMaxMonster)

            if (canUseWeapon) {
                val shouldUseWeapon = !weaponIsFresh || monster.value >= WEAPON_PRESERVATION_THRESHOLD

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
     * IMPROVED: Prefer fresh weapons over degraded ones, even if value is lower.
     * A fresh 5♦ can block 5 from ANY monster.
     * A degraded 9♦ (max=7) can only block from monsters ≤7.
     */
    private fun shouldEquipWeapon(
        current: dev.mattbachmann.scoundroid.data.model.WeaponState?,
        newWeapon: Card,
    ): Boolean {
        if (current == null) return true

        val currentValue = current.weapon.value
        val currentMaxMonster = current.maxMonsterValue

        // New weapon has higher value - definitely equip
        if (newWeapon.value > currentValue) return true

        // NEW: If current weapon is degraded, prefer fresh weapon if it can block meaningful damage
        if (currentMaxMonster != null) {
            // Fresh weapon can hit ANY monster
            // If current is degraded significantly, swap to fresh
            if (newWeapon.value >= currentMaxMonster - 2) {
                return true // Fresh weapon is nearly as good and has no restrictions
            }
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
        val effectiveWeaponValue = bestNewWeapon?.value ?: state.weaponState?.weapon?.value ?: 0

        if (bestNewWeapon != null) result.add(bestNewWeapon)
        result.addAll(weapons.filter { it !in result })

        val estimatedDamage =
            monsters.sumOf {
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

        val weaponIsDegraded = weapon.maxMonsterValue != null
        if (weaponIsDegraded) {
            return state.fightMonsterWithWeapon(monster)
        }

        if (monster.value >= WEAPON_PRESERVATION_THRESHOLD) {
            return state.fightMonsterWithWeapon(monster)
        }

        return state.fightMonsterBarehanded(monster)
    }

    private fun processEndGame(
        state: GameState,
        room: List<Card>,
    ): GameState {
        val orderedCards = orderCardsForProcessing(state, room)
        var currentState = state.copy(currentRoom = null)

        for (card in orderedCards) {
            currentState = processCard(currentState, card)
            if (currentState.isGameOver) return currentState
        }

        return currentState
    }

    private fun isActuallyWon(state: GameState): Boolean =
        state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0
}

/**
 * A player that traces every decision it makes for analysis.
 */
class TracingPlayer {
    private var roomNumber = 0

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
                "Health: ${state.health}, Weapon: ${state.weaponState?.let {
                    "${formatCard(
                        it.weapon,
                    )} (max: ${it.maxMonsterValue ?: "fresh"})"
                } ?: "none"}",
            )
            println("Cards: ${room.map { formatCard(it) }}")

            // Check skip decision
            val shouldSkip = !state.lastRoomAvoided && shouldSkipRoom(state, room)
            if (shouldSkip) {
                println("DECISION: SKIP ROOM (lastRoomAvoided=${state.lastRoomAvoided})")
                return state.avoidRoom()
            }

            // Process the room
            return processRoomWithTrace(state, room)
        }

        // End game
        roomNumber++
        println("--- Final Room $roomNumber (${room.size} cards) ---")
        println(
            "Health: ${state.health}, Weapon: ${state.weaponState?.let {
                "${formatCard(
                    it.weapon,
                )} (max: ${it.maxMonsterValue ?: "fresh"})"
            } ?: "none"}",
        )
        println("Cards: ${room.map { formatCard(it) }}")
        return processEndGameWithTrace(state, room)
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

        val threshold1 = state.health - HeuristicPlayer.SKIP_DAMAGE_HEALTH_BUFFER
        if (estimatedNetDamage >= threshold1) {
            println("  Skip check: estimated damage $estimatedNetDamage >= threshold $threshold1 → SKIP")
            return true
        }

        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful =
            state.weaponState?.let { ws ->
                monsters.any { ws.canDefeat(it) }
            } ?: false

        if (!currentWeaponUseful && !hasWeaponInRoom) {
            val threshold2 = (state.health * HeuristicPlayer.SKIP_WITHOUT_WEAPON_FRACTION).toInt()
            if (estimatedNetDamage > threshold2) {
                println("  Skip check: no weapon help, damage $estimatedNetDamage > $threshold2 → SKIP")
                return true
            }
        }

        println("  Skip check: estimated damage $estimatedNetDamage, health ${state.health} → PROCESS")
        return false
    }

    private fun chooseCardToLeave(
        state: GameState,
        room: List<Card>,
    ): Card {
        var bestCardToLeave = room.first()
        var bestScore = Int.MAX_VALUE

        for (candidate in room) {
            val cardsToProcess = room.filter { it != candidate }
            val netDamage = simulateNetDamage(state, cardsToProcess)
            val leftoverPenalty =
                when (candidate.type) {
                    CardType.MONSTER -> candidate.value
                    CardType.POTION -> 0
                    CardType.WEAPON -> {
                        val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                        if (candidate.value > currentWeaponValue) candidate.value else 0
                    }
                }
            val score = netDamage + leftoverPenalty

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
        var weaponMaxMonster: Int? = if (weaponIsFresh) null else currentWeapon?.maxMonsterValue

        for (monster in monsters) {
            val canUseWeapon =
                effectiveWeaponValue > 0 &&
                    (weaponMaxMonster == null || monster.value <= weaponMaxMonster)

            if (canUseWeapon) {
                val shouldUseWeapon = !weaponIsFresh || monster.value >= HeuristicPlayer.WEAPON_PRESERVATION_THRESHOLD

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

    private fun shouldEquipWeapon(
        current: dev.mattbachmann.scoundroid.data.model.WeaponState?,
        newWeapon: Card,
    ): Boolean {
        if (current == null) return true
        if (newWeapon.value > current.weapon.value) return true
        if (current.maxMonsterValue != null &&
            current.maxMonsterValue < HeuristicPlayer.EQUIP_FRESH_IF_DEGRADED_BELOW
        ) {
            return newWeapon.value >= current.maxMonsterValue
        }
        return false
    }

    private fun processRoomWithTrace(
        state: GameState,
        room: List<Card>,
    ): GameState {
        val cardToLeave = chooseCardToLeave(state, room)
        val cardsToProcess = room.filter { it != cardToLeave }

        println("DECISION: Leave ${formatCard(cardToLeave)}, process ${cardsToProcess.map { formatCard(it) }}")

        val orderedCards = orderCardsForProcessing(state, cardsToProcess)

        var currentState =
            state.copy(
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

    private fun orderCardsForProcessing(
        state: GameState,
        cards: List<Card>,
    ): List<Card> {
        val weapons = cards.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
        val potions = cards.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }

        val result = mutableListOf<Card>()
        val bestNewWeapon = weapons.firstOrNull { shouldEquipWeapon(state.weaponState, it) }
        val effectiveWeaponValue = bestNewWeapon?.value ?: state.weaponState?.weapon?.value ?: 0

        if (bestNewWeapon != null) result.add(bestNewWeapon)
        result.addAll(weapons.filter { it !in result })

        val estimatedDamage =
            monsters.sumOf {
                if (effectiveWeaponValue >
                    0
                ) {
                    maxOf(0, it.value - effectiveWeaponValue)
                } else {
                    it.value
                }
            }
        val needsHealingFirst = state.health <= estimatedDamage / 2

        if (needsHealingFirst && potions.isNotEmpty()) {
            result.add(potions.first())
        }

        result.addAll(monsters)
        result.addAll(potions.filter { it !in result })

        return result
    }

    private fun processCardWithTrace(
        state: GameState,
        card: Card,
    ): GameState =
        when (card.type) {
            CardType.MONSTER -> processCombatWithTrace(state, card)
            CardType.WEAPON -> {
                if (shouldEquipWeapon(state.weaponState, card)) {
                    println("  Equip ${formatCard(card)}")
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

    private fun processCombatWithTrace(
        state: GameState,
        monster: Card,
    ): GameState {
        val weapon = state.weaponState

        if (weapon == null || !weapon.canDefeat(monster)) {
            println("  Fight ${formatCard(monster)} BAREHANDED (no usable weapon)")
            return state.fightMonsterBarehanded(monster)
        }

        val weaponIsDegraded = weapon.maxMonsterValue != null
        if (weaponIsDegraded) {
            val damage = maxOf(0, monster.value - weapon.weapon.value)
            println(
                "  Fight ${formatCard(
                    monster,
                )} with ${formatCard(weapon.weapon)} (degraded, max=${weapon.maxMonsterValue}) → $damage damage",
            )
            return state.fightMonsterWithWeapon(monster)
        }

        if (monster.value >= HeuristicPlayer.WEAPON_PRESERVATION_THRESHOLD) {
            val damage = maxOf(0, monster.value - weapon.weapon.value)
            println(
                "  Fight ${formatCard(
                    monster,
                )} with ${formatCard(
                    weapon.weapon,
                )} (fresh, monster >= ${HeuristicPlayer.WEAPON_PRESERVATION_THRESHOLD}) → $damage damage",
            )
            return state.fightMonsterWithWeapon(monster)
        }

        println("  Fight ${formatCard(monster)} BAREHANDED (preserve fresh weapon for big monsters)")
        return state.fightMonsterBarehanded(monster)
    }

    private fun processEndGameWithTrace(
        state: GameState,
        room: List<Card>,
    ): GameState {
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

/**
 * A player with configurable heal-first threshold.
 * Heals BEFORE combat if health <= estimatedDamage * threshold
 */
class HealThresholdPlayer(
    private val healThreshold: Double,
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

        if (estimatedNetDamage >= state.health - 5) return true
        if (estimatedNetDamage > state.health * 0.4) return true

        val hasWeaponInRoom = room.any { it.type == CardType.WEAPON }
        val currentWeaponUseful =
            state.weaponState?.let { ws ->
                monsters.any { ws.canDefeat(it) }
            } ?: false

        if (!currentWeaponUseful && !hasWeaponInRoom) {
            if (estimatedNetDamage > state.health * 0.444) return true
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
            if (currentState.isGameOver) return currentState
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
            val netDamage = simulateNetDamage(state, cardsToProcess)
            val leftoverPenalty =
                when (candidate.type) {
                    CardType.MONSTER -> candidate.value
                    CardType.POTION -> 0
                    CardType.WEAPON -> {
                        val currentWeaponValue = state.weaponState?.weapon?.value ?: 0
                        if (candidate.value > currentWeaponValue) candidate.value else 0
                    }
                }
            val score = netDamage + leftoverPenalty
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
        var weaponMaxMonster: Int? = if (weaponIsFresh) null else currentWeapon?.maxMonsterValue

        for (monster in monsters) {
            val canUseWeapon =
                effectiveWeaponValue > 0 &&
                    (weaponMaxMonster == null || monster.value <= weaponMaxMonster)

            if (canUseWeapon) {
                val shouldUseWeapon = !weaponIsFresh || monster.value >= 9

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

    private fun shouldEquipWeapon(
        current: dev.mattbachmann.scoundroid.data.model.WeaponState?,
        newWeapon: Card,
    ): Boolean {
        if (current == null) return true
        if (newWeapon.value > current.weapon.value) return true
        if (current.maxMonsterValue != null && current.maxMonsterValue < 10) {
            return newWeapon.value >= current.maxMonsterValue
        }
        return false
    }

    /**
     * Key difference: configurable heal threshold
     */
    private fun orderCardsForProcessing(
        state: GameState,
        cards: List<Card>,
    ): List<Card> {
        val weapons = cards.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
        val potions = cards.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }

        val result = mutableListOf<Card>()
        val bestNewWeapon = weapons.firstOrNull { shouldEquipWeapon(state.weaponState, it) }
        val effectiveWeaponValue = bestNewWeapon?.value ?: state.weaponState?.weapon?.value ?: 0

        if (bestNewWeapon != null) result.add(bestNewWeapon)
        result.addAll(weapons.filter { it !in result })

        val estimatedDamage =
            monsters.sumOf {
                if (effectiveWeaponValue > 0) maxOf(0, it.value - effectiveWeaponValue) else it.value
            }

        // CONFIGURABLE: heal first if health <= estimatedDamage * threshold
        val needsHealingFirst = state.health <= estimatedDamage * healThreshold

        if (needsHealingFirst && potions.isNotEmpty()) {
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

        val weaponIsDegraded = weapon.maxMonsterValue != null
        if (weaponIsDegraded) {
            return state.fightMonsterWithWeapon(monster)
        }

        if (monster.value >= 9) {
            return state.fightMonsterWithWeapon(monster)
        }

        return state.fightMonsterBarehanded(monster)
    }

    private fun processEndGame(
        state: GameState,
        room: List<Card>,
    ): GameState {
        val orderedCards = orderCardsForProcessing(state, room)
        var currentState = state.copy(currentRoom = null)

        for (card in orderedCards) {
            currentState = processCard(currentState, card)
            if (currentState.isGameOver) return currentState
        }

        return currentState
    }

    private fun isActuallyWon(state: GameState): Boolean =
        state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0
}
