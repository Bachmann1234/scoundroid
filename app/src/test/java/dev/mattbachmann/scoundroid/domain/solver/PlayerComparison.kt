package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import kotlin.random.Random

/**
 * Utility to compare InformedPlayer vs HeuristicPlayer decisions.
 */
object PlayerComparison {
    fun compareSeed(seed: Long): String {
        val game = GameState.newGame(Random(seed))
        val sb = StringBuilder()

        sb.appendLine("=== SEED $seed ===")
        sb.appendLine("Initial deck order:")
        game.deck.cards.take(12).forEachIndexed { i, card ->
            sb.appendLine("  $i: ${card.displayName} (${card.type})")
        }
        sb.appendLine("  ...")

        // Run both players
        val heuristicPlayer = HeuristicPlayer()
        val informedPlayer = InformedPlayer()

        val heuristicResult = heuristicPlayer.playGame(game)
        val informedResult = informedPlayer.playGame(game)

        val heuristicWon = isWin(heuristicResult)
        val informedWon = isWin(informedResult)

        sb.appendLine("\n=== RESULTS ===")
        sb.appendLine("Heuristic: ${if (heuristicWon) "WIN" else "LOSS"} (health=${heuristicResult.health})")
        sb.appendLine("Informed:  ${if (informedWon) "WIN" else "LOSS"} (health=${informedResult.health})")

        return sb.toString()
    }

    fun traceInformedPlayer(seed: Long): String {
        val game = GameState.newGame(Random(seed))
        val sb = StringBuilder()

        sb.appendLine("=== INFORMED PLAYER TRACE: SEED $seed ===")

        var state = game
        var knowledge = DeckKnowledge.initial()
        var roomNum = 0

        while (!state.isGameOver && !isWin(state)) {
            if (state.currentRoom == null || state.currentRoom.isEmpty()) {
                state = state.drawRoom()
                continue
            }

            val room = state.currentRoom
            if (room.size < GameState.ROOM_SIZE && !state.deck.isEmpty) {
                state = state.drawRoom()
                continue
            }

            if (room.size == GameState.ROOM_SIZE) {
                roomNum++
                sb.appendLine("\n--- Room $roomNum ---")
                sb.appendLine("Health: ${state.health}, Deck: ${state.deck.size}")
                sb.appendLine(
                    "Weapon: ${state.weaponState?.weapon?.displayName ?: "none"} " +
                        "(degraded to: ${state.weaponState?.maxMonsterValue ?: "fresh"})",
                )
                sb.appendLine("Room: ${room.map { it.displayName }}")
                sb.appendLine("Max monster remaining: ${knowledge.maxMonsterRemaining}")
                sb.appendLine("Weapons remaining: ${knowledge.remainingWeapons.size}")

                // Check skip decision
                val wouldSkip = !state.lastRoomAvoided && wouldInformedSkip(state, room, knowledge)
                sb.appendLine("Would skip: $wouldSkip (lastSkipped=${state.lastRoomAvoided})")

                if (wouldSkip) {
                    knowledge = knowledge.roomSkipped(room)
                    state = state.avoidRoom()
                    sb.appendLine("  -> SKIPPED")
                    continue
                }

                // Process room
                val cardToLeave = findCardToLeave(state, room, knowledge)
                val cardsToProcess = room.filter { it != cardToLeave }
                sb.appendLine("Leave: ${cardToLeave.displayName}")
                sb.appendLine("Process: ${cardsToProcess.map { it.displayName }}")

                state = state.copy(currentRoom = listOf(cardToLeave), usedPotionThisTurn = false)

                for (card in orderCards(state, cardsToProcess, knowledge)) {
                    val oldHealth = state.health
                    val oldWeapon = state.weaponState

                    when (card.type) {
                        CardType.MONSTER -> {
                            val threshold = getThreshold(knowledge)
                            val weaponUsable = state.weaponState?.canDefeat(card) == true
                            val weaponFresh = state.weaponState?.maxMonsterValue == null

                            val useWeapon = weaponUsable && (!weaponFresh || card.value >= threshold)

                            if (useWeapon) {
                                state = state.fightMonsterWithWeapon(card)
                                sb.appendLine(
                                    "  ${card.displayName}: weapon (threshold=$threshold) -> health $oldHealth -> ${state.health}",
                                )
                            } else {
                                state = state.fightMonsterBarehanded(card)
                                sb.appendLine(
                                    "  ${card.displayName}: barehanded -> health $oldHealth -> ${state.health}",
                                )
                            }
                        }
                        CardType.WEAPON -> {
                            val shouldEquip = shouldEquipWithKnowledge(state.weaponState, card, knowledge)
                            if (shouldEquip) {
                                state = state.equipWeapon(card)
                                sb.appendLine(
                                    "  ${card.displayName}: equipped (was ${oldWeapon?.weapon?.displayName ?: "none"})",
                                )
                            } else {
                                sb.appendLine(
                                    "  ${card.displayName}: skipped (keeping ${oldWeapon?.weapon?.displayName})",
                                )
                            }
                        }
                        CardType.POTION -> {
                            state = state.usePotion(card)
                            sb.appendLine("  ${card.displayName}: heal -> health $oldHealth -> ${state.health}")
                        }
                    }
                    knowledge = knowledge.cardProcessed(card)

                    if (state.isGameOver) {
                        sb.appendLine("  DIED!")
                        break
                    }
                }
            } else if (room.size < 4 && state.deck.isEmpty) {
                // End game
                roomNum++
                sb.appendLine("\n--- End Game (Room $roomNum) ---")
                sb.appendLine("Health: ${state.health}")
                sb.appendLine("Room: ${room.map { it.displayName }}")

                state = state.copy(currentRoom = null)
                for (card in orderCards(state, room, knowledge)) {
                    val oldHealth = state.health
                    when (card.type) {
                        CardType.MONSTER -> {
                            val useWeapon = state.weaponState?.canDefeat(card) == true
                            state =
                                if (useWeapon) {
                                    state.fightMonsterWithWeapon(
                                        card,
                                    )
                                } else {
                                    state.fightMonsterBarehanded(card)
                                }
                            sb.appendLine(
                                "  ${card.displayName}: ${if (useWeapon) "weapon" else "bare"} -> health $oldHealth -> ${state.health}",
                            )
                        }
                        CardType.WEAPON -> {
                            if (shouldEquipWithKnowledge(state.weaponState, card, knowledge)) {
                                state = state.equipWeapon(card)
                                sb.appendLine("  ${card.displayName}: equipped")
                            }
                        }
                        CardType.POTION -> {
                            state = state.usePotion(card)
                            sb.appendLine("  ${card.displayName}: heal -> $oldHealth -> ${state.health}")
                        }
                    }
                    knowledge = knowledge.cardProcessed(card)
                    if (state.isGameOver) {
                        sb.appendLine("  DIED!")
                        break
                    }
                }
            }

            if (state.isGameOver) break
        }

        sb.appendLine("\n=== FINAL ===")
        sb.appendLine("Health: ${state.health}")
        sb.appendLine("Won: ${isWin(state)}")

        return sb.toString()
    }

    private fun isWin(state: GameState): Boolean =
        state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0

    private fun getThreshold(knowledge: DeckKnowledge): Int =
        minOf(InformedPlayer.BASE_WEAPON_PRESERVATION_THRESHOLD, knowledge.maxMonsterRemaining)

    private fun wouldInformedSkip(
        state: GameState,
        room: List<Card>,
        knowledge: DeckKnowledge,
    ): Boolean {
        // Simplified check - just see if it would skip
        val monsters = room.filter { it.type == CardType.MONSTER }
        if (monsters.isEmpty()) return false

        val monsterDamage = monsters.sumOf { it.value }
        val weaponHelp =
            if (state.weaponState != null) {
                monsters
                    .filter { state.weaponState.canDefeat(it) }
                    .sumOf { minOf(it.value, state.weaponState.weapon.value) }
            } else {
                0
            }
        val potionHelp = room.filter { it.type == CardType.POTION }.sumOf { it.value }
        val netDamage = monsterDamage - weaponHelp - potionHelp

        return netDamage >= state.health - InformedPlayer.SKIP_DAMAGE_HEALTH_BUFFER
    }

    private fun findCardToLeave(
        state: GameState,
        room: List<Card>,
        knowledge: DeckKnowledge,
    ): Card =
        room.minByOrNull {
            cardLeaveScore(it, state, knowledge)
        } ?: room.first()

    private fun cardLeaveScore(
        card: Card,
        state: GameState,
        knowledge: DeckKnowledge,
    ): Int =
        when (card.type) {
            CardType.MONSTER -> -card.value // Leave big monsters (negative = bad to leave)
            CardType.POTION -> if (knowledge.remainingPotions.size <= 3) card.value else 0
            CardType.WEAPON -> {
                val currentVal = state.weaponState?.weapon?.value ?: 0
                if (card.value > currentVal) card.value * 2 else 0
            }
        }

    private fun orderCards(
        state: GameState,
        cards: List<Card>,
        knowledge: DeckKnowledge,
    ): List<Card> {
        val weapons = cards.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
        val potions = cards.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
        val monsters = cards.filter { it.type == CardType.MONSTER }.sortedByDescending { it.value }
        return weapons + monsters + potions
    }

    private fun shouldEquipWithKnowledge(
        current: dev.mattbachmann.scoundroid.data.model.WeaponState?,
        new: Card,
        knowledge: DeckKnowledge,
    ): Boolean {
        if (current == null) return true
        if (new.value > current.weapon.value) return true
        val maxRemaining = knowledge.maxMonsterRemaining
        if (current.maxMonsterValue != null && current.maxMonsterValue >= maxRemaining) {
            return new.value > current.weapon.value
        }
        if (current.maxMonsterValue != null && current.maxMonsterValue < InformedPlayer.EQUIP_FRESH_IF_DEGRADED_BELOW) {
            return new.value >= current.maxMonsterValue
        }
        return false
    }
}
