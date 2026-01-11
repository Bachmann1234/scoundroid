package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState

/**
 * Solves a Scoundrel game seed by exploring all possible decision paths.
 *
 * For a given initial game state (determined by seed), this solver enumerates
 * every possible sequence of player decisions and tracks:
 * - Total number of distinct paths
 * - Number of winning vs losing paths
 * - Maximum achievable score
 * - Win probability (assuming uniform random decisions)
 */
class GameSolver {

    /**
     * Solves the game from the given state, exploring all possible paths.
     *
     * @param state The current game state to solve from
     * @return SolveResult containing statistics about all possible outcomes
     */
    fun solve(state: GameState): SolveResult {
        // Base case: game over (loss)
        if (state.isGameOver) {
            return SolveResult.loss(state.calculateScore())
        }

        // Base case: game won (deck empty and room empty/null)
        if (state.deck.isEmpty && (state.currentRoom == null || state.currentRoom.isEmpty())) {
            return SolveResult.win(state.calculateScore())
        }

        // If no room or room needs more cards, try to draw
        if (state.currentRoom == null || state.currentRoom.isEmpty()) {
            return solve(state.drawRoom())
        }

        val room = state.currentRoom

        // If room has < 4 cards, we need to fill it (if deck has cards)
        if (room.size < GameState.ROOM_SIZE && !state.deck.isEmpty) {
            return solve(state.drawRoom())
        }

        // If room has 4 cards, normal room processing
        if (room.size == GameState.ROOM_SIZE) {
            return solveRoom(state)
        }

        // Room has < 4 cards and deck is empty - process all remaining cards
        // This is the end game scenario
        return solveEndGame(state, room)
    }

    /**
     * Solves the end game when deck is empty and we have < 4 cards left.
     * We must process all remaining cards (no choice to leave one).
     */
    private fun solveEndGame(state: GameState, remainingCards: List<Card>): SolveResult {
        if (remainingCards.isEmpty()) {
            return SolveResult.win(state.calculateScore())
        }

        // Try all orderings of remaining cards
        val results = mutableListOf<SolveResult>()
        for (ordering in remainingCards.permutations()) {
            val result = solveEndGameSequence(state.copy(currentRoom = null), ordering)
            results.add(result)
        }
        return SolveResult.combine(results)
    }

    /**
     * Processes a sequence of cards in end game (no leftover).
     */
    private fun solveEndGameSequence(state: GameState, cards: List<Card>): SolveResult {
        var currentState = state

        for (card in cards) {
            // Check for combat choice
            if (card.type == CardType.MONSTER &&
                currentState.weaponState != null &&
                currentState.weaponState.canDefeat(card)
            ) {
                // Branch on combat choice, then continue with remaining cards
                val cardIndex = cards.indexOf(card)
                val remainingCards = if (cardIndex < cards.lastIndex) {
                    cards.subList(cardIndex + 1, cards.size)
                } else {
                    emptyList()
                }

                val results = mutableListOf<SolveResult>()
                for (useWeapon in listOf(true, false)) {
                    val afterCombat = processCard(currentState, card, useWeapon)
                    if (afterCombat.isGameOver) {
                        results.add(SolveResult.loss(afterCombat.calculateScore()))
                    } else if (remainingCards.isEmpty()) {
                        results.add(SolveResult.win(afterCombat.calculateScore()))
                    } else {
                        results.add(solveEndGameSequence(afterCombat, remainingCards))
                    }
                }
                return SolveResult.combine(results)
            }

            // No combat choice, just process
            currentState = processCard(currentState, card, useWeapon = false)

            if (currentState.isGameOver) {
                return SolveResult.loss(currentState.calculateScore())
            }
        }

        // All cards processed, game won
        return SolveResult.win(currentState.calculateScore())
    }

    /**
     * Solves a room with 4 cards, exploring all decision branches.
     */
    private fun solveRoom(state: GameState): SolveResult {
        val room = state.currentRoom ?: return solve(state.drawRoom())
        require(room.size == GameState.ROOM_SIZE) { "Room must have exactly 4 cards" }

        val results = mutableListOf<SolveResult>()

        // Option 1: Avoid room (if allowed)
        if (!state.lastRoomAvoided) {
            val avoidedState = state.avoidRoom()
            results.add(solve(avoidedState))
        }

        // Option 2: Process room - choose which card to leave and order to process
        for (leaveIndex in room.indices) {
            val cardToLeave = room[leaveIndex]
            val cardsToProcess = room.filterIndexed { i, _ -> i != leaveIndex }

            // Enumerate all orderings of the 3 cards to process
            for (ordering in cardsToProcess.permutations()) {
                val result = solveProcessingSequence(state, ordering, cardToLeave)
                results.add(result)
            }
        }

        return SolveResult.combine(results)
    }

    /**
     * Solves processing a sequence of cards in order, leaving one card for next room.
     */
    private fun solveProcessingSequence(
        state: GameState,
        cardsToProcess: List<Card>,
        cardToLeave: Card
    ): SolveResult {
        // After processing, room will have just the leftover card
        var currentState = state.copy(
            currentRoom = listOf(cardToLeave),
            usedPotionThisTurn = false // Reset for new turn
        )

        for ((index, card) in cardsToProcess.withIndex()) {
            // Check for combat choice
            if (card.type == CardType.MONSTER &&
                currentState.weaponState != null &&
                currentState.weaponState.canDefeat(card)
            ) {
                // Branch on combat choice
                return solveWithCombatChoice(
                    currentState,
                    cardsToProcess,
                    index,
                    cardToLeave
                )
            }

            // No combat choice, just process the card
            currentState = processCard(currentState, card, useWeapon = false)

            // Check for game over
            if (currentState.isGameOver) {
                return SolveResult.loss(currentState.calculateScore())
            }
        }

        // All 3 cards processed, continue to next room
        // Room now has the leftover card, so solve will draw to fill it
        return solve(currentState)
    }

    /**
     * Handles a processing sequence where a combat choice is involved.
     */
    private fun solveWithCombatChoice(
        state: GameState,
        cardsToProcess: List<Card>,
        combatIndex: Int,
        cardToLeave: Card
    ): SolveResult {
        val results = mutableListOf<SolveResult>()

        // Try both: use weapon and go barehanded
        for (useWeapon in listOf(true, false)) {
            var currentState = state.copy(currentRoom = listOf(cardToLeave))

            // Process cards up to and including the combat choice
            for (i in 0..combatIndex) {
                val card = cardsToProcess[i]
                val shouldUseWeapon = if (i == combatIndex) useWeapon else false
                currentState = processCard(currentState, card, shouldUseWeapon)

                if (currentState.isGameOver) {
                    results.add(SolveResult.loss(currentState.calculateScore()))
                    break
                }
            }

            if (currentState.isGameOver) continue

            // Process remaining cards (might have more combat choices)
            if (combatIndex < cardsToProcess.lastIndex) {
                val remainingCards = cardsToProcess.subList(combatIndex + 1, cardsToProcess.size)
                val result = solveProcessingSequence(currentState, remainingCards, cardToLeave)
                results.add(result)
            } else {
                // No more cards to process, continue to next room
                results.add(solve(currentState))
            }
        }

        return SolveResult.combine(results)
    }

    /**
     * Processes a single card, returning the new game state.
     */
    private fun processCard(state: GameState, card: Card, useWeapon: Boolean): GameState {
        return when (card.type) {
            CardType.MONSTER -> {
                if (useWeapon && state.weaponState != null && state.weaponState.canDefeat(card)) {
                    state.fightMonsterWithWeapon(card)
                } else {
                    state.fightMonsterBarehanded(card)
                }
            }
            CardType.WEAPON -> state.equipWeapon(card)
            CardType.POTION -> state.usePotion(card)
        }
    }

    /**
     * Generates all permutations of a list.
     */
    private fun <T> List<T>.permutations(): List<List<T>> {
        if (size <= 1) return listOf(this)

        val result = mutableListOf<List<T>>()
        for (i in indices) {
            val element = this[i]
            val remaining = this.filterIndexed { index, _ -> index != i }
            for (perm in remaining.permutations()) {
                result.add(listOf(element) + perm)
            }
        }
        return result
    }
}
