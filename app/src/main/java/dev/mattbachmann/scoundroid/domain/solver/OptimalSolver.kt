package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState

/**
 * Result of optimal solving - finds if a seed is winnable and the best possible score.
 */
data class OptimalResult(
    val isWinnable: Boolean,
    val bestScore: Int?,
    val nodesExplored: Long,
    val winningPathFound: Boolean,
)

/**
 * Optimal solver using depth-first search with pruning.
 *
 * Instead of exploring all paths, this solver:
 * 1. Stops as soon as it finds ANY winning path (for winnability check)
 * 2. Can optionally continue to find the BEST score
 * 3. Prunes branches that can't possibly improve the best known score
 */
class OptimalSolver {

    private var nodesExplored = 0L
    private var bestScoreFound: Int? = null
    private var foundWin = false

    /**
     * Checks if the game is winnable with optimal play.
     * Stops as soon as a winning path is found.
     *
     * @param initialState The starting game state
     * @param findBestScore If true, continues searching for best score even after finding a win
     * @param maxNodes Maximum nodes to explore before giving up (0 = unlimited)
     * @return OptimalResult with winnability and optional best score
     */
    fun solve(
        initialState: GameState,
        findBestScore: Boolean = false,
        maxNodes: Long = 0
    ): OptimalResult {
        nodesExplored = 0
        bestScoreFound = null
        foundWin = false

        val result = search(initialState, findBestScore, maxNodes)

        return OptimalResult(
            isWinnable = foundWin,
            bestScore = bestScoreFound,
            nodesExplored = nodesExplored,
            winningPathFound = foundWin,
        )
    }

    /**
     * Recursive DFS with early termination.
     * Returns true if a winning path was found in this subtree.
     */
    private fun search(
        state: GameState,
        findBestScore: Boolean,
        maxNodes: Long
    ): Boolean {
        nodesExplored++

        // Check node limit
        if (maxNodes > 0 && nodesExplored >= maxNodes) {
            return foundWin
        }

        // Base case: game over (loss)
        if (state.isGameOver) {
            val score = state.calculateScore()
            if (bestScoreFound == null || score > bestScoreFound!!) {
                bestScoreFound = score
            }
            return false
        }

        // Base case: game won
        if (isActuallyWon(state)) {
            foundWin = true
            val score = state.calculateScore()
            if (bestScoreFound == null || score > bestScoreFound!!) {
                bestScoreFound = score
            }
            return true
        }

        // If we already found a win and don't need best score, we can stop
        if (foundWin && !findBestScore) {
            return true
        }

        // Generate and explore all possible moves
        val nextStates = generateNextStates(state)

        for (nextState in nextStates) {
            val foundWinInBranch = search(nextState, findBestScore, maxNodes)

            // Early termination if we found a win and don't need best score
            if (foundWinInBranch && !findBestScore) {
                return true
            }

            // Check node limit
            if (maxNodes > 0 && nodesExplored >= maxNodes) {
                return foundWin
            }
        }

        return foundWin
    }

    /**
     * Generates all possible next states from the current state.
     * Orders moves heuristically to find wins faster.
     */
    private fun generateNextStates(state: GameState): List<GameState> {
        // If no room, draw one
        if (state.currentRoom == null || state.currentRoom.isEmpty()) {
            return listOf(state.drawRoom())
        }

        val room = state.currentRoom

        // If room has < 4 cards and deck has cards, draw more
        if (room.size < GameState.ROOM_SIZE && !state.deck.isEmpty) {
            return listOf(state.drawRoom())
        }

        // If room has 4 cards, enumerate decisions
        if (room.size == GameState.ROOM_SIZE) {
            return generateRoomDecisions(state, room)
        }

        // End game: process remaining cards
        return generateEndGameDecisions(state, room)
    }

    /**
     * Generates all possible decisions for a room of 4 cards.
     * Orders decisions to prioritize likely winning moves.
     */
    private fun generateRoomDecisions(state: GameState, room: List<Card>): List<GameState> {
        val decisions = mutableListOf<GameState>()

        // Option 1: Process room (try this first - avoiding delays finding wins)
        for (leaveIndex in room.indices) {
            val cardToLeave = room[leaveIndex]
            val cardsToProcess = room.filterIndexed { i, _ -> i != leaveIndex }

            // Order permutations heuristically: weapons first, then potions, then monsters
            val orderedPermutations = cardsToProcess.permutations()
                .sortedByDescending { ordering -> scoreOrdering(ordering, state) }

            for (ordering in orderedPermutations) {
                val resultStates = processCardsInOrder(state, ordering, cardToLeave)
                decisions.addAll(resultStates)
            }
        }

        // Option 2: Avoid room (if allowed) - try last since it delays
        if (!state.lastRoomAvoided) {
            decisions.add(state.avoidRoom())
        }

        return decisions
    }

    /**
     * Scores an ordering heuristically. Higher = more likely to survive.
     * Prefers: weapons early, potions before big monsters, small monsters last.
     */
    private fun scoreOrdering(ordering: List<Card>, state: GameState): Int {
        var score = 0
        var hasWeapon = state.weaponState != null

        for ((index, card) in ordering.withIndex()) {
            when (card.type) {
                CardType.WEAPON -> {
                    // Weapons early is good
                    score += (3 - index) * 10
                    hasWeapon = true
                }
                CardType.POTION -> {
                    // Potions early is okay
                    score += (3 - index) * 5
                }
                CardType.MONSTER -> {
                    // Big monsters late is good, especially if we have weapon
                    if (hasWeapon) {
                        score += index * card.value
                    } else {
                        // Without weapon, small monsters are better
                        score -= card.value
                    }
                }
            }
        }

        return score
    }

    /**
     * Processes cards in a specific order, handling combat choices.
     * Returns all possible resulting states (branches on combat choices).
     */
    private fun processCardsInOrder(
        state: GameState,
        cardsToProcess: List<Card>,
        cardToLeave: Card
    ): List<GameState> {
        var currentStates = listOf(
            state.copy(
                currentRoom = listOf(cardToLeave),
                usedPotionThisTurn = false
            )
        )

        for (card in cardsToProcess) {
            val nextStates = mutableListOf<GameState>()

            for (currentState in currentStates) {
                if (currentState.isGameOver) {
                    nextStates.add(currentState)
                    continue
                }

                when (card.type) {
                    CardType.MONSTER -> {
                        val canUseWeapon = currentState.weaponState != null &&
                            currentState.weaponState.canDefeat(card)

                        if (canUseWeapon) {
                            // Branch: try weapon first (usually better)
                            nextStates.add(currentState.fightMonsterWithWeapon(card))
                            nextStates.add(currentState.fightMonsterBarehanded(card))
                        } else {
                            nextStates.add(currentState.fightMonsterBarehanded(card))
                        }
                    }
                    CardType.WEAPON -> {
                        nextStates.add(currentState.equipWeapon(card))
                    }
                    CardType.POTION -> {
                        nextStates.add(currentState.usePotion(card))
                    }
                }
            }

            currentStates = nextStates
        }

        return currentStates.filter { !it.isGameOver }
    }

    /**
     * Generates all possible decisions for end game (< 4 cards, empty deck).
     */
    private fun generateEndGameDecisions(state: GameState, room: List<Card>): List<GameState> {
        val results = mutableListOf<GameState>()

        // Try all orderings
        for (ordering in room.permutations()) {
            val endStates = processEndGameCards(state.copy(currentRoom = null), ordering)
            results.addAll(endStates)
        }

        return results
    }

    /**
     * Processes end game cards, handling combat choices.
     */
    private fun processEndGameCards(state: GameState, cards: List<Card>): List<GameState> {
        var currentStates = listOf(state)

        for (card in cards) {
            val nextStates = mutableListOf<GameState>()

            for (currentState in currentStates) {
                if (currentState.isGameOver) {
                    nextStates.add(currentState)
                    continue
                }

                when (card.type) {
                    CardType.MONSTER -> {
                        val canUseWeapon = currentState.weaponState != null &&
                            currentState.weaponState.canDefeat(card)

                        if (canUseWeapon) {
                            nextStates.add(currentState.fightMonsterWithWeapon(card))
                            nextStates.add(currentState.fightMonsterBarehanded(card))
                        } else {
                            nextStates.add(currentState.fightMonsterBarehanded(card))
                        }
                    }
                    CardType.WEAPON -> {
                        nextStates.add(currentState.equipWeapon(card))
                    }
                    CardType.POTION -> {
                        nextStates.add(currentState.usePotion(card))
                    }
                }
            }

            currentStates = nextStates
        }

        return currentStates
    }

    private fun isActuallyWon(state: GameState): Boolean {
        return state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0
    }

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
