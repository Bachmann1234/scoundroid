package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import kotlin.random.Random

/**
 * Result of Monte Carlo simulation for a game seed.
 *
 * @property samples Number of games simulated
 * @property wins Number of winning games
 * @property losses Number of losing games
 * @property winProbability Estimated probability of winning with random play
 * @property averageWinScore Average score among winning games
 * @property averageLossScore Average score among losing games
 * @property maxScore Highest score observed
 * @property minScore Lowest score observed
 */
data class SimulationResult(
    val samples: Int,
    val wins: Int,
    val losses: Int,
    val winProbability: Double,
    val averageWinScore: Double?,
    val averageLossScore: Double?,
    val maxScore: Int,
    val minScore: Int,
) {
    val isWinnable: Boolean get() = wins > 0
}

/**
 * Monte Carlo simulator for Scoundrel games.
 *
 * Instead of exhaustively exploring all paths (which is intractable),
 * this simulator runs many random games and estimates win probability.
 */
class MonteCarloSimulator {
    /**
     * Simulates many random games from a given starting state.
     *
     * @param initialState The starting game state (from a specific seed)
     * @param samples Number of games to simulate
     * @param random Random source for making decisions
     * @return SimulationResult with statistics
     */
    fun simulate(
        initialState: GameState,
        samples: Int = 10000,
        random: Random = Random,
    ): SimulationResult {
        var wins = 0
        var losses = 0
        var totalWinScore = 0L
        var totalLossScore = 0L
        var maxScore = Int.MIN_VALUE
        var minScore = Int.MAX_VALUE

        repeat(samples) {
            val finalState = playRandomGame(initialState, random)
            val score = finalState.calculateScore()

            maxScore = maxOf(maxScore, score)
            minScore = minOf(minScore, score)

            if (isActuallyWon(finalState)) {
                wins++
                totalWinScore += score
            } else {
                losses++
                totalLossScore += score
            }
        }

        return SimulationResult(
            samples = samples,
            wins = wins,
            losses = losses,
            winProbability = wins.toDouble() / samples,
            averageWinScore = if (wins > 0) totalWinScore.toDouble() / wins else null,
            averageLossScore = if (losses > 0) totalLossScore.toDouble() / losses else null,
            maxScore = maxScore,
            minScore = minScore,
        )
    }

    /**
     * Plays a single random game to completion.
     */
    private fun playRandomGame(
        initialState: GameState,
        random: Random,
    ): GameState {
        var state = initialState

        while (!state.isGameOver && !isActuallyWon(state)) {
            state = playOneStep(state, random)
        }

        return state
    }

    /**
     * Checks if the game is actually won (deck empty AND no cards left to process).
     * The GameState.isGameWon only checks deck.isEmpty, but we need room to be empty too.
     */
    private fun isActuallyWon(state: GameState): Boolean =
        state.deck.isEmpty &&
            (state.currentRoom == null || state.currentRoom.isEmpty()) &&
            state.health > 0

    /**
     * Plays one step of the game with random decisions.
     */
    private fun playOneStep(
        state: GameState,
        random: Random,
    ): GameState {
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
            // Random choice to avoid (if allowed)
            if (!state.lastRoomAvoided && random.nextBoolean()) {
                return state.avoidRoom()
            }

            // Process the room: randomly choose which card to leave
            val leaveIndex = random.nextInt(room.size)
            val cardToLeave = room[leaveIndex]
            val cardsToProcess = room.filterIndexed { i, _ -> i != leaveIndex }.shuffled(random)

            // Process the 3 cards in random order
            var currentState =
                state.copy(
                    currentRoom = listOf(cardToLeave),
                    usedPotionThisTurn = false,
                )

            for (card in cardsToProcess) {
                currentState = processCard(currentState, card, random)
                if (currentState.isGameOver) {
                    return currentState
                }
            }

            return currentState
        }

        // End game: room has < 4 cards and deck is empty
        // Process all remaining cards in random order
        val cardsToProcess = room.shuffled(random)
        var currentState = state.copy(currentRoom = null)

        for (card in cardsToProcess) {
            currentState = processCard(currentState, card, random)
            if (currentState.isGameOver) {
                return currentState
            }
        }

        return currentState
    }

    /**
     * Processes a single card with random combat choice if applicable.
     */
    private fun processCard(
        state: GameState,
        card: Card,
        random: Random,
    ): GameState =
        when (card.type) {
            CardType.MONSTER -> {
                // If we have a usable weapon, randomly decide whether to use it
                val canUseWeapon =
                    state.weaponState != null &&
                        state.weaponState.canDefeat(card)

                if (canUseWeapon && random.nextBoolean()) {
                    state.fightMonsterWithWeapon(card)
                } else {
                    state.fightMonsterBarehanded(card)
                }
            }
            CardType.WEAPON -> state.equipWeapon(card)
            CardType.POTION -> state.usePotion(card)
        }

    /**
     * Simulates games for multiple seeds and returns aggregate statistics.
     */
    fun simulateMultipleSeeds(
        seedRange: LongRange,
        samplesPerSeed: Int = 1000,
        random: Random = Random,
    ): Map<Long, SimulationResult> =
        seedRange.associateWith { seed ->
            val game = GameState.newGame(Random(seed))
            simulate(game, samplesPerSeed, random)
        }
}
