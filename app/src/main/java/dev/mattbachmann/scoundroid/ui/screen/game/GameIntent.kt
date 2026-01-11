package dev.mattbachmann.scoundroid.ui.screen.game

import dev.mattbachmann.scoundroid.data.model.Card

/**
 * Represents user intents/actions in the game.
 * Following MVI (Model-View-Intent) pattern.
 */
sealed class GameIntent {
    /**
     * Start a new game with a fresh deck and reset state.
     */
    data object NewGame : GameIntent()

    /**
     * Draw a new room of 4 cards from the deck.
     */
    data object DrawRoom : GameIntent()

    /**
     * Avoid the current room, sending all cards to the bottom of the deck.
     */
    data object AvoidRoom : GameIntent()

    /**
     * Select and process 3 cards from the room.
     * This combines selection and processing into a single atomic operation.
     * @param selectedCards The 3 cards chosen from the room to process
     */
    data class ProcessSelectedCards(
        val selectedCards: List<Card>,
    ) : GameIntent()

    /**
     * Signal that the game has ended. Used to save the score.
     * @param score The final score
     * @param won Whether the player won
     */
    data class GameEnded(
        val score: Int,
        val won: Boolean,
    ) : GameIntent()

    /**
     * Show the help/rules screen.
     */
    data object ShowHelp : GameIntent()

    /**
     * Hide the help/rules screen.
     */
    data object HideHelp : GameIntent()

    /**
     * Show the action log.
     */
    data object ShowActionLog : GameIntent()

    /**
     * Hide the action log.
     */
    data object HideActionLog : GameIntent()

    /**
     * Resolve a pending combat choice.
     * @param useWeapon true to use weapon (reduced damage, weapon degrades),
     *                  false to fight barehanded (full damage, weapon unchanged)
     */
    data class ResolveCombatChoice(
        val useWeapon: Boolean,
    ) : GameIntent()

    /**
     * Retry the current game with the same seed.
     * Resets the game state but uses the same deck shuffle as the current game.
     */
    data object RetryGame : GameIntent()

    /**
     * Start a new game with a specific seed.
     * @param seed The seed to use for deck shuffling
     */
    data class NewGameWithSeed(
        val seed: Long,
    ) : GameIntent()
}
