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
    data class ProcessSelectedCards(val selectedCards: List<Card>) : GameIntent()

    /**
     * Signal that the game has ended. Used to save the score.
     * @param score The final score
     * @param won Whether the player won
     */
    data class GameEnded(val score: Int, val won: Boolean) : GameIntent()
}
