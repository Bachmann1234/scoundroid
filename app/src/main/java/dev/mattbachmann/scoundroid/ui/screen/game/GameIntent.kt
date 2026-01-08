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
     * Select 3 of the 4 cards in the room to process.
     * @param selectedCards The 3 cards chosen from the room
     */
    data class SelectCards(val selectedCards: List<Card>) : GameIntent()

    /**
     * Process a single card (monster, weapon, or potion).
     * @param card The card to process
     */
    data class ProcessCard(val card: Card) : GameIntent()

    /**
     * Clear the current room (after processing the last card).
     */
    data object ClearRoom : GameIntent()
}
