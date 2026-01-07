package dev.mattbachmann.scoundroid.ui.screen.game

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.WeaponState

/**
 * UI state for the game screen.
 * Represents all information needed to render the UI.
 */
data class GameUiState(
    /** Current player health (0-20) */
    val health: Int,
    /** Number of cards remaining in the deck */
    val deckSize: Int,
    /** Current room cards (4 cards when drawn, 1 after selection, null when no room) */
    val currentRoom: List<Card>?,
    /** Currently equipped weapon with degradation tracking, if any */
    val weaponState: WeaponState?,
    /** Number of monsters defeated */
    val defeatedMonstersCount: Int,
    /** Current score */
    val score: Int,
    /** Whether the game is over (health = 0) */
    val isGameOver: Boolean,
    /** Whether the player has won (deck empty, health > 0) */
    val isGameWon: Boolean,
    /** Whether the last room was avoided (for tracking consecutive avoidance) */
    val lastRoomAvoided: Boolean,
    /** Whether the player can avoid the current room */
    val canAvoidRoom: Boolean,
)
