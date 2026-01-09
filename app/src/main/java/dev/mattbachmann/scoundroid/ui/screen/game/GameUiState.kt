package dev.mattbachmann.scoundroid.ui.screen.game

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.LogEntry
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
    /** The highest score ever achieved, null if no scores saved yet */
    val highestScore: Int? = null,
    /** Whether the current score is a new high score */
    val isNewHighScore: Boolean = false,
    /** Whether to show the help/rules screen */
    val showHelp: Boolean = false,
    /** The action log entries */
    val actionLog: List<LogEntry> = emptyList(),
    /** Whether to show the action log */
    val showActionLog: Boolean = false,
    /** Pending combat choice when player must decide weapon vs barehanded */
    val pendingCombatChoice: PendingCombatChoice? = null,
)

/**
 * Represents a pending combat choice when the player has a weapon
 * that can defeat the current monster.
 */
data class PendingCombatChoice(
    /** The monster being fought */
    val monster: Card,
    /** The weapon available to use */
    val weapon: Card,
    /** Damage taken if using weapon */
    val weaponDamage: Int,
    /** Damage taken if fighting barehanded */
    val barehandedDamage: Int,
    /** What the weapon's max monster value will be after use */
    val weaponDegradedTo: Int,
    /** Cards remaining to process after this monster */
    val remainingCards: List<Card>,
)
