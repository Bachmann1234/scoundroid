package dev.mattbachmann.scoundroid.data.persistence

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

/**
 * Stores a winning game's complete action log for later analysis.
 * This allows comparing human play decisions against AI strategies.
 */
@Entity(
    tableName = "winning_games",
    indices = [Index(value = ["seed"], unique = true)],
)
data class WinningGame(
    @PrimaryKey(autoGenerate = true)
    val id: Long = 0,
    /** The seed used for this game's deck shuffle */
    val seed: Long,
    /** Timestamp when the game was won */
    val timestamp: Long = System.currentTimeMillis(),
    /** Final health (the winning score) */
    val finalHealth: Int,
    /** JSON-serialized List<LogEntry> */
    val actionLogJson: String,
)
