package dev.mattbachmann.scoundroid.data.persistence

import androidx.room.Entity
import androidx.room.Index
import androidx.room.PrimaryKey

@Entity(
    tableName = "high_scores",
    indices = [Index(value = ["score"])],
)
data class HighScore(
    @PrimaryKey(autoGenerate = true)
    val id: Int = 0,
    val score: Int,
    val timestamp: Long = System.currentTimeMillis(),
    val won: Boolean = false,
)
