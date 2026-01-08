package dev.mattbachmann.scoundroid.data.persistence

import androidx.room.Dao
import androidx.room.Delete
import androidx.room.Insert
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface HighScoreDao {
    @Insert
    suspend fun insert(highScore: HighScore)

    @Delete
    suspend fun delete(highScore: HighScore)

    @Query("DELETE FROM high_scores")
    suspend fun deleteAll()

    @Query("SELECT * FROM high_scores ORDER BY score DESC")
    fun getAllHighScores(): Flow<List<HighScore>>

    @Query("SELECT * FROM high_scores ORDER BY score DESC LIMIT :limit")
    fun getTopHighScores(limit: Int): Flow<List<HighScore>>

    @Query("SELECT MAX(score) FROM high_scores")
    suspend fun getHighestScore(): Int?

    @Query("SELECT COUNT(*) FROM high_scores")
    suspend fun getScoreCount(): Int
}
