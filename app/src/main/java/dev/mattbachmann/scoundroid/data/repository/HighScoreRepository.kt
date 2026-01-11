package dev.mattbachmann.scoundroid.data.repository

import dev.mattbachmann.scoundroid.data.persistence.HighScore
import dev.mattbachmann.scoundroid.data.persistence.HighScoreDao
import kotlinx.coroutines.flow.Flow

class HighScoreRepository(
    private val dao: HighScoreDao,
) {
    suspend fun saveScore(
        score: Int,
        won: Boolean,
    ) {
        dao.insert(HighScore(score = score, won = won))
    }

    fun getTopScores(limit: Int): Flow<List<HighScore>> = dao.getTopHighScores(limit)

    fun getAllScores(): Flow<List<HighScore>> = dao.getAllHighScores()

    suspend fun getHighestScore(): Int? = dao.getHighestScore()

    suspend fun isNewHighScore(score: Int): Boolean {
        val currentHighest = dao.getHighestScore()
        return currentHighest == null || score > currentHighest
    }

    suspend fun clearAllScores() {
        dao.deleteAll()
    }

    suspend fun getScoreCount(): Int = dao.getScoreCount()
}
