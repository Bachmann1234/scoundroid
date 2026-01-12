package dev.mattbachmann.scoundroid.data.persistence

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import kotlinx.coroutines.flow.Flow

@Dao
interface WinningGameDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(winningGame: WinningGame)

    @Query("SELECT * FROM winning_games ORDER BY timestamp DESC")
    fun getAllWinningGames(): Flow<List<WinningGame>>

    @Query("SELECT * FROM winning_games ORDER BY timestamp DESC")
    suspend fun getAllWinningGamesSync(): List<WinningGame>

    @Query("SELECT COUNT(*) FROM winning_games")
    suspend fun getWinCount(): Int

    @Query("SELECT * FROM winning_games WHERE seed = :seed LIMIT 1")
    suspend fun getGameBySeed(seed: Long): WinningGame?

    @Query("DELETE FROM winning_games")
    suspend fun deleteAll()
}
