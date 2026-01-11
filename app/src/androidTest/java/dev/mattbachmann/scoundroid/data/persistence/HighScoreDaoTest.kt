package dev.mattbachmann.scoundroid.data.persistence

import android.content.Context
import androidx.room.Room
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.runBlocking
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.test.assertEquals
import kotlin.test.assertTrue

@RunWith(AndroidJUnit4::class)
class HighScoreDaoTest {
    private lateinit var database: AppDatabase
    private lateinit var dao: HighScoreDao

    @Before
    fun setup() {
        val context = ApplicationProvider.getApplicationContext<Context>()
        database =
            Room
                .inMemoryDatabaseBuilder(context, AppDatabase::class.java)
                .allowMainThreadQueries()
                .build()
        dao = database.highScoreDao()
    }

    @After
    fun teardown() {
        database.close()
    }

    @Test
    fun insertHighScore_insertsSuccessfully() =
        runBlocking {
            val highScore = HighScore(score = 15, won = true)
            dao.insert(highScore)

            val allScores = dao.getAllHighScores().first()
            assertEquals(1, allScores.size)
            assertEquals(15, allScores[0].score)
        }

    @Test
    fun getAllHighScores_returnsOrderedByScoreDescending() =
        runBlocking {
            dao.insert(HighScore(score = 10, won = true))
            dao.insert(HighScore(score = 20, won = true))
            dao.insert(HighScore(score = 15, won = true))

            val allScores = dao.getAllHighScores().first()
            assertEquals(3, allScores.size)
            assertEquals(20, allScores[0].score)
            assertEquals(15, allScores[1].score)
            assertEquals(10, allScores[2].score)
        }

    @Test
    fun getTopHighScores_returnsLimitedResults() =
        runBlocking {
            repeat(15) { i ->
                dao.insert(HighScore(score = i, won = true))
            }

            val topScores = dao.getTopHighScores(10).first()
            assertEquals(10, topScores.size)
            assertEquals(14, topScores[0].score)
        }

    @Test
    fun getTopHighScores_returnsOrderedByScoreDescending() =
        runBlocking {
            dao.insert(HighScore(score = 5, won = true))
            dao.insert(HighScore(score = 18, won = true))
            dao.insert(HighScore(score = -10, won = false))
            dao.insert(HighScore(score = 12, won = true))

            val topScores = dao.getTopHighScores(10).first()
            assertEquals(4, topScores.size)
            assertEquals(18, topScores[0].score)
            assertEquals(12, topScores[1].score)
            assertEquals(5, topScores[2].score)
            assertEquals(-10, topScores[3].score)
        }

    @Test
    fun getHighestScore_returnsHighestWhenExists() =
        runBlocking {
            dao.insert(HighScore(score = 10, won = true))
            dao.insert(HighScore(score = 20, won = true))
            dao.insert(HighScore(score = 15, won = true))

            val highest = dao.getHighestScore()
            assertEquals(20, highest)
        }

    @Test
    fun getHighestScore_returnsNullWhenEmpty() =
        runBlocking {
            val highest = dao.getHighestScore()
            assertEquals(null, highest)
        }

    @Test
    fun deleteHighScore_removesScore() =
        runBlocking {
            val highScore = HighScore(score = 15, won = true)
            dao.insert(highScore)

            val allScores = dao.getAllHighScores().first()
            assertEquals(1, allScores.size)

            dao.delete(allScores[0])

            val afterDelete = dao.getAllHighScores().first()
            assertTrue(afterDelete.isEmpty())
        }

    @Test
    fun deleteAllHighScores_removesAllScores() =
        runBlocking {
            dao.insert(HighScore(score = 10, won = true))
            dao.insert(HighScore(score = 20, won = true))
            dao.insert(HighScore(score = 15, won = true))

            dao.deleteAll()

            val allScores = dao.getAllHighScores().first()
            assertTrue(allScores.isEmpty())
        }

    @Test
    fun getScoreCount_returnsCorrectCount() =
        runBlocking {
            assertEquals(0, dao.getScoreCount())

            dao.insert(HighScore(score = 10, won = true))
            dao.insert(HighScore(score = 20, won = true))

            assertEquals(2, dao.getScoreCount())
        }

    @Test
    fun highScoresFlow_emitsUpdatesOnChange() =
        runBlocking {
            val initialScores = dao.getAllHighScores().first()
            assertTrue(initialScores.isEmpty())

            dao.insert(HighScore(score = 15, won = true))

            val updatedScores = dao.getAllHighScores().first()
            assertEquals(1, updatedScores.size)
        }
}
