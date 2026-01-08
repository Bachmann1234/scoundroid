package dev.mattbachmann.scoundroid.data.repository

import dev.mattbachmann.scoundroid.data.persistence.HighScore
import dev.mattbachmann.scoundroid.data.persistence.HighScoreDao
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.every
import io.mockk.mockk
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.test.runTest
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNull
import kotlin.test.assertTrue

class HighScoreRepositoryTest {
    private val mockDao = mockk<HighScoreDao>(relaxed = true)
    private val repository = HighScoreRepository(mockDao)

    @Test
    fun `saveScore inserts high score with correct values`() =
        runTest {
            repository.saveScore(score = 15, won = true)

            coVerify {
                mockDao.insert(
                    match { it.score == 15 && it.won },
                )
            }
        }

    @Test
    fun `saveScore inserts losing score correctly`() =
        runTest {
            repository.saveScore(score = -25, won = false)

            coVerify {
                mockDao.insert(
                    match { it.score == -25 && !it.won },
                )
            }
        }

    @Test
    fun `getTopScores returns flow from dao`() =
        runTest {
            val scores =
                listOf(
                    HighScore(id = 1, score = 20, won = true),
                    HighScore(id = 2, score = 15, won = true),
                )
            every { mockDao.getTopHighScores(10) } returns flowOf(scores)

            val result = repository.getTopScores(10).first()

            assertEquals(2, result.size)
            assertEquals(20, result[0].score)
        }

    @Test
    fun `getAllScores returns flow from dao`() =
        runTest {
            val scores =
                listOf(
                    HighScore(id = 1, score = 20, won = true),
                    HighScore(id = 2, score = 10, won = true),
                )
            every { mockDao.getAllHighScores() } returns flowOf(scores)

            val result = repository.getAllScores().first()

            assertEquals(2, result.size)
        }

    @Test
    fun `getHighestScore returns highest from dao`() =
        runTest {
            coEvery { mockDao.getHighestScore() } returns 20

            val result = repository.getHighestScore()

            assertEquals(20, result)
        }

    @Test
    fun `getHighestScore returns null when no scores`() =
        runTest {
            coEvery { mockDao.getHighestScore() } returns null

            val result = repository.getHighestScore()

            assertNull(result)
        }

    @Test
    fun `isNewHighScore returns true when score beats existing`() =
        runTest {
            coEvery { mockDao.getHighestScore() } returns 15

            val result = repository.isNewHighScore(20)

            assertTrue(result)
        }

    @Test
    fun `isNewHighScore returns false when score equals existing`() =
        runTest {
            coEvery { mockDao.getHighestScore() } returns 20

            val result = repository.isNewHighScore(20)

            assertTrue(!result)
        }

    @Test
    fun `isNewHighScore returns false when score below existing`() =
        runTest {
            coEvery { mockDao.getHighestScore() } returns 20

            val result = repository.isNewHighScore(15)

            assertTrue(!result)
        }

    @Test
    fun `isNewHighScore returns true when no existing scores`() =
        runTest {
            coEvery { mockDao.getHighestScore() } returns null

            val result = repository.isNewHighScore(10)

            assertTrue(result)
        }

    @Test
    fun `clearAllScores calls dao deleteAll`() =
        runTest {
            repository.clearAllScores()

            coVerify { mockDao.deleteAll() }
        }

    @Test
    fun `getScoreCount returns count from dao`() =
        runTest {
            coEvery { mockDao.getScoreCount() } returns 5

            val result = repository.getScoreCount()

            assertEquals(5, result)
        }
}
