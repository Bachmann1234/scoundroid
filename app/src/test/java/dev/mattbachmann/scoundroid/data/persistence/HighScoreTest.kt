package dev.mattbachmann.scoundroid.data.persistence

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue

class HighScoreTest {
    @Test
    fun `HighScore stores score value`() {
        val highScore = HighScore(score = 15)
        assertEquals(15, highScore.score)
    }

    @Test
    fun `HighScore stores timestamp`() {
        val timestamp = System.currentTimeMillis()
        val highScore = HighScore(score = 10, timestamp = timestamp)
        assertEquals(timestamp, highScore.timestamp)
    }

    @Test
    fun `HighScore has default timestamp`() {
        val before = System.currentTimeMillis()
        val highScore = HighScore(score = 10)
        val after = System.currentTimeMillis()
        assertTrue(highScore.timestamp in before..after)
    }

    @Test
    fun `HighScore has auto-generated id when not specified`() {
        val highScore = HighScore(score = 10)
        assertEquals(0, highScore.id)
    }

    @Test
    fun `HighScore can have explicit id`() {
        val highScore = HighScore(id = 5, score = 10)
        assertEquals(5, highScore.id)
    }

    @Test
    fun `HighScore stores whether game was won`() {
        val wonGame = HighScore(score = 15, won = true)
        val lostGame = HighScore(score = -20, won = false)
        assertTrue(wonGame.won)
        assertTrue(!lostGame.won)
    }

    @Test
    fun `HighScore equality is based on all fields`() {
        val timestamp = 1000L
        val score1 = HighScore(id = 1, score = 15, timestamp = timestamp, won = true)
        val score2 = HighScore(id = 1, score = 15, timestamp = timestamp, won = true)
        val score3 = HighScore(id = 2, score = 15, timestamp = timestamp, won = true)

        assertEquals(score1, score2)
        assertNotEquals(score1, score3)
    }

    @Test
    fun `HighScore can store negative scores for losses`() {
        val highScore = HighScore(score = -45, won = false)
        assertEquals(-45, highScore.score)
    }

    @Test
    fun `HighScore can store max possible score of 20`() {
        val highScore = HighScore(score = 20, won = true)
        assertEquals(20, highScore.score)
    }

    @Test
    fun `HighScore can store score above 20 for potion bonus`() {
        val highScore = HighScore(score = 30, won = true)
        assertEquals(30, highScore.score)
    }
}
