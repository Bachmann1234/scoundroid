package dev.mattbachmann.scoundroid.ui.screen.game

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertIs

/**
 * Tests for GameDisplayMode to ensure correct priority ordering.
 *
 * Priority (highest to lowest):
 * 1. GameOver - player died (health <= 0)
 * 2. GameWon - player survived the dungeon (deck empty, health > 0)
 * 3. CombatChoice - player must choose weapon vs barehanded
 * 4. ActiveGame - normal gameplay
 *
 * These tests are critical to prevent the bug where combat buttons
 * were shown after game over in expanded mode.
 */
class GameDisplayModeTest {
    private fun createUiState(
        health: Int = 20,
        isGameOver: Boolean = false,
        isGameWon: Boolean = false,
        pendingCombatChoice: PendingCombatChoice? = null,
        currentRoom: List<Card>? = null,
        canAvoidRoom: Boolean = false,
    ) = GameUiState(
        health = health,
        deckSize = 40,
        currentRoom = currentRoom,
        weaponState = null,
        defeatedMonstersCount = 0,
        score = health,
        isGameOver = isGameOver,
        isGameWon = isGameWon,
        lastRoomAvoided = false,
        canAvoidRoom = canAvoidRoom,
        pendingCombatChoice = pendingCombatChoice,
        gameSeed = 12345L,
    )

    private fun createPendingCombatChoice() =
        PendingCombatChoice(
            monster = Card(Suit.CLUBS, Rank.QUEEN),
            weapon = Card(Suit.DIAMONDS, Rank.FIVE),
            weaponDamage = 7,
            barehandedDamage = 12,
            weaponDegradedTo = 12,
            remainingCards = emptyList(),
        )

    // ==================== Basic Mode Tests ====================

    @Test
    fun `displayMode returns GameOver when isGameOver is true`() {
        val uiState = createUiState(isGameOver = true, health = 0)

        assertIs<GameDisplayMode.GameOver>(uiState.displayMode)
    }

    @Test
    fun `displayMode returns GameWon when isGameWon is true`() {
        val uiState = createUiState(isGameWon = true)

        assertIs<GameDisplayMode.GameWon>(uiState.displayMode)
    }

    @Test
    fun `displayMode returns CombatChoice when pendingCombatChoice is set`() {
        val uiState = createUiState(pendingCombatChoice = createPendingCombatChoice())

        assertIs<GameDisplayMode.CombatChoice>(uiState.displayMode)
    }

    @Test
    fun `displayMode returns ActiveGame when no special state`() {
        val uiState = createUiState()

        assertIs<GameDisplayMode.ActiveGame>(uiState.displayMode)
    }

    // ==================== Priority Tests ====================

    @Test
    fun `GameOver takes priority over CombatChoice - the core bug fix`() {
        // This is THE bug: player dies during combat but combat UI was still shown
        val uiState =
            createUiState(
                isGameOver = true,
                health = 0,
                pendingCombatChoice = createPendingCombatChoice(),
            )

        val mode = uiState.displayMode

        // Should be GameOver, NOT CombatChoice
        assertIs<GameDisplayMode.GameOver>(mode)
    }

    @Test
    fun `GameOver takes priority over GameWon`() {
        // Edge case: shouldn't happen in practice, but verify priority
        val uiState =
            createUiState(
                isGameOver = true,
                isGameWon = true,
            )

        assertIs<GameDisplayMode.GameOver>(uiState.displayMode)
    }

    @Test
    fun `GameWon takes priority over CombatChoice`() {
        // Edge case: combat choice present but deck is empty and player alive
        val uiState =
            createUiState(
                isGameWon = true,
                pendingCombatChoice = createPendingCombatChoice(),
            )

        assertIs<GameDisplayMode.GameWon>(uiState.displayMode)
    }

    @Test
    fun `CombatChoice takes priority over ActiveGame`() {
        val uiState =
            createUiState(
                pendingCombatChoice = createPendingCombatChoice(),
                currentRoom = listOf(Card(Suit.CLUBS, Rank.TWO)),
            )

        assertIs<GameDisplayMode.CombatChoice>(uiState.displayMode)
    }

    // ==================== Data Passing Tests ====================

    @Test
    fun `GameOver mode contains correct score and seed`() {
        val uiState =
            createUiState(
                isGameOver = true,
                health = 0,
            ).copy(score = -30, highestScore = 10, isNewHighScore = false)

        val mode = uiState.displayMode as GameDisplayMode.GameOver

        assertEquals(-30, mode.score)
        assertEquals(10, mode.highestScore)
        assertEquals(false, mode.isNewHighScore)
        assertEquals(12345L, mode.gameSeed)
    }

    @Test
    fun `GameWon mode contains correct score and seed`() {
        val uiState =
            createUiState(
                isGameWon = true,
                health = 15,
            ).copy(score = 15, highestScore = 10, isNewHighScore = true)

        val mode = uiState.displayMode as GameDisplayMode.GameWon

        assertEquals(15, mode.score)
        assertEquals(10, mode.highestScore)
        assertEquals(true, mode.isNewHighScore)
        assertEquals(12345L, mode.gameSeed)
    }

    @Test
    fun `CombatChoice mode contains the pending choice`() {
        val choice = createPendingCombatChoice()
        val uiState = createUiState(pendingCombatChoice = choice)

        val mode = uiState.displayMode as GameDisplayMode.CombatChoice

        assertEquals(choice, mode.choice)
    }

    @Test
    fun `ActiveGame mode contains room and avoid info`() {
        val room =
            listOf(
                Card(Suit.CLUBS, Rank.TWO),
                Card(Suit.SPADES, Rank.THREE),
                Card(Suit.HEARTS, Rank.FOUR),
                Card(Suit.DIAMONDS, Rank.FIVE),
            )
        val uiState = createUiState(currentRoom = room, canAvoidRoom = true)

        val mode = uiState.displayMode as GameDisplayMode.ActiveGame

        assertEquals(room, mode.currentRoom)
        assertEquals(true, mode.canAvoidRoom)
    }

    @Test
    fun `ActiveGame mode with null room`() {
        val uiState = createUiState(currentRoom = null)

        val mode = uiState.displayMode as GameDisplayMode.ActiveGame

        assertEquals(null, mode.currentRoom)
    }
}
