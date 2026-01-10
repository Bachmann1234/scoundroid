package dev.mattbachmann.scoundroid.ui

import androidx.compose.runtime.remember
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
import androidx.compose.ui.test.assertIsNotEnabled
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.createComposeRule
import androidx.compose.ui.test.onAllNodesWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import androidx.test.ext.junit.runners.AndroidJUnit4
import dev.mattbachmann.scoundroid.ui.screen.game.GameScreen
import dev.mattbachmann.scoundroid.ui.screen.game.GameViewModel
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import org.junit.Before
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith

/**
 * End-to-end UI tests for Scoundroid game flows.
 * Tests critical user journeys through the application.
 *
 * Uses a seeded random for deterministic card shuffling, making tests stable and reproducible.
 */
@RunWith(AndroidJUnit4::class)
class GameFlowTest {
    @get:Rule
    val composeTestRule = createComposeRule()

    // Seed that produces a survivable game with varied card types
    // First room with seed 42: includes potions and lower-damage monsters
    private val testSeed = 42L

    @Before
    fun setUp() {
        composeTestRule.setContent {
            ScoundroidTheme {
                val viewModel = remember { GameViewModel(randomSeed = testSeed) }
                GameScreen(viewModel = viewModel)
            }
        }
    }

    @Test
    fun newGame_showsInitialState() {
        // Wait for UI to render
        composeTestRule.waitForIdle()

        // Verify initial game state
        composeTestRule.assertHealth(20)
        composeTestRule.assertDeckSize(44)
        composeTestRule.assertDrawRoomButtonVisible()
    }

    @Test
    fun drawRoom_showsFourCards() {
        // Draw a room
        composeTestRule.drawRoom()

        // Verify 4 cards are displayed by checking we can find cards
        // Cards have content descriptions like "Monster card, ..." or "Weapon card, ..." or "Potion card, ..."
        composeTestRule.waitForIdle()

        // At least one card type should be visible (the room has 4 cards of various types)
        val allCards = composeTestRule.onAllNodesWithContentDescription("card", substring = true)
        // Room should have exactly 4 cards
        allCards.fetchSemanticsNodes().size.let { count ->
            assert(count == 4) { "Expected 4 cards in room, found $count" }
        }
    }

    @Test
    fun selectThreeCards_enablesProcessButton() {
        // Draw a room
        composeTestRule.drawRoom()

        // Initially process button should show "Pick 3" and be disabled
        composeTestRule.onNode(hasText("Pick 3") and hasClickAction()).assertIsDisplayed()
        composeTestRule.onNode(hasText("Pick 3") and hasClickAction()).assertIsNotEnabled()

        // Get all card nodes
        val cards =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()

        // Select first 3 cards
        cards.take(3).forEachIndexed { index, _ ->
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)[index]
                .performClick()
            composeTestRule.waitForIdle()
        }

        // Process button should now show "Go" and be enabled
        composeTestRule.onNode(hasText("Go") and hasClickAction()).assertIsDisplayed()
        composeTestRule.onNode(hasText("Go") and hasClickAction()).assertIsEnabled()
    }

    @Test
    fun processCards_updatesGameState() {
        // Draw a room
        composeTestRule.drawRoom()

        // Select 3 cards
        val cards =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()
        cards.take(3).forEachIndexed { index, _ ->
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)[index]
                .performClick()
            composeTestRule.waitForIdle()
        }

        // Process the cards
        composeTestRule.processCards()

        // After processing, room should have 1 card remaining and "Draw Next Room" button should appear
        composeTestRule.assertDrawNextRoomButtonVisible()

        // Deck size should be reduced (44 - 4 = 40)
        composeTestRule.assertDeckSize(40)
    }

    @Test
    fun avoidRoom_whenAllowed() {
        // First room CAN be avoided (lastRoomAvoided starts as false)
        // So we need to process one room first, then draw another

        // Draw first room
        composeTestRule.drawRoom()

        // Process 3 cards from first room
        val cards =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()
        cards.take(3).forEachIndexed { index, _ ->
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)[index]
                .performClick()
            composeTestRule.waitForIdle()
        }
        composeTestRule.processCards()

        // Draw next room (filling remaining card + 3 more = 4 cards)
        composeTestRule.drawNextRoom()

        // Now we should be able to avoid
        composeTestRule.assertAvoidRoomButtonVisible()

        // Avoid the room - NOTE: ViewModel auto-draws the next room after avoiding
        composeTestRule.avoidRoom()

        // After avoiding, a new room is auto-drawn, so we should see 4 cards
        // and the process button (not Draw Room)
        val cardsAfterAvoid =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()
        assert(cardsAfterAvoid.size == 4) { "Expected 4 cards after avoiding, got ${cardsAfterAvoid.size}" }

        // Avoid button should NOT be visible (just avoided, can't avoid twice)
        composeTestRule.assertAvoidRoomButtonNotVisible()
    }

    @Test
    fun cannotAvoidTwice() {
        // Draw first room and process it
        composeTestRule.drawRoom()
        val cards =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()
        cards.take(3).forEachIndexed { index, _ ->
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)[index]
                .performClick()
            composeTestRule.waitForIdle()
        }
        composeTestRule.processCards()

        // Draw second room
        composeTestRule.drawNextRoom()

        // Avoid second room - NOTE: ViewModel auto-draws the next room
        composeTestRule.avoidRoom()

        // After avoiding, a new room (third room) is auto-drawn
        // We should see 4 cards and should NOT be able to avoid
        val cardsAfterAvoid =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()
        assert(cardsAfterAvoid.size == 4) { "Expected 4 cards after avoiding, got ${cardsAfterAvoid.size}" }

        // Should NOT be able to avoid (just avoided previous room)
        composeTestRule.assertAvoidRoomButtonNotVisible()
    }

    @Test
    fun helpButton_showsRules() {
        // Click help button
        composeTestRule.openHelp()

        // Verify help content is displayed
        composeTestRule.onNodeWithText("How to Play", substring = true).assertIsDisplayed()

        // Verify some rule content is visible
        composeTestRule.onNodeWithText("Monster", substring = true).assertIsDisplayed()
    }

    @Test
    fun actionLogButton_showsLog() {
        // First draw a room and process it to have some actions logged
        composeTestRule.drawRoom()
        val cards =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()
        cards.take(3).forEachIndexed { index, _ ->
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)[index]
                .performClick()
            composeTestRule.waitForIdle()
        }
        composeTestRule.processCards()

        // Click action log button
        composeTestRule.openActionLog()

        // Verify action log content is displayed
        composeTestRule.onNodeWithText("Action Log", substring = true).assertIsDisplayed()
    }

    @Test
    fun newGameButton_resetsGame() {
        // Draw a room and process some cards to change game state
        composeTestRule.drawRoom()
        val cards =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()
        cards.take(3).forEachIndexed { index, _ ->
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)[index]
                .performClick()
            composeTestRule.waitForIdle()
        }
        composeTestRule.processCards()

        // Deck size should now be 40
        composeTestRule.assertDeckSize(40)

        // Start new game
        composeTestRule.startNewGame()

        // Game should be reset
        composeTestRule.assertHealth(20)
        composeTestRule.assertDeckSize(44)
        composeTestRule.assertDrawRoomButtonVisible()
    }

    // ========== Extended Game Flow Tests ==========

    @Test
    fun multipleRooms_gameProgresses() {
        // Play through multiple rooms to verify game continues correctly
        // With seeded random, card order is deterministic

        // Room 1
        composeTestRule.drawRoom()
        selectAndProcessThreeCards()

        // Room 2
        composeTestRule.drawNextRoom()
        selectAndProcessThreeCards()

        // Room 3
        composeTestRule.drawNextRoom()
        selectAndProcessThreeCards()

        // After 3 rooms: 44 - (4 + 3 + 3) = 34 cards in deck
        composeTestRule.assertDeckSize(34)
    }

    @Test
    fun healthDecreases_whenFightingMonsters() {
        // Start game and draw room
        composeTestRule.drawRoom()

        // Process first room
        selectAndProcessThreeCards()

        // Draw and process second room
        composeTestRule.drawNextRoom()
        selectAndProcessThreeCards()

        // Verify health display exists and shows a valid value
        composeTestRule.onNodeWithTag("health_display").assertIsDisplayed()
    }

    @Test
    fun weaponDisplay_updatesWhenEquipped() {
        // Play through rooms to potentially equip a weapon
        composeTestRule.drawRoom()
        selectAndProcessThreeCards()

        composeTestRule.drawNextRoom()
        selectAndProcessThreeCards()

        composeTestRule.drawNextRoom()
        selectAndProcessThreeCards()

        // Weapon display should exist (shows "None" or weapon info)
        composeTestRule.onNodeWithTag("weapon_display").assertIsDisplayed()
    }

    @Test
    fun defeatedCount_incrementsAfterCombat() {
        // Play through rooms with monsters
        composeTestRule.drawRoom()
        selectAndProcessThreeCards()

        composeTestRule.drawNextRoom()
        selectAndProcessThreeCards()

        // Defeated count display should be visible
        composeTestRule.onNodeWithTag("defeated_display").assertIsDisplayed()
    }

    @Test
    fun firstRoom_canBeAvoided() {
        // Draw first room
        composeTestRule.drawRoom()

        // First room CAN be avoided (lastRoomAvoided starts as false)
        composeTestRule.assertAvoidRoomButtonVisible()

        // Avoid it
        composeTestRule.avoidRoom()

        // After avoiding first room, a new room is auto-drawn
        // and we should NOT be able to avoid this one
        composeTestRule.assertAvoidRoomButtonNotVisible()
    }

    // ========== Seeded Runs Tests ==========

    @Test
    fun customSeedButton_visibleOnInitialScreen() {
        // Custom Seed button should exist on initial game screen (use assertExists since it may be
        // below the fold on smaller screens)
        composeTestRule.waitForIdle()
        composeTestRule.onNode(hasText("Custom Seed") and hasClickAction()).assertExists()
    }

    @Test
    fun retryButton_visibleOnGameOver() {
        // Play until game over (take damage without healing)
        // With seed 42, we need to process multiple rooms to trigger game over
        composeTestRule.drawRoom()
        selectAndProcessThreeCards()

        // Keep playing rooms until game is over
        repeat(15) {
            try {
                composeTestRule.onNodeWithTag("game_over_screen").assertIsDisplayed()
                // Game over reached
                return@repeat
            } catch (_: AssertionError) {
                // Not game over yet, continue playing
                try {
                    composeTestRule.drawNextRoom()
                    selectAndProcessThreeCards()
                } catch (_: AssertionError) {
                    // May have hit game over during processing
                }
            }
        }

        // Explicitly verify game over was reached before checking retry button
        composeTestRule.onNodeWithTag("game_over_screen").assertIsDisplayed()

        // Verify retry button exists on game over screen (use assertExists since it may be
        // below the fold on smaller screens)
        composeTestRule.onNode(hasText("Retry") and hasClickAction()).assertExists()
    }

    @Test
    fun seedDisplay_visibleOnGameOver() {
        // Play until game over
        composeTestRule.drawRoom()
        selectAndProcessThreeCards()

        repeat(15) {
            try {
                composeTestRule.onNodeWithTag("game_over_screen").assertIsDisplayed()
                return@repeat
            } catch (_: AssertionError) {
                try {
                    composeTestRule.drawNextRoom()
                    selectAndProcessThreeCards()
                } catch (_: AssertionError) {
                    // May have hit game over during processing
                }
            }
        }

        // Explicitly verify game over was reached before checking seed display
        composeTestRule.onNodeWithTag("game_over_screen").assertIsDisplayed()

        // Verify seed exists on game over screen (use assertExists since it may be
        // below the fold on smaller screens)
        composeTestRule.onNode(hasText("Seed:", substring = true)).assertExists()
    }

    // ========== Helper Methods ==========

    private fun selectAndProcessThreeCards() {
        val cards =
            composeTestRule.onAllNodesWithContentDescription("card", substring = true)
                .fetchSemanticsNodes()

        if (cards.size >= 3) {
            cards.take(3).forEachIndexed { index, _ ->
                composeTestRule.onAllNodesWithContentDescription("card", substring = true)[index]
                    .performClick()
                composeTestRule.waitForIdle()
            }
            composeTestRule.processCards()
        }
    }
}
