package dev.mattbachmann.scoundroid.ui

import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertTextContains
import androidx.compose.ui.test.hasClickAction
import androidx.compose.ui.test.hasText
import androidx.compose.ui.test.junit4.ComposeTestRule
import androidx.compose.ui.test.onNodeWithContentDescription
import androidx.compose.ui.test.onNodeWithTag
import androidx.compose.ui.test.performClick
import androidx.compose.ui.test.performScrollTo

/*
 * Test utility functions for Scoundroid E2E tests.
 * Provides common operations for interacting with the game UI.
 *
 * Uses waitUntil with extended timeouts to handle slower CI emulators.
 */

// Extended timeout for CI emulators which are slower than local hardware-accelerated emulators
private const val CI_TIMEOUT_MS = 10_000L

/**
 * Clicks the "Draw Room" button to draw a new room.
 * Waits for button to be available, scrolls to it, then clicks.
 */
fun ComposeTestRule.drawRoom() {
    waitUntilNodeExists(hasText("Draw Room") and hasClickAction())
    onNode(hasText("Draw Room") and hasClickAction())
        .performScrollTo()
        .performClick()
    waitForIdle()
}

/**
 * Waits until a node matching the given matcher exists in the UI tree.
 * Uses extended timeout for slower CI emulators.
 */
fun ComposeTestRule.waitUntilNodeExists(matcher: SemanticsMatcher) {
    waitUntil(CI_TIMEOUT_MS) {
        onAllNodes(matcher).fetchSemanticsNodes().isNotEmpty()
    }
}

/**
 * Waits until a node matching the given matcher exists, scrolls to it, and verifies it's displayed.
 * This handles both timing issues on slow CI emulators and buttons that may be off-screen
 * on smaller screen sizes.
 */
private fun ComposeTestRule.waitUntilNodeIsDisplayed(matcher: SemanticsMatcher) {
    waitUntil(CI_TIMEOUT_MS) {
        onAllNodes(matcher).fetchSemanticsNodes().isNotEmpty()
    }
    onNode(matcher).performScrollTo()
    onNode(matcher).assertIsDisplayed()
}

/**
 * Clicks the "Draw Next Room" button to draw cards into an existing room.
 * Scrolls to the button first in case it's off-screen.
 */
fun ComposeTestRule.drawNextRoom() {
    onNode(hasText("Draw Next Room") and hasClickAction())
        .performScrollTo()
        .performClick()
    waitForIdle()
}

/**
 * Clicks the process cards button and handles any combat choices that appear.
 * Only works when exactly 3 cards are selected.
 * Scrolls to the button first in case it's off-screen.
 * Combat choices are automatically resolved by using the weapon.
 */
fun ComposeTestRule.processCards() {
    onNode(hasText("Go") and hasClickAction())
        .performScrollTo()
        .performClick()
    waitForIdle()

    // Handle any combat choices that appear during processing
    handleCombatChoices()
}

/**
 * Handles combat choice panels by clicking "Use Weapon" until no more choices appear.
 * This is needed because processing cards may pause for player decisions on weapon use.
 */
fun ComposeTestRule.handleCombatChoices() {
    var maxAttempts = 10 // Safety limit
    while (maxAttempts > 0) {
        // Wait for any pending UI updates
        waitForIdle()

        try {
            // Check if combat choice panel is showing and click it
            onNode(hasText("Use Weapon") and hasClickAction())
                .performClick()
            waitForIdle()
            maxAttempts--
        } catch (_: AssertionError) {
            // No combat choice visible, we're done
            break
        }
    }
}

/**
 * Clicks the "Avoid Room" button to avoid the current room.
 * Only available when room avoidance is allowed.
 * Scrolls to the button first in case it's off-screen.
 */
fun ComposeTestRule.avoidRoom() {
    onNode(hasText("Avoid Room") and hasClickAction())
        .performScrollTo()
        .performClick()
    waitForIdle()
}

/**
 * Starts a new game by clicking the "New Game" button.
 * Scrolls to the button first in case it's off-screen.
 */
fun ComposeTestRule.startNewGame() {
    onNode(hasText("New Game") and hasClickAction())
        .performScrollTo()
        .performClick()
    waitForIdle()
}

/**
 * Opens the help modal by clicking the help button.
 */
fun ComposeTestRule.openHelp() {
    onNodeWithContentDescription("Help").performClick()
    waitForIdle()
}

/**
 * Opens the action log modal by clicking the action log button.
 */
fun ComposeTestRule.openActionLog() {
    onNodeWithContentDescription("Action Log").performClick()
    waitForIdle()
}

/**
 * Asserts that the health display shows the expected value.
 */
fun ComposeTestRule.assertHealth(expected: Int) {
    onNodeWithTag("health_display").assertTextContains("$expected / 20")
}

/**
 * Asserts that the deck size display shows the expected value.
 */
fun ComposeTestRule.assertDeckSize(expected: Int) {
    onNodeWithTag("deck_size_display").assertTextContains("$expected cards")
}

/**
 * Asserts that the "Draw Room" button is visible.
 * Waits for the button, scrolls to it, and verifies it's displayed.
 */
fun ComposeTestRule.assertDrawRoomButtonVisible() {
    waitUntilNodeIsDisplayed(hasText("Draw Room") and hasClickAction())
}

/**
 * Asserts that the "Draw Next Room" button is visible.
 * Waits for the button, scrolls to it, and verifies it's displayed.
 */
fun ComposeTestRule.assertDrawNextRoomButtonVisible() {
    waitUntilNodeIsDisplayed(hasText("Draw Next Room") and hasClickAction())
}

/**
 * Asserts that the "Avoid Room" button is visible.
 * Waits for the button, scrolls to it, and verifies it's displayed.
 */
fun ComposeTestRule.assertAvoidRoomButtonVisible() {
    waitUntilNodeIsDisplayed(hasText("Avoid Room") and hasClickAction())
}

/**
 * Asserts that the "Avoid Room" button is not visible (doesn't exist or hidden).
 */
fun ComposeTestRule.assertAvoidRoomButtonNotVisible() {
    // Use assertDoesNotExist since the button is conditionally rendered
    onNode(hasText("Avoid Room") and hasClickAction()).assertDoesNotExist()
}
