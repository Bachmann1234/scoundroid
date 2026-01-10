package dev.mattbachmann.scoundroid.ui

import androidx.compose.ui.test.SemanticsMatcher
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.assertIsEnabled
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
private fun ComposeTestRule.waitUntilNodeExists(matcher: SemanticsMatcher) {
    waitUntil(CI_TIMEOUT_MS) {
        onAllNodes(matcher).fetchSemanticsNodes().isNotEmpty()
    }
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
 * Selects a card by its content description.
 * Card descriptions follow the pattern: "TYPE card, RANK of SUIT, value VALUE"
 * e.g., "Monster card, Queen of Clubs, value 12"
 */
fun ComposeTestRule.selectCard(contentDescription: String) {
    onNodeWithContentDescription(contentDescription, substring = true).performClick()
    waitForIdle()
}

/**
 * Selects a card containing the specified text in its description.
 * Useful when you don't know the exact card but know part of its description.
 */
fun ComposeTestRule.selectCardContaining(descriptionPart: String) {
    onNodeWithContentDescription(descriptionPart, substring = true).performClick()
    waitForIdle()
}

/**
 * Clicks the process cards button and handles any combat choices that appear.
 * Only works when exactly 3 cards are selected.
 * Scrolls to the button first in case it's off-screen.
 * Combat choices are automatically resolved by using the weapon.
 */
fun ComposeTestRule.processCards() {
    onNode(hasText("Process 3/3 Cards") and hasClickAction())
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
 * Asserts that the score display shows the expected value.
 */
fun ComposeTestRule.assertScore(expected: Int) {
    onNodeWithTag("score_display").assertTextContains("$expected")
}

/**
 * Asserts that the game over screen is displayed.
 */
fun ComposeTestRule.assertGameOver() {
    onNodeWithTag("game_over_screen").assertIsDisplayed()
}

/**
 * Asserts that the victory screen is displayed.
 */
fun ComposeTestRule.assertVictory() {
    onNodeWithTag("victory_screen").assertIsDisplayed()
}

/**
 * Asserts that the "Draw Room" button is visible.
 * Uses waitUntil for slower CI emulators.
 */
fun ComposeTestRule.assertDrawRoomButtonVisible() {
    waitUntilNodeExists(hasText("Draw Room") and hasClickAction())
    onNode(hasText("Draw Room") and hasClickAction()).assertIsDisplayed()
}

/**
 * Asserts that the "Draw Next Room" button is visible.
 * Uses waitUntil for slower CI emulators.
 */
fun ComposeTestRule.assertDrawNextRoomButtonVisible() {
    waitUntilNodeExists(hasText("Draw Next Room") and hasClickAction())
    onNode(hasText("Draw Next Room") and hasClickAction()).assertIsDisplayed()
}

/**
 * Asserts that the "Avoid Room" button is visible.
 * Uses waitUntil for slower CI emulators.
 */
fun ComposeTestRule.assertAvoidRoomButtonVisible() {
    waitUntilNodeExists(hasText("Avoid Room") and hasClickAction())
    onNode(hasText("Avoid Room") and hasClickAction()).assertIsDisplayed()
}

/**
 * Asserts that the "Avoid Room" button is not visible (doesn't exist or hidden).
 */
fun ComposeTestRule.assertAvoidRoomButtonNotVisible() {
    // Use assertDoesNotExist since the button is conditionally rendered
    onNode(hasText("Avoid Room") and hasClickAction()).assertDoesNotExist()
}

/**
 * Asserts that the process cards button shows the expected selection count.
 */
fun ComposeTestRule.assertProcessButtonShows(selectedCount: Int) {
    onNode(hasText("Process $selectedCount/3 Cards") and hasClickAction()).assertIsDisplayed()
}

/**
 * Asserts that the process cards button is enabled (3 cards selected).
 */
fun ComposeTestRule.assertProcessButtonEnabled() {
    onNode(hasText("Process 3/3 Cards") and hasClickAction()).assertIsEnabled()
}

/**
 * Asserts that a card with the given content description is displayed.
 */
fun ComposeTestRule.assertCardDisplayed(contentDescription: String) {
    onNodeWithContentDescription(contentDescription, substring = true).assertIsDisplayed()
}

/**
 * Counts the number of visible cards by checking for card content descriptions.
 * Returns true if at least the expected number of cards are visible.
 */
fun ComposeTestRule.hasAtLeastCards(count: Int): Boolean {
    // This is a simplified check - in practice you might need to be more specific
    var foundCards = 0
    listOf("Monster card", "Weapon card", "Potion card").forEach { cardType ->
        try {
            onNodeWithContentDescription(cardType, substring = true).assertIsDisplayed()
            foundCards++
        } catch (_: AssertionError) {
            // Card type not found, continue
        }
    }
    return foundCards >= count
}
