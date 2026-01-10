package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

/**
 * Displays the game rules and help content.
 * Based on Scoundrel by Zach Gage and Kurt Bieg.
 */
@Composable
fun HelpContent(modifier: Modifier = Modifier) {
    Column(
        modifier =
            modifier
                .fillMaxWidth()
                .verticalScroll(rememberScrollState())
                .padding(start = 16.dp, top = 16.dp, end = 16.dp, bottom = 24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "How to Play",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        HelpSection(
            title = "Goal",
            content =
                "Survive the dungeon by making your way through all 44 cards. " +
                    "Your score is your remaining health when you win.",
        )

        HelpSection(
            title = "Card Types",
            content =
                """
                Monsters (Clubs & Spades): Deal damage equal to their value. Face cards: J=11, Q=12, K=13, A=14.

                Weapons (Diamonds): Reduce monster damage by the weapon's value. Picking up a new weapon discards the old one.

                Potions (Hearts): Restore health by their value (max 20 health). Only one potion works per turn.
                """.trimIndent(),
        )

        HelpSection(
            title = "Rooms",
            content =
                """
                Each turn, draw 4 cards to form a Room.

                You may Avoid the room (send all 4 cards to the bottom of the deck), but you cannot avoid two rooms in a row.

                If you don't avoid, you must choose and process 3 of the 4 cards. The 4th card stays for the next room.
                """.trimIndent(),
        )

        HelpSection(
            title = "Combat",
            content =
                """
                Fight monsters barehanded (take full damage) or with your equipped weapon (reduced damage).

                Weapon degradation: Once a weapon defeats a monster, it can only be used on monsters with equal or lower value than the last one it defeated.

                Example: If your 5-weapon defeats a Queen (12), it can still fight any monster up to 12. But if it then defeats a 6, it can now only fight monsters 6 or lower.
                """.trimIndent(),
        )

        HelpSection(
            title = "Scoring",
            content =
                """
                Win: Your score is your remaining health. Special: If health is 20 and your last card was a potion, add that potion's value.

                Lose: Your score is negative (sum of all remaining monsters in the deck).
                """.trimIndent(),
        )

        HelpSection(
            title = "Playing with Real Cards",
            content =
                """
                Setup: Remove all Jokers, red face cards (J/Q/K of Hearts and Diamonds), and red Aces. You should have 44 cards.

                Track health: Use a d20 die, paper, or 20 tokens (remove as you take damage).

                Track weapon degradation: Keep the last defeated monster under your weapon as a reminder of its limit.

                Room layout: Draw 4 cards in a row. When leaving one behind, slide it left and draw 3 more for the next room.

                Skip tracking: Use a coin—heads means you can skip, tails means you must play. Flip after each room.
                """.trimIndent(),
        )

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = "Based on Scoundrel by Zach Gage & Kurt Bieg",
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

@Composable
private fun HelpSection(
    title: String,
    content: String,
) {
    Column {
        Text(
            text = title,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.SemiBold,
            color = MaterialTheme.colorScheme.secondary,
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = content,
            style = MaterialTheme.typography.bodyMedium,
            color = MaterialTheme.colorScheme.onSurface,
        )
    }
}
