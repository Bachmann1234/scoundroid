package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Layers
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.LogEntry

/**
 * Displays the game action log with all recorded events.
 * Shows entries in reverse chronological order (newest first).
 */
@Composable
fun ActionLogPanel(
    actionLog: List<LogEntry>,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier =
            modifier
                .fillMaxWidth()
                .padding(start = 16.dp, top = 16.dp, end = 16.dp, bottom = 24.dp),
    ) {
        Text(
            text = "Action Log",
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Spacer(modifier = Modifier.height(16.dp))

        if (actionLog.isEmpty()) {
            Text(
                text = "No actions recorded yet.",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        } else {
            val reversedLog = actionLog.reversed()
            LazyColumn(
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                itemsIndexed(reversedLog) { index, entry ->
                    LogEntryRow(entry = entry)
                    if (index < reversedLog.lastIndex) {
                        HorizontalDivider(
                            modifier = Modifier.padding(top = 8.dp),
                            color = MaterialTheme.colorScheme.outlineVariant,
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun LogEntryRow(entry: LogEntry) {
    val (icon, iconColor, title, description) =
        when (entry) {
            is LogEntry.GameStarted ->
                LogEntryDisplay(
                    icon = Icons.Default.PlayArrow,
                    iconColor = MaterialTheme.colorScheme.primary,
                    title = "Game Started",
                    description = "New dungeon adventure begins!",
                )
            is LogEntry.RoomDrawn ->
                LogEntryDisplay(
                    icon = Icons.Default.Layers,
                    iconColor = MaterialTheme.colorScheme.tertiary,
                    title = "Room Drawn",
                    description = "Drew ${entry.cardsDrawn} cards (${entry.deckSizeAfter} remaining)",
                )
            is LogEntry.RoomAvoided ->
                LogEntryDisplay(
                    icon = Icons.AutoMirrored.Filled.ArrowBack,
                    iconColor = MaterialTheme.colorScheme.secondary,
                    title = "Room Avoided",
                    description = "${entry.cardsReturned} cards sent to bottom of deck",
                )
            is LogEntry.MonsterFought -> {
                val weaponInfo =
                    if (entry.weaponUsed != null) {
                        "with ${entry.weaponUsed.displayName} - blocked ${entry.damageBlocked}"
                    } else {
                        "barehanded"
                    }
                val healthChange = "${entry.healthBefore} -> ${entry.healthAfter} HP"
                LogEntryDisplay(
                    icon = Icons.Default.Shield,
                    // Dark red for monsters
                    iconColor = Color(0xFFB71C1C),
                    title = "Fought ${entry.monster.displayName}",
                    description = "$weaponInfo, took ${entry.damageTaken} damage ($healthChange)",
                )
            }
            is LogEntry.WeaponEquipped -> {
                val replaceInfo =
                    if (entry.replacedWeapon != null) {
                        " (replaced ${entry.replacedWeapon.displayName})"
                    } else {
                        ""
                    }
                LogEntryDisplay(
                    icon = Icons.Default.Shield,
                    // Blue for weapons
                    iconColor = Color(0xFF1565C0),
                    title = "Equipped ${entry.weapon.displayName}",
                    description = "Weapon ready for combat$replaceInfo",
                )
            }
            is LogEntry.PotionUsed -> {
                if (entry.wasDiscarded) {
                    LogEntryDisplay(
                        icon = Icons.Default.Favorite,
                        iconColor = MaterialTheme.colorScheme.error,
                        title = "Discarded ${entry.potion.displayName}",
                        description = "Already used a potion this turn",
                    )
                } else {
                    val healthChange = "${entry.healthBefore} -> ${entry.healthAfter} HP"
                    LogEntryDisplay(
                        icon = Icons.Default.Favorite,
                        // Green for healing
                        iconColor = Color(0xFF2E7D32),
                        title = "Used ${entry.potion.displayName}",
                        description = "Restored ${entry.healthRestored} HP ($healthChange)",
                    )
                }
            }
        }

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.Top,
    ) {
        Icon(
            imageVector = icon,
            contentDescription = null,
            tint = iconColor,
            modifier = Modifier.size(24.dp),
        )
        Spacer(modifier = Modifier.width(12.dp))
        Column {
            Text(
                text = title,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.SemiBold,
                color = MaterialTheme.colorScheme.onSurface,
            )
            Text(
                text = description,
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

private data class LogEntryDisplay(
    val icon: ImageVector,
    val iconColor: Color,
    val title: String,
    val description: String,
)
