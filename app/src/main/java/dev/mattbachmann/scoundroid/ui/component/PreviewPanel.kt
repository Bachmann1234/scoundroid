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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Favorite
import androidx.compose.material.icons.filled.Shield
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
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

private val PREVIEW_PANEL_HEIGHT_COMPACT = 95.dp

/**
 * Displays a preview of what will happen when processing the selected cards.
 * Shows log entries in processing order (not reversed like action log).
 */
@Composable
fun PreviewPanel(
    previewEntries: List<LogEntry>,
    modifier: Modifier = Modifier,
    placeholderText: String = "Select cards to see preview",
    isCompact: Boolean = false,
) {
    // Only use fixed height in compact mode to prevent layout jumping
    // In expanded mode, let content determine height
    val panelModifier = if (isCompact) {
        modifier.fillMaxWidth().height(PREVIEW_PANEL_HEIGHT_COMPACT)
    } else {
        modifier.fillMaxWidth()
    }
    Card(
        modifier = panelModifier,
        colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.5f),
            ),
    ) {
        Column(
            modifier =
                Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState())
                    .padding(if (isCompact) 8.dp else 12.dp),
            verticalArrangement = Arrangement.spacedBy(if (isCompact) 4.dp else 8.dp),
        ) {
            Text(
                text = "Preview",
                style = if (isCompact) MaterialTheme.typography.titleSmall else MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary,
            )

            if (previewEntries.isEmpty()) {
                Text(
                    text = placeholderText,
                    style = if (isCompact) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            } else {
                // Show entries in processing order (not reversed)
                previewEntries.forEach { entry ->
                    PreviewEntryRow(entry = entry, isCompact = isCompact)
                }
            }
        }
    }
}

@Composable
private fun PreviewEntryRow(entry: LogEntry, isCompact: Boolean = false) {
    val (icon, iconColor, description) =
        when (entry) {
            is LogEntry.MonsterFought -> {
                val weaponInfo =
                    if (entry.weaponUsed != null) {
                        "with ${entry.weaponUsed.displayName} - blocked ${entry.damageBlocked}"
                    } else {
                        "barehanded"
                    }
                val healthChange = "${entry.healthBefore} -> ${entry.healthAfter} HP"
                // Dark red for monsters
                PreviewEntryDisplay(
                    icon = Icons.Default.Shield,
                    iconColor = Color(0xFFB71C1C),
                    description =
                        "Fight ${entry.monster.displayName} $weaponInfo, " +
                            "take ${entry.damageTaken} damage ($healthChange)",
                )
            }
            is LogEntry.WeaponEquipped -> {
                val replaceInfo =
                    if (entry.replacedWeapon != null) {
                        " (replace ${entry.replacedWeapon.displayName})"
                    } else {
                        ""
                    }
                // Blue for weapons
                PreviewEntryDisplay(
                    icon = Icons.Default.Shield,
                    iconColor = Color(0xFF1565C0),
                    description = "Equip ${entry.weapon.displayName}$replaceInfo",
                )
            }
            is LogEntry.PotionUsed -> {
                if (entry.wasDiscarded) {
                    PreviewEntryDisplay(
                        icon = Icons.Default.Favorite,
                        iconColor = MaterialTheme.colorScheme.error,
                        description = "Discard ${entry.potion.displayName} (already used potion this turn)",
                    )
                } else {
                    val healthChange = "${entry.healthBefore} -> ${entry.healthAfter} HP"
                    // Green for healing
                    PreviewEntryDisplay(
                        icon = Icons.Default.Favorite,
                        iconColor = Color(0xFF2E7D32),
                        description =
                            "Drink ${entry.potion.displayName}, " +
                                "restore ${entry.healthRestored} HP ($healthChange)",
                    )
                }
            }
            // These entry types won't appear in preview
            is LogEntry.GameStarted,
            is LogEntry.RoomDrawn,
            is LogEntry.RoomAvoided,
            ->
                PreviewEntryDisplay(
                    icon = Icons.Default.Shield,
                    iconColor = MaterialTheme.colorScheme.onSurface,
                    description = "",
                )
        }

    if (description.isNotEmpty()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Icon(
                imageVector = icon,
                contentDescription = null,
                tint = iconColor,
                modifier = Modifier.size(18.dp),
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = description,
                style = if (isCompact) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurface,
            )
        }
    }
}

private data class PreviewEntryDisplay(
    val icon: ImageVector,
    val iconColor: Color,
    val description: String,
)
