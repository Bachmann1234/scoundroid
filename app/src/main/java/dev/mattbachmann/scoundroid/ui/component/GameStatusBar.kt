package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.data.model.WeaponState
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

/**
 * Displays current game status: health, score, deck size, and weapon info.
 */
@Composable
fun GameStatusBar(
    health: Int,
    score: Int,
    deckSize: Int,
    weaponState: WeaponState?,
    defeatedMonstersCount: Int,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
            ),
    ) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            // Health and Score
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                StatusItem(label = "Health", value = "$health / 20")
                StatusItem(label = "Score", value = "$score")
            }

            // Deck and Defeated Monsters
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
            ) {
                StatusItem(label = "Deck", value = "$deckSize cards")
                StatusItem(label = "Defeated", value = "$defeatedMonstersCount")
            }

            // Weapon status
            if (weaponState != null) {
                WeaponStatus(weaponState = weaponState)
            } else {
                Text(
                    text = "No weapon equipped",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onPrimaryContainer,
                )
            }
        }
    }
}

@Composable
private fun StatusItem(
    label: String,
    value: String,
) {
    Column(
        horizontalAlignment = Alignment.Start,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
        )
        Text(
            text = value,
            style = MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onPrimaryContainer,
        )
    }
}

@Composable
private fun WeaponStatus(weaponState: WeaponState) {
    val weaponInfo =
        if (weaponState.maxMonsterValue != null) {
            "${weaponState.weapon.suit.symbol}${weaponState.weapon.rank.displayName} " +
                "(value: ${weaponState.weapon.value}, max monster: ${weaponState.maxMonsterValue})"
        } else {
            "${weaponState.weapon.suit.symbol}${weaponState.weapon.rank.displayName} " +
                "(value: ${weaponState.weapon.value}, fresh)"
        }

    Column {
        Text(
            text = "Weapon:",
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
        )
        Text(
            text = weaponInfo,
            style = MaterialTheme.typography.bodyMedium,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.onPrimaryContainer,
        )
    }
}

@Preview(showBackground = true)
@Composable
fun GameStatusBarPreview() {
    ScoundroidTheme {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            // No weapon
            GameStatusBar(
                health = 15,
                score = 15,
                deckSize = 30,
                weaponState = null,
                defeatedMonstersCount = 5,
            )

            // With fresh weapon
            GameStatusBar(
                health = 18,
                score = 18,
                deckSize = 25,
                weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.SEVEN)),
                defeatedMonstersCount = 3,
            )

            // With degraded weapon
            GameStatusBar(
                health = 12,
                score = 12,
                deckSize = 20,
                weaponState =
                    WeaponState(
                        weapon = Card(Suit.DIAMONDS, Rank.FIVE),
                        maxMonsterValue = 8,
                    ),
                defeatedMonstersCount = 8,
            )
        }
    }
}
