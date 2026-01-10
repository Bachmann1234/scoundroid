package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.ui.screen.game.PendingCombatChoice
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import dev.mattbachmann.scoundroid.data.model.Card as GameCard

/**
 * Panel displayed when player must choose between using weapon or fighting barehanded.
 * Shows the monster, weapon, and damage consequences for each choice.
 */
@Composable
fun CombatChoicePanel(
    choice: PendingCombatChoice,
    onUseWeapon: () -> Unit,
    onFightBarehanded: () -> Unit,
    modifier: Modifier = Modifier,
) {
    Card(
        modifier = modifier.fillMaxWidth(),
        colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.surfaceVariant,
            ),
        elevation = CardDefaults.cardElevation(defaultElevation = 4.dp),
    ) {
        Column(
            modifier =
                Modifier
                    .fillMaxWidth()
                    .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "Combat Choice",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
            )

            Spacer(modifier = Modifier.height(12.dp))

            // Show the cards
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Monster",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    CardView(
                        card = choice.monster,
                        cardWidth = 90.dp,
                        cardHeight = 126.dp,
                    )
                }

                Text(
                    text = "VS",
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Weapon",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    CardView(
                        card = choice.weapon,
                        cardWidth = 90.dp,
                        cardHeight = 126.dp,
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Choice buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                // Use Weapon button
                Button(
                    onClick = onUseWeapon,
                    modifier = Modifier.weight(1f),
                    colors =
                        ButtonDefaults.buttonColors(
                            // Blue for weapon
                            containerColor = Color(0xFF1976D2),
                        ),
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Use Weapon",
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text =
                                if (choice.weaponDamage == 0) {
                                    "No damage!"
                                } else {
                                    "Take ${choice.weaponDamage} damage"
                                },
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            text = "Degrades to ${choice.weaponDegradedTo}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }

                // Fight Barehanded button
                OutlinedButton(
                    onClick = onFightBarehanded,
                    modifier = Modifier.weight(1f),
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Barehanded",
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "Take ${choice.barehandedDamage} damage",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Text(
                            text = "Weapon unchanged",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }

            // Additional context if there are more cards
            if (choice.remainingCards.isNotEmpty()) {
                Spacer(modifier = Modifier.height(8.dp))
                Text(
                    text = "${choice.remainingCards.size} more card(s) to process",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun CombatChoicePanelPreview() {
    ScoundroidTheme {
        CombatChoicePanel(
            choice =
                PendingCombatChoice(
                    monster = GameCard(Suit.SPADES, Rank.TEN),
                    weapon = GameCard(Suit.DIAMONDS, Rank.FIVE),
                    weaponDamage = 5,
                    barehandedDamage = 10,
                    weaponDegradedTo = 10,
                    remainingCards = listOf(GameCard(Suit.HEARTS, Rank.THREE)),
                ),
            onUseWeapon = {},
            onFightBarehanded = {},
            modifier = Modifier.padding(16.dp),
        )
    }
}
