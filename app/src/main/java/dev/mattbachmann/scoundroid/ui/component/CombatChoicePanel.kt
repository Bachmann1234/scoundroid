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
import dev.mattbachmann.scoundroid.ui.screen.game.ScreenSizeClass
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
    screenSizeClass: ScreenSizeClass = ScreenSizeClass.MEDIUM,
    showButtons: Boolean = true,
) {
    val isTablet = screenSizeClass == ScreenSizeClass.TABLET || screenSizeClass == ScreenSizeClass.TABLET_PORTRAIT
    val isLandscape = screenSizeClass == ScreenSizeClass.LANDSCAPE

    // Card sizing based on screen class
    val cardWidth =
        when {
            isTablet -> 120.dp
            isLandscape -> 100.dp
            else -> 90.dp
        }
    val cardHeight =
        when {
            isTablet -> 168.dp
            isLandscape -> 140.dp
            else -> 126.dp
        }
    val padding =
        when {
            isTablet -> 24.dp
            isLandscape -> 20.dp
            else -> 16.dp
        }
    val cardSpacing =
        when {
            isTablet -> 20.dp
            isLandscape -> 16.dp
            else -> 12.dp
        }
    val labelSpacing =
        when {
            isTablet -> 8.dp
            isLandscape -> 6.dp
            else -> 4.dp
        }
    val buttonSpacing =
        when {
            isTablet -> 24.dp
            isLandscape -> 20.dp
            else -> 16.dp
        }

    // Typography styles
    val titleStyle =
        when {
            isTablet -> MaterialTheme.typography.headlineMedium
            isLandscape -> MaterialTheme.typography.headlineSmall
            else -> MaterialTheme.typography.titleLarge
        }
    val labelStyle =
        when {
            isTablet -> MaterialTheme.typography.titleMedium
            isLandscape -> MaterialTheme.typography.titleSmall
            else -> MaterialTheme.typography.labelMedium
        }
    val vsStyle =
        when {
            isTablet -> MaterialTheme.typography.headlineSmall
            isLandscape -> MaterialTheme.typography.titleLarge
            else -> MaterialTheme.typography.titleMedium
        }
    val buttonTitleStyle =
        when {
            isTablet -> MaterialTheme.typography.titleMedium
            isLandscape -> MaterialTheme.typography.titleSmall
            else -> MaterialTheme.typography.bodyMedium
        }
    val buttonBodyStyle =
        when {
            isTablet -> MaterialTheme.typography.bodyMedium
            isLandscape -> MaterialTheme.typography.bodySmall
            else -> MaterialTheme.typography.bodySmall
        }

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
                    .padding(padding),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "Combat Choice",
                style = titleStyle,
                fontWeight = FontWeight.Bold,
            )

            Spacer(modifier = Modifier.height(cardSpacing))

            // Show the cards
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Monster",
                        style = labelStyle,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.height(labelSpacing))
                    CardView(
                        card = choice.monster,
                        cardWidth = cardWidth,
                        cardHeight = cardHeight,
                    )
                }

                Text(
                    text = "VS",
                    style = vsStyle,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )

                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Text(
                        text = "Weapon",
                        style = labelStyle,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    Spacer(modifier = Modifier.height(labelSpacing))
                    CardView(
                        card = choice.weapon,
                        cardWidth = cardWidth,
                        cardHeight = cardHeight,
                    )
                }
            }

            if (showButtons) {
                Spacer(modifier = Modifier.height(buttonSpacing))

                // Choice buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(cardSpacing),
                ) {
                    // Use Weapon button
                    Button(
                        onClick = onUseWeapon,
                        modifier = Modifier.weight(1f),
                        colors =
                            ButtonDefaults.buttonColors(
                                // Blue for weapon
                                containerColor = Color(0xFF1976D2),
                                contentColor = Color.White,
                            ),
                    ) {
                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                            Text(
                                text = "Use Weapon",
                                style = buttonTitleStyle,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                text =
                                    if (choice.weaponDamage == 0) {
                                        "No damage!"
                                    } else {
                                        "Take ${choice.weaponDamage} damage"
                                    },
                                style = buttonBodyStyle,
                            )
                            Text(
                                text = "Degrades to ${choice.weaponDegradedTo}",
                                style = buttonBodyStyle,
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
                                style = buttonTitleStyle,
                                fontWeight = FontWeight.Bold,
                            )
                            Text(
                                text = "Take ${choice.barehandedDamage} damage",
                                style = buttonBodyStyle,
                            )
                            Text(
                                text = "Keeps weapon",
                                style = buttonBodyStyle,
                            )
                        }
                    }
                }
            }
        }
    }
}

private val previewCombatChoice =
    PendingCombatChoice(
        monster = GameCard(Suit.SPADES, Rank.TEN),
        weapon = GameCard(Suit.DIAMONDS, Rank.FIVE),
        weaponDamage = 5,
        barehandedDamage = 10,
        weaponDegradedTo = 10,
        remainingCards = listOf(GameCard(Suit.HEARTS, Rank.THREE)),
    )

@Preview(
    showBackground = true,
    name = "Small Phone (COMPACT)",
    device = "spec:width=360dp,height=640dp,dpi=420",
)
@Composable
fun CombatChoicePanelCompactPreview() {
    ScoundroidTheme {
        CombatChoicePanel(
            choice = previewCombatChoice,
            onUseWeapon = {},
            onFightBarehanded = {},
            modifier = Modifier.padding(16.dp),
            screenSizeClass = ScreenSizeClass.COMPACT,
        )
    }
}

@Preview(
    showBackground = true,
    name = "Regular Phone (MEDIUM)",
    device = "spec:width=411dp,height=891dp,dpi=420",
)
@Composable
fun CombatChoicePanelMediumPreview() {
    ScoundroidTheme {
        CombatChoicePanel(
            choice = previewCombatChoice,
            onUseWeapon = {},
            onFightBarehanded = {},
            modifier = Modifier.padding(16.dp),
            screenSizeClass = ScreenSizeClass.MEDIUM,
        )
    }
}

@Preview(
    showBackground = true,
    name = "Tablet/Unfolded (TABLET)",
    device = "spec:width=600dp,height=900dp,dpi=420",
)
@Composable
fun CombatChoicePanelTabletPreview() {
    ScoundroidTheme {
        CombatChoicePanel(
            choice = previewCombatChoice,
            onUseWeapon = {},
            onFightBarehanded = {},
            modifier = Modifier.padding(16.dp),
            screenSizeClass = ScreenSizeClass.TABLET,
        )
    }
}
