package dev.mattbachmann.scoundroid.ui.component

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
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
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.data.model.WeaponState
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import kotlinx.coroutines.delay

/**
 * Health flash animation states.
 */
private enum class HealthFlashState {
    NONE,
    DAMAGE,
    HEALING,
}

/**
 * Layout modes for the status bar.
 */
enum class StatusBarLayout {
    /** Compact 2-row grid for narrow screens */
    COMPACT,

    /** Horizontal inline for bottom panel */
    INLINE,
}

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
    layout: StatusBarLayout = StatusBarLayout.COMPACT,
) {
    val isCompact = layout == StatusBarLayout.COMPACT

    // Health change animation state
    var previousHealth by remember { mutableIntStateOf(health) }
    var healthFlashState by remember { mutableStateOf(HealthFlashState.NONE) }

    // Detect health changes and trigger flash
    LaunchedEffect(health) {
        if (health < previousHealth) {
            healthFlashState = HealthFlashState.DAMAGE
            delay(400)
            healthFlashState = HealthFlashState.NONE
        } else if (health > previousHealth) {
            healthFlashState = HealthFlashState.HEALING
            delay(400)
            healthFlashState = HealthFlashState.NONE
        }
        previousHealth = health
    }

    // Low health pulse animation - only run when health is low to avoid unnecessary computation
    val isLowHealth = health <= 5
    val pulseAlpha =
        if (isLowHealth) {
            val infiniteTransition = rememberInfiniteTransition(label = "lowHealthPulse")
            infiniteTransition.animateFloat(
                initialValue = 1f,
                targetValue = 0.3f,
                animationSpec =
                    infiniteRepeatable(
                        animation = tween(500),
                        repeatMode = RepeatMode.Reverse,
                    ),
                label = "pulseAlpha",
            ).value
        } else {
            1f
        }

    // Animated health color - incorporates flash and low health pulse
    val baseColor =
        when (healthFlashState) {
            HealthFlashState.DAMAGE -> Color(0xFFFF1744) // Bright red
            HealthFlashState.HEALING -> Color(0xFF00E676) // Bright green
            HealthFlashState.NONE ->
                if (isLowHealth) Color(0xFFFF1744) else MaterialTheme.colorScheme.onPrimaryContainer
        }
    val healthTextColor by animateColorAsState(
        targetValue =
            if (isLowHealth && healthFlashState == HealthFlashState.NONE) {
                baseColor.copy(alpha = pulseAlpha)
            } else {
                baseColor
            },
        animationSpec = tween(durationMillis = 200),
        label = "healthColor",
    )

    Card(
        modifier = modifier.fillMaxWidth(),
        colors =
            CardDefaults.cardColors(
                containerColor = MaterialTheme.colorScheme.primaryContainer,
            ),
    ) {
        Column(
            modifier = Modifier.padding(if (isCompact) 10.dp else 16.dp),
            verticalArrangement = Arrangement.spacedBy(if (isCompact) 6.dp else 12.dp),
        ) {
            when (layout) {
                StatusBarLayout.COMPACT -> {
                    // Compact mode: 2-row grid
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        StatusItem(
                            label = "Health",
                            value = "$health / 20",
                            isCompact = true,
                            valueColor = healthTextColor,
                        )
                        StatusItem(
                            label = "Score",
                            value = "$score",
                            isCompact = true,
                            horizontalAlignment = Alignment.End,
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        StatusItem(label = "Deck", value = "$deckSize cards", isCompact = true)
                        StatusItem(
                            label = "Defeated",
                            value = "$defeatedMonstersCount",
                            isCompact = true,
                            horizontalAlignment = Alignment.End,
                        )
                    }
                }
                StatusBarLayout.INLINE -> {
                    // Horizontal inline for bottom panel - all 4 items in single row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                    ) {
                        StatusItem(label = "Health", value = "$health / 20", valueColor = healthTextColor)
                        StatusItem(label = "Score", value = "$score")
                        StatusItem(label = "Deck", value = "$deckSize cards")
                        StatusItem(label = "Defeated", value = "$defeatedMonstersCount")
                    }
                }
            }

            // Weapon status - use consistent Column structure to prevent layout jumping
            Column {
                Text(
                    text = "Weapon:",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
                )
                if (weaponState != null) {
                    val weaponInfo =
                        if (weaponState.maxMonsterValue != null) {
                            "${weaponState.weapon.suit.symbol}${weaponState.weapon.rank.displayName} " +
                                "(value: ${weaponState.weapon.value}, max monster: ${weaponState.maxMonsterValue})"
                        } else {
                            "${weaponState.weapon.suit.symbol}${weaponState.weapon.rank.displayName} " +
                                "(value: ${weaponState.weapon.value}, fresh)"
                        }
                    val textStyle =
                        if (isCompact) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyMedium
                    Text(
                        text = weaponInfo,
                        style = textStyle,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                } else {
                    val textStyle =
                        if (isCompact) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyMedium
                    Text(
                        text = "None",
                        style = textStyle,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                    )
                }
            }
        }
    }
}

@Composable
private fun StatusItem(
    label: String,
    value: String,
    isCompact: Boolean = false,
    horizontalAlignment: Alignment.Horizontal = Alignment.Start,
    valueColor: Color? = null,
) {
    Column(
        horizontalAlignment = horizontalAlignment,
    ) {
        Text(
            text = label,
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.7f),
        )
        Text(
            text = value,
            style = if (isCompact) MaterialTheme.typography.titleSmall else MaterialTheme.typography.titleMedium,
            fontWeight = FontWeight.Bold,
            color = valueColor ?: MaterialTheme.colorScheme.onPrimaryContainer,
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
