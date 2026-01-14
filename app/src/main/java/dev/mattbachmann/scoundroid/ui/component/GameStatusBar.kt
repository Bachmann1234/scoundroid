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
import androidx.compose.ui.platform.testTag
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
    /** Single row with abbreviated labels for very small phones */
    COMPACT,

    /** 2-row grid for medium screens (fold cover, regular phones) */
    MEDIUM,

    /** Horizontal inline for expanded screens (tablets, unfolded) */
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
            infiniteTransition
                .animateFloat(
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
            modifier = Modifier.padding(if (isCompact) 12.dp else 16.dp),
            verticalArrangement = Arrangement.spacedBy(if (isCompact) 6.dp else 12.dp),
        ) {
            when (layout) {
                StatusBarLayout.COMPACT -> {
                    // Compact mode: single row with all 4 stats (abbreviated labels)
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        StatusItem(
                            label = "HP",
                            value = "$health/20",
                            isCompact = true,
                            valueColor = healthTextColor,
                            valueTestTag = "health_display",
                        )
                        StatusItem(
                            label = "Score",
                            value = "$score",
                            isCompact = true,
                            valueTestTag = "score_display",
                        )
                        StatusItem(
                            label = "Deck",
                            value = "$deckSize",
                            isCompact = true,
                            valueTestTag = "deck_size_display",
                        )
                        StatusItem(
                            label = "Kills",
                            value = "$defeatedMonstersCount",
                            isCompact = true,
                            horizontalAlignment = Alignment.End,
                            valueTestTag = "defeated_display",
                        )
                    }
                }
                StatusBarLayout.MEDIUM -> {
                    // Medium mode: 2-row grid with full labels
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        StatusItem(
                            label = "Health",
                            value = "$health / 20",
                            isCompact = true,
                            valueColor = healthTextColor,
                            valueTestTag = "health_display",
                        )
                        StatusItem(
                            label = "Score",
                            value = "$score",
                            isCompact = true,
                            horizontalAlignment = Alignment.End,
                            valueTestTag = "score_display",
                        )
                    }
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceBetween,
                    ) {
                        StatusItem(
                            label = "Deck",
                            value = "$deckSize cards",
                            isCompact = true,
                            valueTestTag = "deck_size_display",
                        )
                        StatusItem(
                            label = "Defeated",
                            value = "$defeatedMonstersCount",
                            isCompact = true,
                            horizontalAlignment = Alignment.End,
                            valueTestTag = "defeated_display",
                        )
                    }
                }
                StatusBarLayout.INLINE -> {
                    // Horizontal inline for bottom panel - all 4 items in single row
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                    ) {
                        StatusItem(
                            label = "Health",
                            value = "$health / 20",
                            valueColor = healthTextColor,
                            valueTestTag = "health_display",
                        )
                        StatusItem(
                            label = "Score",
                            value = "$score",
                            valueTestTag = "score_display",
                        )
                        StatusItem(
                            label = "Deck",
                            value = "$deckSize cards",
                            valueTestTag = "deck_size_display",
                        )
                        StatusItem(
                            label = "Defeated",
                            value = "$defeatedMonstersCount",
                            valueTestTag = "defeated_display",
                        )
                    }
                }
            }

            // Weapon status - use consistent Column structure to prevent layout jumping
            Column {
                Text(
                    text = "WEAPON",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f),
                    fontWeight = FontWeight.Medium,
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
                        modifier = Modifier.testTag("weapon_display"),
                    )
                } else {
                    val textStyle =
                        if (isCompact) MaterialTheme.typography.bodySmall else MaterialTheme.typography.bodyMedium
                    Text(
                        text = "None",
                        style = textStyle,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.onPrimaryContainer,
                        modifier = Modifier.testTag("weapon_display"),
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
    valueTestTag: String? = null,
) {
    Column(
        horizontalAlignment = horizontalAlignment,
    ) {
        Text(
            text = label.uppercase(),
            style = MaterialTheme.typography.labelSmall,
            color = MaterialTheme.colorScheme.onPrimaryContainer.copy(alpha = 0.8f),
            fontWeight = FontWeight.Medium,
        )
        Text(
            text = value,
            style = if (isCompact) MaterialTheme.typography.titleMedium else MaterialTheme.typography.titleLarge,
            fontWeight = FontWeight.Bold,
            color = valueColor ?: MaterialTheme.colorScheme.onPrimaryContainer,
            modifier = if (valueTestTag != null) Modifier.testTag(valueTestTag) else Modifier,
        )
    }
}

@Preview(
    showBackground = true,
    name = "Small Phone (COMPACT)",
    device = "spec:width=360dp,height=640dp,dpi=420",
)
@Composable
fun GameStatusBarCompactPreview() {
    ScoundroidTheme {
        GameStatusBar(
            health = 15,
            score = 15,
            deckSize = 30,
            weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.SEVEN)),
            defeatedMonstersCount = 5,
            layout = StatusBarLayout.COMPACT,
        )
    }
}

@Preview(
    showBackground = true,
    name = "Regular Phone (MEDIUM)",
    device = "spec:width=411dp,height=891dp,dpi=420",
)
@Composable
fun GameStatusBarMediumPreview() {
    ScoundroidTheme {
        GameStatusBar(
            health = 15,
            score = 15,
            deckSize = 30,
            weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.SEVEN)),
            defeatedMonstersCount = 5,
            layout = StatusBarLayout.MEDIUM,
        )
    }
}

@Preview(
    showBackground = true,
    name = "Tablet/Unfolded (INLINE)",
    device = "spec:width=600dp,height=900dp,dpi=420",
)
@Composable
fun GameStatusBarInlinePreview() {
    ScoundroidTheme {
        GameStatusBar(
            health = 15,
            score = 15,
            deckSize = 30,
            weaponState =
                WeaponState(
                    weapon = Card(Suit.DIAMONDS, Rank.FIVE),
                    maxMonsterValue = 8,
                ),
            defeatedMonstersCount = 5,
            layout = StatusBarLayout.INLINE,
        )
    }
}
