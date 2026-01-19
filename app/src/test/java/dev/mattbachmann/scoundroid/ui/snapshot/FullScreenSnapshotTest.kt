package dev.mattbachmann.scoundroid.ui.snapshot

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Help
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import app.cash.paparazzi.Paparazzi
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.data.model.WeaponState
import dev.mattbachmann.scoundroid.ui.component.GameStatusBar
import dev.mattbachmann.scoundroid.ui.component.PreviewPanel
import dev.mattbachmann.scoundroid.ui.component.RoomDisplay
import dev.mattbachmann.scoundroid.ui.component.StatusBarLayout
import dev.mattbachmann.scoundroid.ui.screen.game.ScreenSizeClass
import dev.mattbachmann.scoundroid.ui.theme.ButtonPrimary
import dev.mattbachmann.scoundroid.ui.theme.GradientBottom
import dev.mattbachmann.scoundroid.ui.theme.GradientTop
import dev.mattbachmann.scoundroid.ui.theme.Purple80
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import org.junit.Rule
import org.junit.Test

private val testRoom =
    listOf(
        Card(Suit.CLUBS, Rank.QUEEN),
        Card(Suit.DIAMONDS, Rank.FIVE),
        Card(Suit.HEARTS, Rank.SEVEN),
        Card(Suit.SPADES, Rank.TEN),
    )

/**
 * Full screen layout snapshots for Galaxy A01 (COMPACT).
 */
class FullScreenCompactSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.GALAXY_A01,
        )

    @Test
    fun fullScreen_initialState() {
        paparazzi.snapshot {
            ScoundroidTheme {
                CompactFullScreenLayout(
                    health = 20,
                    score = 20,
                    deckSize = 44,
                    weaponState = null,
                    defeatedMonstersCount = 0,
                    currentRoom = null,
                    selectedCards = emptyList(),
                )
            }
        }
    }

    @Test
    fun fullScreen_roomDrawn() {
        paparazzi.snapshot {
            ScoundroidTheme {
                CompactFullScreenLayout(
                    health = 20,
                    score = 20,
                    deckSize = 40,
                    weaponState = null,
                    defeatedMonstersCount = 0,
                    currentRoom = testRoom,
                    selectedCards = emptyList(),
                )
            }
        }
    }

    @Test
    fun fullScreen_cardsSelected() {
        paparazzi.snapshot {
            ScoundroidTheme {
                CompactFullScreenLayout(
                    health = 15,
                    score = 15,
                    deckSize = 36,
                    weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.FIVE)),
                    defeatedMonstersCount = 2,
                    currentRoom = testRoom,
                    selectedCards = listOf(testRoom[0], testRoom[2], testRoom[3]),
                )
            }
        }
    }

    @Test
    fun fullScreen_midGame() {
        paparazzi.snapshot {
            ScoundroidTheme {
                CompactFullScreenLayout(
                    health = 8,
                    score = 8,
                    deckSize = 20,
                    weaponState =
                        WeaponState(
                            weapon = Card(Suit.DIAMONDS, Rank.SEVEN),
                            maxMonsterValue = 10,
                        ),
                    defeatedMonstersCount = 8,
                    currentRoom = testRoom,
                    selectedCards = listOf(testRoom[1], testRoom[2], testRoom[3]),
                )
            }
        }
    }
}

/**
 * Full screen layout snapshots for Pixel 7 (MEDIUM).
 */
class FullScreenMediumSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PIXEL_7,
        )

    @Test
    fun fullScreen_initialState() {
        paparazzi.snapshot {
            ScoundroidTheme {
                MediumFullScreenLayout(
                    health = 20,
                    score = 20,
                    deckSize = 44,
                    weaponState = null,
                    defeatedMonstersCount = 0,
                    currentRoom = null,
                    selectedCards = emptyList(),
                )
            }
        }
    }

    @Test
    fun fullScreen_roomDrawn() {
        paparazzi.snapshot {
            ScoundroidTheme {
                MediumFullScreenLayout(
                    health = 20,
                    score = 20,
                    deckSize = 40,
                    weaponState = null,
                    defeatedMonstersCount = 0,
                    currentRoom = testRoom,
                    selectedCards = emptyList(),
                )
            }
        }
    }

    @Test
    fun fullScreen_cardsSelected() {
        paparazzi.snapshot {
            ScoundroidTheme {
                MediumFullScreenLayout(
                    health = 15,
                    score = 15,
                    deckSize = 36,
                    weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.FIVE)),
                    defeatedMonstersCount = 2,
                    currentRoom = testRoom,
                    selectedCards = listOf(testRoom[0], testRoom[2], testRoom[3]),
                )
            }
        }
    }
}

/**
 * Compact full screen layout - matches GameScreen portrait layout for small phones.
 */
@Composable
private fun CompactFullScreenLayout(
    health: Int,
    score: Int,
    deckSize: Int,
    weaponState: WeaponState?,
    defeatedMonstersCount: Int,
    currentRoom: List<Card>?,
    selectedCards: List<Card>,
) {
    val backgroundGradient =
        Brush.verticalGradient(
            colors = listOf(GradientTop, GradientBottom),
        )

    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .background(backgroundGradient)
                .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // Title row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Scoundroid",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = Purple80,
            )
            Row {
                IconButton(onClick = {}) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.List,
                        contentDescription = "Action Log",
                        tint = Purple80,
                    )
                }
                IconButton(onClick = {}) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Help,
                        contentDescription = "Help",
                        tint = Purple80,
                    )
                }
            }
        }

        // Status bar
        GameStatusBar(
            health = health,
            score = score,
            deckSize = deckSize,
            weaponState = weaponState,
            defeatedMonstersCount = defeatedMonstersCount,
            layout = StatusBarLayout.COMPACT,
        )

        // Room display
        RoomDisplay(
            cards = currentRoom ?: emptyList(),
            selectedCards = selectedCards,
            onCardClick = {},
            screenSizeClass = ScreenSizeClass.COMPACT,
            showPlaceholders = currentRoom == null,
        )

        // Preview panel
        PreviewPanel(
            previewEntries = emptyList(),
            placeholderText =
                if (currentRoom == null) "Draw a room to see preview" else "Select cards to see preview",
            isCompact = true,
        )

        // Action buttons
        if (currentRoom == null) {
            Button(
                onClick = {},
                modifier = Modifier.fillMaxWidth(),
                colors =
                    ButtonDefaults.buttonColors(
                        containerColor = ButtonPrimary,
                        contentColor = Color.White,
                    ),
            ) {
                Text("Draw Room", style = MaterialTheme.typography.titleMedium)
            }
            OutlinedButton(
                onClick = {},
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80),
            ) {
                Text("Custom Seed", style = MaterialTheme.typography.titleMedium)
            }
        } else {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(
                    onClick = {},
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Avoid Room", style = MaterialTheme.typography.titleMedium)
                }
                Button(
                    onClick = {},
                    modifier = Modifier.weight(1f),
                    enabled = selectedCards.size == 3,
                    colors =
                        ButtonDefaults.buttonColors(
                            containerColor = ButtonPrimary,
                            contentColor = Color.White,
                        ),
                ) {
                    Text(
                        text = if (selectedCards.size == 3) "Go" else "Pick ${3 - selectedCards.size}",
                        style = MaterialTheme.typography.titleMedium,
                    )
                }
            }
            OutlinedButton(
                onClick = {},
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80),
            ) {
                Text("New Game", style = MaterialTheme.typography.titleMedium)
            }
        }
    }
}

/**
 * Medium full screen layout - matches GameScreen portrait layout for regular phones.
 */
@Composable
private fun MediumFullScreenLayout(
    health: Int,
    score: Int,
    deckSize: Int,
    weaponState: WeaponState?,
    defeatedMonstersCount: Int,
    currentRoom: List<Card>?,
    selectedCards: List<Card>,
) {
    val backgroundGradient =
        Brush.verticalGradient(
            colors = listOf(GradientTop, GradientBottom),
        )

    Column(
        modifier =
            Modifier
                .fillMaxSize()
                .background(backgroundGradient)
                .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        // Title row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(
                text = "Scoundroid",
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
                color = Purple80,
            )
            Row {
                IconButton(onClick = {}) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.List,
                        contentDescription = "Action Log",
                        tint = Purple80,
                    )
                }
                IconButton(onClick = {}) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.Help,
                        contentDescription = "Help",
                        tint = Purple80,
                    )
                }
            }
        }

        // Status bar
        GameStatusBar(
            health = health,
            score = score,
            deckSize = deckSize,
            weaponState = weaponState,
            defeatedMonstersCount = defeatedMonstersCount,
            layout = StatusBarLayout.MEDIUM,
        )

        // Room display
        RoomDisplay(
            cards = currentRoom ?: emptyList(),
            selectedCards = selectedCards,
            onCardClick = {},
            screenSizeClass = ScreenSizeClass.MEDIUM,
            showPlaceholders = currentRoom == null,
        )

        // Preview panel
        PreviewPanel(
            previewEntries = emptyList(),
            placeholderText =
                if (currentRoom == null) "Draw a room to see preview" else "Select cards to see preview",
            isCompact = true,
        )

        // Action buttons
        if (currentRoom == null) {
            Button(
                onClick = {},
                modifier = Modifier.fillMaxWidth(),
                colors =
                    ButtonDefaults.buttonColors(
                        containerColor = ButtonPrimary,
                        contentColor = Color.White,
                    ),
            ) {
                Text("Draw Room", style = MaterialTheme.typography.titleLarge)
            }
            OutlinedButton(
                onClick = {},
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80),
            ) {
                Text("Custom Seed", style = MaterialTheme.typography.titleLarge)
            }
        } else {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                OutlinedButton(
                    onClick = {},
                    modifier = Modifier.weight(1f),
                ) {
                    Text("Avoid Room", style = MaterialTheme.typography.titleLarge)
                }
                Button(
                    onClick = {},
                    modifier = Modifier.weight(1f),
                    enabled = selectedCards.size == 3,
                    colors =
                        ButtonDefaults.buttonColors(
                            containerColor = ButtonPrimary,
                            contentColor = Color.White,
                        ),
                ) {
                    Text(
                        text = if (selectedCards.size == 3) "Go" else "Pick ${3 - selectedCards.size}",
                        style = MaterialTheme.typography.titleLarge,
                    )
                }
            }
            OutlinedButton(
                onClick = {},
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80),
            ) {
                Text("New Game", style = MaterialTheme.typography.titleLarge)
            }
        }
    }
}
