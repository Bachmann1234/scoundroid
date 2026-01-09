package dev.mattbachmann.scoundroid.ui.screen.game

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Help
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.LogEntry
import dev.mattbachmann.scoundroid.ui.component.ActionLogPanel
import dev.mattbachmann.scoundroid.ui.component.GameStatusBar
import dev.mattbachmann.scoundroid.ui.component.HelpContent
import dev.mattbachmann.scoundroid.ui.component.PreviewPanel
import dev.mattbachmann.scoundroid.ui.component.RoomDisplay
import dev.mattbachmann.scoundroid.ui.component.StatusBarLayout
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

/**
 * Main game screen for Scoundrel.
 * Displays the current game state and handles user interactions.
 * Supports responsive layouts for foldable devices.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GameScreen(
    modifier: Modifier = Modifier,
    viewModel: GameViewModel = viewModel(),
    isExpandedScreen: Boolean = false,
) {
    val uiState by viewModel.uiState.collectAsState()
    var selectedCards by remember { mutableStateOf(listOf<Card>()) }
    var scoreSaved by remember { mutableStateOf(false) }
    val sheetState = rememberModalBottomSheetState()

    // Save score when game ends, reset flag when new game starts
    LaunchedEffect(uiState.isGameOver, uiState.isGameWon) {
        if ((uiState.isGameOver || uiState.isGameWon) && !scoreSaved) {
            viewModel.onIntent(GameIntent.GameEnded(uiState.score, uiState.isGameWon))
            scoreSaved = true
        } else if (!uiState.isGameOver && !uiState.isGameWon) {
            scoreSaved = false
        }
    }

    // Help bottom sheet
    if (uiState.showHelp) {
        ModalBottomSheet(
            onDismissRequest = { viewModel.onIntent(GameIntent.HideHelp) },
            sheetState = sheetState,
        ) {
            HelpContent()
        }
    }

    // Action log bottom sheet
    if (uiState.showActionLog) {
        ModalBottomSheet(
            onDismissRequest = { viewModel.onIntent(GameIntent.HideActionLog) },
            sheetState = rememberModalBottomSheetState(),
        ) {
            ActionLogPanel(actionLog = uiState.actionLog)
        }
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
    ) { innerPadding ->
        if (isExpandedScreen) {
            // Expanded layout: title at top, cards in center, controls on bottom
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .padding(16.dp),
            ) {
                // Title row at the top
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "Scoundroid",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Row {
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowActionLog) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.List,
                                contentDescription = "Action Log",
                                tint = MaterialTheme.colorScheme.primary,
                            )
                        }
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowHelp) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.Help,
                                contentDescription = "Help",
                                tint = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }
                }

                // Center section: Cards (takes available space)
                Column(
                    modifier =
                        Modifier
                            .weight(1f)
                            .fillMaxWidth(),
                    verticalArrangement = Arrangement.Center,
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    ExpandedCardsSection(
                        uiState = uiState,
                        selectedCards = selectedCards,
                        onSelectedCardsChange = { selectedCards = it },
                    )
                }

                // Bottom section: status, preview, buttons
                Column(
                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    // Status bar (inline horizontal)
                    GameStatusBar(
                        health = uiState.health,
                        score = uiState.score,
                        deckSize = uiState.deckSize,
                        weaponState = uiState.weaponState,
                        defeatedMonstersCount = uiState.defeatedMonstersCount,
                        layout = StatusBarLayout.INLINE,
                    )

                    // Controls section
                    ExpandedControlsSection(
                        uiState = uiState,
                        selectedCards = selectedCards,
                        onSelectedCardsChange = { selectedCards = it },
                        onIntent = viewModel::onIntent,
                        simulateProcessing = viewModel::simulateProcessing,
                    )
                }
            }
        } else {
            // Compact layout: vertical stack (no scroll - fits on screen)
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .padding(12.dp),
                verticalArrangement = Arrangement.spacedBy(10.dp),
            ) {
                // Title with action log and help buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "Scoundroid",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    Row {
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowActionLog) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.List,
                                contentDescription = "Action Log",
                                tint = MaterialTheme.colorScheme.primary,
                            )
                        }
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowHelp) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.Help,
                                contentDescription = "Help",
                                tint = MaterialTheme.colorScheme.primary,
                            )
                        }
                    }
                }

                // Game status
                GameStatusBar(
                    health = uiState.health,
                    score = uiState.score,
                    deckSize = uiState.deckSize,
                    weaponState = uiState.weaponState,
                    defeatedMonstersCount = uiState.defeatedMonstersCount,
                    layout = StatusBarLayout.COMPACT,
                )

                GameContent(
                    uiState = uiState,
                    selectedCards = selectedCards,
                    onSelectedCardsChange = { selectedCards = it },
                    onIntent = viewModel::onIntent,
                    simulateProcessing = viewModel::simulateProcessing,
                    isExpandedScreen = false,
                )
            }
        }
    }
}

/**
 * Handles card selection logic - toggles card selection up to max of 3.
 */
private fun toggleCardSelection(
    card: Card,
    selectedCards: List<Card>,
): List<Card> =
    if (card in selectedCards) {
        selectedCards - card
    } else if (selectedCards.size < 3) {
        selectedCards + card
    } else {
        selectedCards
    }

/**
 * Shared action buttons for room interactions.
 * Used by both compact and expanded layouts.
 */
@Composable
private fun RoomActionButtons(
    currentRoom: List<Card>?,
    selectedCards: List<Card>,
    canAvoidRoom: Boolean,
    isGameOver: Boolean,
    isGameWon: Boolean,
    onAvoidRoom: () -> Unit,
    onProcessCards: () -> Unit,
    onDrawRoom: () -> Unit,
    onNewGame: () -> Unit,
    modifier: Modifier = Modifier,
    isCompact: Boolean = false,
) {
    val buttonSpacing = if (isCompact) 4.dp else 8.dp
    val buttonTextStyle = if (isCompact) MaterialTheme.typography.titleMedium else MaterialTheme.typography.titleLarge

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(buttonSpacing),
    ) {
        if (isGameOver || isGameWon) {
            Button(
                onClick = onNewGame,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "New Game",
                    style = buttonTextStyle,
                )
            }
        } else if (currentRoom != null) {
            when (currentRoom.size) {
                4 -> {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.spacedBy(buttonSpacing),
                    ) {
                        if (canAvoidRoom) {
                            OutlinedButton(
                                onClick = onAvoidRoom,
                                modifier = Modifier.weight(1f),
                            ) {
                                Text("Avoid Room")
                            }
                        }

                        Button(
                            onClick = onProcessCards,
                            enabled = selectedCards.size == 3,
                            modifier =
                                if (canAvoidRoom) {
                                    Modifier.weight(1f)
                                } else {
                                    Modifier.fillMaxWidth()
                                },
                        ) {
                            Text("Process ${selectedCards.size}/3 Cards")
                        }
                    }

                    OutlinedButton(
                        onClick = onNewGame,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("New Game")
                    }
                }
                1 -> {
                    if (!isCompact) {
                        Text(
                            text = "This card stays for the next room",
                            style = MaterialTheme.typography.bodyMedium,
                            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
                        )
                    }
                    Button(
                        onClick = onDrawRoom,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text(
                            text = "Draw Next Room",
                            style = buttonTextStyle,
                        )
                    }

                    OutlinedButton(
                        onClick = onNewGame,
                        modifier = Modifier.fillMaxWidth(),
                    ) {
                        Text("New Game")
                    }
                }
            }
        } else {
            Button(
                onClick = onDrawRoom,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "Draw Room",
                    style = buttonTextStyle,
                )
            }

            OutlinedButton(
                onClick = onNewGame,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("New Game")
            }
        }
    }
}

/**
 * Shared room cards display.
 * Used by both compact and expanded layouts.
 */
@Composable
private fun RoomCardsDisplay(
    currentRoom: List<Card>?,
    selectedCards: List<Card>,
    isExpanded: Boolean,
    onCardClick: ((Card) -> Unit)?,
) {
    if (currentRoom != null) {
        if (currentRoom.size == 1) {
            RoomDisplay(
                cards = currentRoom,
                selectedCards = emptyList(),
                onCardClick = null,
                isExpanded = isExpanded,
            )
        } else {
            RoomDisplay(
                cards = currentRoom,
                selectedCards = selectedCards,
                onCardClick = onCardClick,
                isExpanded = isExpanded,
            )
        }
    } else {
        RoomDisplay(
            cards = emptyList(),
            selectedCards = emptyList(),
            onCardClick = null,
            isExpanded = isExpanded,
            showPlaceholders = true,
        )
    }
}

/**
 * Game content that adapts to compact or expanded layouts.
 */
@Composable
private fun GameContent(
    uiState: GameUiState,
    selectedCards: List<Card>,
    onSelectedCardsChange: (List<Card>) -> Unit,
    onIntent: (GameIntent) -> Unit,
    simulateProcessing: (List<Card>) -> List<LogEntry>,
    isExpandedScreen: Boolean,
) {
    // Game over / won message
    if (uiState.isGameOver) {
        GameOverScreen(
            score = uiState.score,
            highestScore = uiState.highestScore,
            isNewHighScore = uiState.isNewHighScore,
            onNewGame = {
                onIntent(GameIntent.NewGame)
                onSelectedCardsChange(emptyList())
            },
        )
    } else if (uiState.isGameWon) {
        GameWonScreen(
            score = uiState.score,
            highestScore = uiState.highestScore,
            isNewHighScore = uiState.isNewHighScore,
            onNewGame = {
                onIntent(GameIntent.NewGame)
                onSelectedCardsChange(emptyList())
            },
        )
    } else {
        // Active game - show room cards
        RoomCardsDisplay(
            currentRoom = uiState.currentRoom,
            selectedCards = selectedCards,
            isExpanded = isExpandedScreen,
            onCardClick = { card ->
                onSelectedCardsChange(toggleCardSelection(card, selectedCards))
            },
        )

        // Always show preview panel to prevent layout jumping
        val currentRoom = uiState.currentRoom
        val isCompact = !isExpandedScreen
        when {
            currentRoom != null && currentRoom.size == 4 -> {
                PreviewPanel(
                    previewEntries = simulateProcessing(selectedCards),
                    isCompact = isCompact,
                )
            }
            currentRoom == null -> {
                PreviewPanel(
                    previewEntries = emptyList(),
                    placeholderText = "Draw a room to see preview",
                    isCompact = isCompact,
                )
            }
            else -> {
                // Room has 1 card remaining
                PreviewPanel(
                    previewEntries = emptyList(),
                    placeholderText = "Draw next room to see preview",
                    isCompact = isCompact,
                )
            }
        }

        // Action buttons
        RoomActionButtons(
            currentRoom = uiState.currentRoom,
            selectedCards = selectedCards,
            canAvoidRoom = uiState.canAvoidRoom,
            isGameOver = false,
            isGameWon = false,
            isCompact = isCompact,
            onAvoidRoom = {
                onIntent(GameIntent.AvoidRoom)
                onSelectedCardsChange(emptyList())
            },
            onProcessCards = {
                onIntent(GameIntent.ProcessSelectedCards(selectedCards))
                onSelectedCardsChange(emptyList())
            },
            onDrawRoom = { onIntent(GameIntent.DrawRoom) },
            onNewGame = {
                onIntent(GameIntent.NewGame)
                onSelectedCardsChange(emptyList())
            },
        )
    }
}

/**
 * Cards section for expanded mode - displays just the room cards.
 */
@Composable
private fun ExpandedCardsSection(
    uiState: GameUiState,
    selectedCards: List<Card>,
    onSelectedCardsChange: (List<Card>) -> Unit,
) {
    if (uiState.isGameOver) {
        GameOverScreen(
            score = uiState.score,
            highestScore = uiState.highestScore,
            isNewHighScore = uiState.isNewHighScore,
            onNewGame = {},
            showButton = false,
        )
    } else if (uiState.isGameWon) {
        GameWonScreen(
            score = uiState.score,
            highestScore = uiState.highestScore,
            isNewHighScore = uiState.isNewHighScore,
            onNewGame = {},
            showButton = false,
        )
    } else {
        RoomCardsDisplay(
            currentRoom = uiState.currentRoom,
            selectedCards = selectedCards,
            isExpanded = true,
            onCardClick = { card ->
                onSelectedCardsChange(toggleCardSelection(card, selectedCards))
            },
        )
    }
}

/**
 * Controls section for expanded mode - action buttons.
 */
@Composable
private fun ExpandedControlsSection(
    uiState: GameUiState,
    selectedCards: List<Card>,
    onSelectedCardsChange: (List<Card>) -> Unit,
    onIntent: (GameIntent) -> Unit,
    simulateProcessing: (List<Card>) -> List<LogEntry>,
) {
    // Always show preview panel to prevent layout jumping
    val currentRoom = uiState.currentRoom
    when {
        uiState.isGameOver || uiState.isGameWon -> {
            // Show empty preview panel during game over to maintain layout
            PreviewPanel(
                previewEntries = emptyList(),
                placeholderText = "Start a new game",
            )
        }
        currentRoom != null && currentRoom.size == 4 -> {
            PreviewPanel(
                previewEntries = simulateProcessing(selectedCards),
            )
        }
        currentRoom == null -> {
            PreviewPanel(
                previewEntries = emptyList(),
                placeholderText = "Draw a room to see preview",
            )
        }
        else -> {
            // Room has 1 card remaining
            PreviewPanel(
                previewEntries = emptyList(),
                placeholderText = "Draw next room to see preview",
            )
        }
    }

    // Action buttons
    RoomActionButtons(
        currentRoom = uiState.currentRoom,
        selectedCards = selectedCards,
        canAvoidRoom = uiState.canAvoidRoom,
        isGameOver = uiState.isGameOver,
        isGameWon = uiState.isGameWon,
        onAvoidRoom = {
            onIntent(GameIntent.AvoidRoom)
            onSelectedCardsChange(emptyList())
        },
        onProcessCards = {
            onIntent(GameIntent.ProcessSelectedCards(selectedCards))
            onSelectedCardsChange(emptyList())
        },
        onDrawRoom = { onIntent(GameIntent.DrawRoom) },
        onNewGame = {
            onIntent(GameIntent.NewGame)
            onSelectedCardsChange(emptyList())
        },
    )
}

@Composable
private fun GameOverScreen(
    score: Int,
    highestScore: Int?,
    isNewHighScore: Boolean,
    onNewGame: () -> Unit,
    showButton: Boolean = true,
) {
    Column(
        modifier =
            Modifier
                .fillMaxWidth()
                .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "Game Over",
            style = MaterialTheme.typography.displayLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.error,
        )

        Text(
            text = "Final Score: $score",
            style = MaterialTheme.typography.headlineLarge,
        )

        if (isNewHighScore) {
            Text(
                text = "NEW HIGH SCORE!",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.tertiary,
            )
        } else if (highestScore != null) {
            Text(
                text = "High Score: $highestScore",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        if (showButton) {
            Button(
                onClick = onNewGame,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "New Game",
                    style = MaterialTheme.typography.titleLarge,
                )
            }
        }
    }
}

@Composable
private fun GameWonScreen(
    score: Int,
    highestScore: Int?,
    isNewHighScore: Boolean,
    onNewGame: () -> Unit,
    showButton: Boolean = true,
) {
    Column(
        modifier =
            Modifier
                .fillMaxWidth()
                .padding(24.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text(
            text = "Victory!",
            style = MaterialTheme.typography.displayLarge,
            fontWeight = FontWeight.Bold,
            color = MaterialTheme.colorScheme.primary,
        )

        Text(
            text = "Final Score: $score",
            style = MaterialTheme.typography.headlineLarge,
        )

        if (isNewHighScore) {
            Text(
                text = "NEW HIGH SCORE!",
                style = MaterialTheme.typography.titleLarge,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.tertiary,
            )
        } else if (highestScore != null) {
            Text(
                text = "High Score: $highestScore",
                style = MaterialTheme.typography.titleMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }

        if (showButton) {
            Button(
                onClick = onNewGame,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "New Game",
                    style = MaterialTheme.typography.titleLarge,
                )
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
fun GameScreenPreview() {
    ScoundroidTheme {
        GameScreen()
    }
}
