package dev.mattbachmann.scoundroid.ui.screen.game

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
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
            // Expanded layout: cards on top, controls on bottom
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .padding(16.dp),
            ) {
                // Top section: Cards (takes available space)
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

                // Bottom section: Title, status, buttons
                Column(
                    modifier =
                        Modifier
                            .fillMaxWidth()
                            .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
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
                    )
                }
            }
        } else {
            // Compact layout: vertical stack
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .padding(16.dp)
                        .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(24.dp),
            ) {
                // Title with action log and help buttons
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "Scoundroid",
                        style = MaterialTheme.typography.displayMedium,
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
    selectedCards: Set<Card>,
): Set<Card> =
    if (card in selectedCards) {
        selectedCards - card
    } else if (selectedCards.size < 3) {
        selectedCards + card
    } else {
        selectedCards
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
        // Active game
        val currentRoom = uiState.currentRoom
        if (currentRoom != null) {
            // Show current room
            if (currentRoom.size == 1) {
                // Single card remaining - show it but don't allow clicking
                // This card becomes part of the next room
                RoomDisplay(
                    cards = currentRoom,
                    selectedCards = emptyList(),
                    onCardClick = null,
                    isExpanded = isExpandedScreen,
                )
            } else {
                // Room of 4 - allow selection
                RoomDisplay(
                    cards = currentRoom,
                    selectedCards = selectedCards,
                    onCardClick = { card ->
                        onSelectedCardsChange(toggleCardSelection(card, selectedCards))
                    },
                    isExpanded = isExpandedScreen,
                )

                // Preview panel - show what will happen when processing selected cards
                PreviewPanel(
                    previewEntries = simulateProcessing(selectedCards),
                )
            }

            // Room actions
            if (currentRoom.size == 4) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    // Avoid room button
                    if (uiState.canAvoidRoom) {
                        OutlinedButton(
                            onClick = {
                                onIntent(GameIntent.AvoidRoom)
                                onSelectedCardsChange(emptyList())
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Avoid Room")
                        }
                    }

                    // Process selected cards button
                    Button(
                        onClick = {
                            onIntent(
                                GameIntent.ProcessSelectedCards(selectedCards),
                            )
                            onSelectedCardsChange(emptyList())
                        },
                        enabled = selectedCards.size == 3,
                        modifier =
                            if (uiState.canAvoidRoom) {
                                Modifier.weight(1f)
                            } else {
                                Modifier.fillMaxWidth()
                            },
                    ) {
                        Text("Process ${selectedCards.size}/3 Cards")
                    }
                }
            } else if (currentRoom.size == 1) {
                // 1 card remaining - show Draw Room button to continue
                Text(
                    text = "This card stays for the next room",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
                )
                Button(
                    onClick = { onIntent(GameIntent.DrawRoom) },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = "Draw Next Room",
                        style = MaterialTheme.typography.titleLarge,
                    )
                }
            }
        } else {
            // No room - show placeholders and draw button
            RoomDisplay(
                cards = emptyList(),
                selectedCards = emptyList(),
                onCardClick = null,
                isExpanded = isExpandedScreen,
                showPlaceholders = true,
            )

            Button(
                onClick = { onIntent(GameIntent.DrawRoom) },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "Draw Room",
                    style = MaterialTheme.typography.titleLarge,
                )
            }
        }

        // New game button (always available)
        Spacer(modifier = Modifier.height(8.dp))
        OutlinedButton(
            onClick = {
                onIntent(GameIntent.NewGame)
                onSelectedCardsChange(emptyList())
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("New Game")
        }
    }
}

/**
 * Cards section for expanded mode - displays just the room cards.
 */
@Composable
private fun ExpandedCardsSection(
    uiState: GameUiState,
    selectedCards: Set<Card>,
    onSelectedCardsChange: (Set<Card>) -> Unit,
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
        val currentRoom = uiState.currentRoom
        if (currentRoom != null) {
            if (currentRoom.size == 1) {
                RoomDisplay(
                    cards = currentRoom,
                    selectedCards = emptySet(),
                    onCardClick = null,
                    isExpanded = true,
                )
            } else {
                RoomDisplay(
                    cards = currentRoom,
                    selectedCards = selectedCards,
                    onCardClick = { card ->
                        onSelectedCardsChange(toggleCardSelection(card, selectedCards))
                    },
                    isExpanded = true,
                )
            }
        } else {
            RoomDisplay(
                cards = emptyList(),
                selectedCards = emptySet(),
                onCardClick = null,
                isExpanded = true,
                showPlaceholders = true,
            )
        }
    }
}

/**
 * Controls section for expanded mode - action buttons.
 */
@Composable
private fun ExpandedControlsSection(
    uiState: GameUiState,
    selectedCards: Set<Card>,
    onSelectedCardsChange: (Set<Card>) -> Unit,
    onIntent: (GameIntent) -> Unit,
) {
    if (uiState.isGameOver || uiState.isGameWon) {
        Button(
            onClick = {
                onIntent(GameIntent.NewGame)
                onSelectedCardsChange(emptySet())
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(
                text = "New Game",
                style = MaterialTheme.typography.titleLarge,
            )
        }
    } else {
        val currentRoom = uiState.currentRoom
        if (currentRoom != null) {
            if (currentRoom.size == 4) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    if (uiState.canAvoidRoom) {
                        OutlinedButton(
                            onClick = {
                                onIntent(GameIntent.AvoidRoom)
                                onSelectedCardsChange(emptySet())
                            },
                            modifier = Modifier.weight(1f),
                        ) {
                            Text("Avoid Room")
                        }
                    }

                    Button(
                        onClick = {
                            onIntent(GameIntent.ProcessSelectedCards(selectedCards.toList()))
                            onSelectedCardsChange(emptySet())
                        },
                        enabled = selectedCards.size == 3,
                        modifier =
                            if (uiState.canAvoidRoom) {
                                Modifier.weight(1f)
                            } else {
                                Modifier.fillMaxWidth()
                            },
                    ) {
                        Text("Process ${selectedCards.size}/3 Cards")
                    }
                }

                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = {
                        onIntent(GameIntent.NewGame)
                        onSelectedCardsChange(emptySet())
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("New Game")
                }
            } else if (currentRoom.size == 1) {
                Text(
                    text = "This card stays for the next room",
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f),
                )
                Button(
                    onClick = { onIntent(GameIntent.DrawRoom) },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = "Draw Next Room",
                        style = MaterialTheme.typography.titleLarge,
                    )
                }

                Spacer(modifier = Modifier.height(8.dp))
                OutlinedButton(
                    onClick = {
                        onIntent(GameIntent.NewGame)
                        onSelectedCardsChange(emptySet())
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("New Game")
                }
            }
        } else {
            Button(
                onClick = { onIntent(GameIntent.DrawRoom) },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "Draw Room",
                    style = MaterialTheme.typography.titleLarge,
                )
            }

            Spacer(modifier = Modifier.height(8.dp))
            OutlinedButton(
                onClick = {
                    onIntent(GameIntent.NewGame)
                    onSelectedCardsChange(emptySet())
                },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("New Game")
            }
        }
    }
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
