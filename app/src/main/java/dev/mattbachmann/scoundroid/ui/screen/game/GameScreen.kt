package dev.mattbachmann.scoundroid.ui.screen.game

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
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
import dev.mattbachmann.scoundroid.ui.component.GameStatusBar
import dev.mattbachmann.scoundroid.ui.component.RoomDisplay
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

/**
 * Main game screen for Scoundrel.
 * Displays the current game state and handles user interactions.
 * Supports responsive layouts for foldable devices.
 */
@Composable
fun GameScreen(
    modifier: Modifier = Modifier,
    viewModel: GameViewModel = viewModel(),
    isExpandedScreen: Boolean = false,
) {
    val uiState by viewModel.uiState.collectAsState()
    var selectedCards by remember { mutableStateOf(setOf<Card>()) }
    var scoreSaved by remember { mutableStateOf(false) }

    // Save score when game ends, reset flag when new game starts
    LaunchedEffect(uiState.isGameOver, uiState.isGameWon) {
        if ((uiState.isGameOver || uiState.isGameWon) && !scoreSaved) {
            viewModel.onIntent(GameIntent.GameEnded(uiState.score, uiState.isGameWon))
            scoreSaved = true
        } else if (!uiState.isGameOver && !uiState.isGameWon) {
            scoreSaved = false
        }
    }

    val isExpanded = isExpandedScreen

    Scaffold(
        modifier = modifier.fillMaxSize(),
    ) { innerPadding ->
        if (isExpanded) {
            // Expanded layout: sidebar on left, game area on right
            Row(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .padding(innerPadding)
                        .padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(24.dp),
            ) {
                // Left sidebar - status bar
                Column(
                    modifier =
                        Modifier
                            .width(200.dp)
                            .fillMaxHeight()
                            .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(16.dp),
                ) {
                    Text(
                        text = "Scoundroid",
                        style = MaterialTheme.typography.headlineMedium,
                        fontWeight = FontWeight.Bold,
                        color = MaterialTheme.colorScheme.primary,
                    )
                    GameStatusBar(
                        health = uiState.health,
                        score = uiState.score,
                        deckSize = uiState.deckSize,
                        weaponState = uiState.weaponState,
                        defeatedMonstersCount = uiState.defeatedMonstersCount,
                        isExpanded = true,
                    )
                }

                // Right side - game area
                Column(
                    modifier =
                        Modifier
                            .weight(1f)
                            .fillMaxHeight()
                            .verticalScroll(rememberScrollState()),
                    verticalArrangement = Arrangement.spacedBy(24.dp),
                    horizontalAlignment = Alignment.CenterHorizontally,
                ) {
                    GameContent(
                        uiState = uiState,
                        selectedCards = selectedCards,
                        onSelectedCardsChange = { selectedCards = it },
                        onIntent = viewModel::onIntent,
                        isExpanded = true,
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
                // Title
                Text(
                    text = "Scoundroid",
                    style = MaterialTheme.typography.displayMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.primary,
                )

                // Game status
                GameStatusBar(
                    health = uiState.health,
                    score = uiState.score,
                    deckSize = uiState.deckSize,
                    weaponState = uiState.weaponState,
                    defeatedMonstersCount = uiState.defeatedMonstersCount,
                    isExpanded = false,
                )

                GameContent(
                    uiState = uiState,
                    selectedCards = selectedCards,
                    onSelectedCardsChange = { selectedCards = it },
                    onIntent = viewModel::onIntent,
                    isExpanded = false,
                )
            }
        }
    }
}

/**
 * Game content that adapts to compact or expanded layouts.
 */
@Composable
private fun GameContent(
    uiState: GameUiState,
    selectedCards: Set<Card>,
    onSelectedCardsChange: (Set<Card>) -> Unit,
    onIntent: (GameIntent) -> Unit,
    isExpanded: Boolean,
) {
    // Game over / won message
    if (uiState.isGameOver) {
        GameOverScreen(
            score = uiState.score,
            highestScore = uiState.highestScore,
            isNewHighScore = uiState.isNewHighScore,
            onNewGame = {
                onIntent(GameIntent.NewGame)
                onSelectedCardsChange(emptySet())
            },
        )
    } else if (uiState.isGameWon) {
        GameWonScreen(
            score = uiState.score,
            highestScore = uiState.highestScore,
            isNewHighScore = uiState.isNewHighScore,
            onNewGame = {
                onIntent(GameIntent.NewGame)
                onSelectedCardsChange(emptySet())
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
                    selectedCards = emptySet(),
                    onCardClick = null,
                    isExpanded = isExpanded,
                )
            } else {
                // Room of 4 - allow selection
                RoomDisplay(
                    cards = currentRoom,
                    selectedCards = selectedCards,
                    onCardClick = { card ->
                        onSelectedCardsChange(
                            if (card in selectedCards) {
                                selectedCards - card
                            } else if (selectedCards.size < 3) {
                                selectedCards + card
                            } else {
                                selectedCards
                            },
                        )
                    },
                    isExpanded = isExpanded,
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
                                onSelectedCardsChange(emptySet())
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
                                GameIntent.ProcessSelectedCards(selectedCards.toList()),
                            )
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
            // No room - show draw button
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
                onSelectedCardsChange(emptySet())
            },
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text("New Game")
        }
    }
}

@Composable
private fun GameOverScreen(
    score: Int,
    highestScore: Int?,
    isNewHighScore: Boolean,
    onNewGame: () -> Unit,
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

@Composable
private fun GameWonScreen(
    score: Int,
    highestScore: Int?,
    isNewHighScore: Boolean,
    onNewGame: () -> Unit,
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

@Preview(showBackground = true)
@Composable
fun GameScreenPreview() {
    ScoundroidTheme {
        GameScreen()
    }
}
