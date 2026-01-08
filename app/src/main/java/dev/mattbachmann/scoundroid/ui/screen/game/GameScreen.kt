package dev.mattbachmann.scoundroid.ui.screen.game

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
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
 */
@Composable
fun GameScreen(
    modifier: Modifier = Modifier,
    viewModel: GameViewModel = viewModel(),
) {
    val uiState by viewModel.uiState.collectAsState()
    var selectedCards by remember { mutableStateOf(setOf<Card>()) }

    Scaffold(
        modifier = modifier.fillMaxSize(),
    ) { innerPadding ->
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
            )

            // Game over / won message
            if (uiState.isGameOver) {
                GameOverScreen(
                    score = uiState.score,
                    onNewGame = {
                        viewModel.onIntent(GameIntent.NewGame)
                        selectedCards = emptySet()
                    },
                )
            } else if (uiState.isGameWon) {
                GameWonScreen(
                    score = uiState.score,
                    onNewGame = {
                        viewModel.onIntent(GameIntent.NewGame)
                        selectedCards = emptySet()
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
                        )
                    } else {
                        // Room of 4 - allow selection
                        RoomDisplay(
                            cards = currentRoom,
                            selectedCards = selectedCards,
                            onCardClick = { card ->
                                selectedCards =
                                    if (card in selectedCards) {
                                        selectedCards - card
                                    } else if (selectedCards.size < 3) {
                                        selectedCards + card
                                    } else {
                                        selectedCards
                                    }
                            },
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
                                        viewModel.onIntent(GameIntent.AvoidRoom)
                                        selectedCards = emptySet()
                                    },
                                    modifier = Modifier.weight(1f),
                                ) {
                                    Text("Avoid Room")
                                }
                            }

                            // Process selected cards button
                            Button(
                                onClick = {
                                    viewModel.onIntent(
                                        GameIntent.ProcessSelectedCards(selectedCards.toList()),
                                    )
                                    selectedCards = emptySet()
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
                            onClick = { viewModel.onIntent(GameIntent.DrawRoom) },
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
                        onClick = { viewModel.onIntent(GameIntent.DrawRoom) },
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
                        viewModel.onIntent(GameIntent.NewGame)
                        selectedCards = emptySet()
                    },
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text("New Game")
                }
            }
        }
    }
}

@Composable
private fun GameOverScreen(
    score: Int,
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
