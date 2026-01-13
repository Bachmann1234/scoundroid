package dev.mattbachmann.scoundroid.ui.screen.game

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Help
import androidx.compose.material.icons.automirrored.filled.List
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilledTonalButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.testTag
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.LogEntry
import dev.mattbachmann.scoundroid.ui.component.ActionLogPanel
import dev.mattbachmann.scoundroid.ui.component.CombatChoicePanel
import dev.mattbachmann.scoundroid.ui.component.GameStatusBar
import dev.mattbachmann.scoundroid.ui.component.HelpContent
import dev.mattbachmann.scoundroid.ui.component.PreviewPanel
import dev.mattbachmann.scoundroid.ui.component.RoomDisplay
import dev.mattbachmann.scoundroid.ui.component.StatusBarLayout
import dev.mattbachmann.scoundroid.ui.theme.ButtonPrimary
import dev.mattbachmann.scoundroid.ui.theme.GradientBottom
import dev.mattbachmann.scoundroid.ui.theme.GradientTop
import dev.mattbachmann.scoundroid.ui.theme.Purple80
import dev.mattbachmann.scoundroid.ui.theme.PurpleGrey80
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

/**
 * Screen size classes for responsive layouts.
 * - COMPACT: Small phones (height < 700dp) - aggressive space saving, 1x4 cards
 * - MEDIUM: Fold cover screens, regular phones - comfortable layout, 2x2 cards
 * - EXPANDED: Tablets, unfolded foldables - large cards in horizontal row
 */
enum class ScreenSizeClass {
    COMPACT,
    MEDIUM,
    EXPANDED,
}

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
    screenSizeClass: ScreenSizeClass = ScreenSizeClass.MEDIUM,
) {
    val isExpandedScreen = screenSizeClass == ScreenSizeClass.EXPANDED
    val uiState by viewModel.uiState.collectAsState()
    var selectedCards by remember { mutableStateOf(listOf<Card>()) }
    val sheetState = rememberModalBottomSheetState()
    var showSeedDialog by remember { mutableStateOf(false) }
    val context = LocalContext.current
    val clipboardManager = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager

    // Save score when game ends
    // Important: Don't trigger GameEnded while a combat choice is pending (Issue #36)
    // This prevents the bug where GameEnded resets UI state mid-processing, causing an infinite loop
    LaunchedEffect(uiState.isGameOver, uiState.isGameWon, uiState.pendingCombatChoice) {
        if ((uiState.isGameOver || uiState.isGameWon) && uiState.pendingCombatChoice == null) {
            viewModel.onIntent(GameIntent.GameEnded(uiState.score, uiState.isGameWon))
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
            ActionLogPanel(
                actionLog = uiState.actionLog,
                gameSeed = uiState.gameSeed,
                onCopySeed = {
                    clipboardManager.setPrimaryClip(ClipData.newPlainText("Seed", uiState.gameSeed.toString()))
                    Toast.makeText(context, "Seed copied!", Toast.LENGTH_SHORT).show()
                },
            )
        }
    }

    // Seed entry dialog
    if (showSeedDialog) {
        SeedEntryDialog(
            onDismiss = { showSeedDialog = false },
            onConfirm = { seed ->
                viewModel.onIntent(GameIntent.NewGameWithSeed(seed))
                selectedCards = emptyList()
                showSeedDialog = false
            },
        )
    }

    Scaffold(
        modifier = modifier.fillMaxSize(),
    ) { innerPadding ->
        // Background gradient brush
        val backgroundGradient =
            remember {
                Brush.verticalGradient(
                    colors = listOf(GradientTop, GradientBottom),
                )
            }

        if (isExpandedScreen) {
            // Expanded layout: title at top, cards in center, controls on bottom
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .background(backgroundGradient)
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
                        color = Purple80,
                    )
                    Row {
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowActionLog) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.List,
                                contentDescription = "Action Log",
                                tint = Purple80,
                            )
                        }
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowHelp) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.Help,
                                contentDescription = "Help",
                                tint = Purple80,
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
                        onIntent = viewModel::onIntent,
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
                        onCopySeed = {
                            clipboardManager.setPrimaryClip(ClipData.newPlainText("Seed", uiState.gameSeed.toString()))
                            Toast.makeText(context, "Seed copied!", Toast.LENGTH_SHORT).show()
                        },
                        onPlaySeed = { showSeedDialog = true },
                    )
                }
            }
        } else {
            // Compact layout: vertical stack with scroll fallback for smaller screens
            Column(
                modifier =
                    Modifier
                        .fillMaxSize()
                        .background(backgroundGradient)
                        .padding(innerPadding)
                        .verticalScroll(rememberScrollState())
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
                        color = Purple80,
                    )
                    Row {
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowActionLog) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.List,
                                contentDescription = "Action Log",
                                tint = Purple80,
                            )
                        }
                        IconButton(onClick = { viewModel.onIntent(GameIntent.ShowHelp) }) {
                            Icon(
                                imageVector = Icons.AutoMirrored.Filled.Help,
                                contentDescription = "Help",
                                tint = Purple80,
                            )
                        }
                    }
                }

                // Game status - use COMPACT for small phones, MEDIUM for fold cover
                val statusBarLayout =
                    when (screenSizeClass) {
                        ScreenSizeClass.COMPACT -> StatusBarLayout.COMPACT
                        ScreenSizeClass.MEDIUM -> StatusBarLayout.MEDIUM
                        ScreenSizeClass.EXPANDED -> StatusBarLayout.INLINE
                    }
                GameStatusBar(
                    health = uiState.health,
                    score = uiState.score,
                    deckSize = uiState.deckSize,
                    weaponState = uiState.weaponState,
                    defeatedMonstersCount = uiState.defeatedMonstersCount,
                    layout = statusBarLayout,
                )

                GameContent(
                    uiState = uiState,
                    selectedCards = selectedCards,
                    onSelectedCardsChange = { selectedCards = it },
                    onIntent = viewModel::onIntent,
                    simulateProcessing = viewModel::simulateProcessing,
                    screenSizeClass = screenSizeClass,
                    onCopySeed = {
                        clipboardManager.setPrimaryClip(ClipData.newPlainText("Seed", uiState.gameSeed.toString()))
                        Toast.makeText(context, "Seed copied!", Toast.LENGTH_SHORT).show()
                    },
                    onPlaySeed = { showSeedDialog = true },
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
    onPlaySeed: (() -> Unit)? = null,
    isCompact: Boolean = false,
) {
    val buttonSpacing = if (isCompact) 4.dp else 8.dp
    val buttonTextStyle = if (isCompact) MaterialTheme.typography.titleMedium else MaterialTheme.typography.titleLarge
    val buttonShape = remember { RoundedCornerShape(12.dp) }
    val primaryButtonColors =
        ButtonDefaults.buttonColors(
            containerColor = ButtonPrimary,
            contentColor = Color.White,
            disabledContainerColor = ButtonPrimary.copy(alpha = 0.15f),
            disabledContentColor = Color.White.copy(alpha = 0.5f),
        )
    val primaryButtonElevation =
        ButtonDefaults.buttonElevation(
            defaultElevation = 4.dp,
            pressedElevation = 8.dp,
        )
    val outlinedButtonColors =
        ButtonDefaults.outlinedButtonColors(
            contentColor = Purple80,
        )

    Column(
        modifier = modifier,
        verticalArrangement = Arrangement.spacedBy(buttonSpacing),
    ) {
        if (isGameOver || isGameWon) {
            Button(
                onClick = onNewGame,
                modifier = Modifier.fillMaxWidth(),
                shape = buttonShape,
                colors = primaryButtonColors,
                elevation = primaryButtonElevation,
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
                            FilledTonalButton(
                                onClick = onAvoidRoom,
                                modifier = Modifier.weight(1f),
                                shape = buttonShape,
                            ) {
                                Text(
                                    text = "Avoid Room",
                                    style = buttonTextStyle,
                                )
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
                            shape = buttonShape,
                            colors = primaryButtonColors,
                            elevation = primaryButtonElevation,
                        ) {
                            Text(
                                text = if (selectedCards.size == 3) "Go" else "Pick ${3 - selectedCards.size}",
                                style = buttonTextStyle,
                            )
                        }
                    }

                    OutlinedButton(
                        onClick = onNewGame,
                        modifier = Modifier.fillMaxWidth(),
                        shape = buttonShape,
                        colors = outlinedButtonColors,
                    ) {
                        Text(
                            text = "New Game",
                            style = buttonTextStyle,
                        )
                    }
                }
                1 -> {
                    Button(
                        onClick = onDrawRoom,
                        modifier = Modifier.fillMaxWidth(),
                        shape = buttonShape,
                        colors = primaryButtonColors,
                        elevation = primaryButtonElevation,
                    ) {
                        Text(
                            text = "Draw Next Room",
                            style = buttonTextStyle,
                        )
                    }

                    OutlinedButton(
                        onClick = onNewGame,
                        modifier = Modifier.fillMaxWidth(),
                        shape = buttonShape,
                        colors = outlinedButtonColors,
                    ) {
                        Text(
                            text = "New Game",
                            style = buttonTextStyle,
                        )
                    }
                }
            }
        } else {
            Button(
                onClick = onDrawRoom,
                modifier = Modifier.fillMaxWidth(),
                shape = buttonShape,
                colors = primaryButtonColors,
                elevation = primaryButtonElevation,
            ) {
                Text(
                    text = "Draw Room",
                    style = buttonTextStyle,
                )
            }

            // New Game and Play Custom Seed in a row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(buttonSpacing),
            ) {
                OutlinedButton(
                    onClick = onNewGame,
                    modifier = Modifier.weight(1f),
                    shape = buttonShape,
                    colors = outlinedButtonColors,
                ) {
                    Text(
                        text = "New Game",
                        style = buttonTextStyle,
                    )
                }

                if (onPlaySeed != null) {
                    OutlinedButton(
                        onClick = onPlaySeed,
                        modifier = Modifier.weight(1f),
                        shape = buttonShape,
                        colors = outlinedButtonColors,
                    ) {
                        Text(
                            text = "Custom Seed",
                            style = buttonTextStyle,
                        )
                    }
                }
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
    screenSizeClass: ScreenSizeClass,
    onCardClick: ((Card) -> Unit)?,
) {
    if (currentRoom != null) {
        if (currentRoom.size == 1) {
            RoomDisplay(
                cards = currentRoom,
                selectedCards = emptyList(),
                onCardClick = null,
                screenSizeClass = screenSizeClass,
            )
        } else {
            RoomDisplay(
                cards = currentRoom,
                selectedCards = selectedCards,
                onCardClick = onCardClick,
                screenSizeClass = screenSizeClass,
            )
        }
    } else {
        RoomDisplay(
            cards = emptyList(),
            selectedCards = emptyList(),
            onCardClick = null,
            screenSizeClass = screenSizeClass,
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
    screenSizeClass: ScreenSizeClass,
    onCopySeed: () -> Unit,
    onPlaySeed: () -> Unit,
) {
    val isExpandedScreen = screenSizeClass == ScreenSizeClass.EXPANDED
    val isCompact = !isExpandedScreen

    // Use displayMode for consistent rendering across layouts
    when (val mode = uiState.displayMode) {
        is GameDisplayMode.GameOver -> {
            GameOverScreen(
                score = mode.score,
                highestScore = mode.highestScore,
                isNewHighScore = mode.isNewHighScore,
                gameSeed = mode.gameSeed,
                onNewGame = {
                    onIntent(GameIntent.NewGame)
                    onSelectedCardsChange(emptyList())
                },
                onRetryGame = {
                    onIntent(GameIntent.RetryGame)
                    onSelectedCardsChange(emptyList())
                },
                onCopySeed = onCopySeed,
                onPlaySeed = onPlaySeed,
            )
        }
        is GameDisplayMode.GameWon -> {
            GameWonScreen(
                score = mode.score,
                highestScore = mode.highestScore,
                isNewHighScore = mode.isNewHighScore,
                gameSeed = mode.gameSeed,
                onNewGame = {
                    onIntent(GameIntent.NewGame)
                    onSelectedCardsChange(emptyList())
                },
                onRetryGame = {
                    onIntent(GameIntent.RetryGame)
                    onSelectedCardsChange(emptyList())
                },
                onCopySeed = onCopySeed,
                onPlaySeed = onPlaySeed,
            )
        }
        is GameDisplayMode.CombatChoice -> {
            CombatChoicePanel(
                choice = mode.choice,
                onUseWeapon = { onIntent(GameIntent.ResolveCombatChoice(useWeapon = true)) },
                onFightBarehanded = { onIntent(GameIntent.ResolveCombatChoice(useWeapon = false)) },
                screenSizeClass = screenSizeClass,
            )
        }
        is GameDisplayMode.ActiveGame -> {
            // Active game - show room cards
            RoomCardsDisplay(
                currentRoom = mode.currentRoom,
                selectedCards = selectedCards,
                screenSizeClass = screenSizeClass,
                onCardClick = { card ->
                    onSelectedCardsChange(toggleCardSelection(card, selectedCards))
                },
            )

            // Always show preview panel to prevent layout jumping
            when {
                mode.currentRoom != null && mode.currentRoom.size == 4 -> {
                    PreviewPanel(
                        previewEntries = simulateProcessing(selectedCards),
                        isCompact = isCompact,
                    )
                }
                mode.currentRoom == null -> {
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
                currentRoom = mode.currentRoom,
                selectedCards = selectedCards,
                canAvoidRoom = mode.canAvoidRoom,
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
                onPlaySeed = onPlaySeed,
            )
        }
    }
}

/**
 * Cards section for expanded mode - displays just the room cards.
 * Uses displayMode for consistent rendering with compact layout.
 */
@Composable
private fun ExpandedCardsSection(
    uiState: GameUiState,
    selectedCards: List<Card>,
    onSelectedCardsChange: (List<Card>) -> Unit,
    onIntent: (GameIntent) -> Unit,
) {
    when (val mode = uiState.displayMode) {
        is GameDisplayMode.GameOver -> {
            GameOverScreen(
                score = mode.score,
                highestScore = mode.highestScore,
                isNewHighScore = mode.isNewHighScore,
                gameSeed = mode.gameSeed,
                onNewGame = {},
                onRetryGame = {},
                onCopySeed = {},
                onPlaySeed = {},
                showButton = false,
            )
        }
        is GameDisplayMode.GameWon -> {
            GameWonScreen(
                score = mode.score,
                highestScore = mode.highestScore,
                isNewHighScore = mode.isNewHighScore,
                gameSeed = mode.gameSeed,
                onNewGame = {},
                onRetryGame = {},
                onCopySeed = {},
                onPlaySeed = {},
                showButton = false,
            )
        }
        is GameDisplayMode.CombatChoice -> {
            // Combat choice needed - show cards only, buttons go in controls section
            CombatChoicePanel(
                choice = mode.choice,
                onUseWeapon = { onIntent(GameIntent.ResolveCombatChoice(useWeapon = true)) },
                onFightBarehanded = { onIntent(GameIntent.ResolveCombatChoice(useWeapon = false)) },
                screenSizeClass = ScreenSizeClass.EXPANDED,
                showButtons = false,
            )
        }
        is GameDisplayMode.ActiveGame -> {
            RoomCardsDisplay(
                currentRoom = mode.currentRoom,
                selectedCards = selectedCards,
                screenSizeClass = ScreenSizeClass.EXPANDED,
                onCardClick = { card ->
                    onSelectedCardsChange(toggleCardSelection(card, selectedCards))
                },
            )
        }
    }
}

/**
 * Controls section for expanded mode - action buttons.
 * Uses displayMode for consistent rendering with compact layout.
 */
@Composable
private fun ExpandedControlsSection(
    uiState: GameUiState,
    selectedCards: List<Card>,
    onSelectedCardsChange: (List<Card>) -> Unit,
    onIntent: (GameIntent) -> Unit,
    simulateProcessing: (List<Card>) -> List<LogEntry>,
    onCopySeed: () -> Unit,
    onPlaySeed: () -> Unit,
) {
    val buttonShape = remember { RoundedCornerShape(12.dp) }
    val primaryButtonColors = ButtonDefaults.buttonColors(containerColor = ButtonPrimary, contentColor = Color.White)
    val primaryButtonElevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp, pressedElevation = 8.dp)
    val outlinedButtonColors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80)

    when (val mode = uiState.displayMode) {
        is GameDisplayMode.CombatChoice -> {
            val choice = mode.choice
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                // Use Weapon button
                Button(
                    onClick = { onIntent(GameIntent.ResolveCombatChoice(useWeapon = true)) },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors =
                        ButtonDefaults.buttonColors(
                            containerColor = Color(0xFF1976D2),
                            contentColor = Color.White,
                        ),
                    elevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp),
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Use Weapon",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text =
                                if (choice.weaponDamage == 0) {
                                    "No damage!"
                                } else {
                                    "Take ${choice.weaponDamage} damage"
                                },
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text(
                            text = "Degrades to ${choice.weaponDegradedTo}",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }

                // Fight Barehanded button
                OutlinedButton(
                    onClick = { onIntent(GameIntent.ResolveCombatChoice(useWeapon = false)) },
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80),
                ) {
                    Column(horizontalAlignment = Alignment.CenterHorizontally) {
                        Text(
                            text = "Fight Barehanded",
                            style = MaterialTheme.typography.titleLarge,
                            fontWeight = FontWeight.Bold,
                        )
                        Text(
                            text = "Take ${choice.barehandedDamage} damage",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                        Text(
                            text = "Keeps weapon",
                            style = MaterialTheme.typography.bodyMedium,
                        )
                    }
                }
            }
        }
        is GameDisplayMode.GameOver, is GameDisplayMode.GameWon -> {
            // Show seed display and buttons during game over/won
            val gameSeed =
                when (mode) {
                    is GameDisplayMode.GameOver -> mode.gameSeed
                    is GameDisplayMode.GameWon -> mode.gameSeed
                }
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                // Seed display with copy button
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Text(
                        text = "Seed: $gameSeed",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                    )
                    IconButton(
                        onClick = onCopySeed,
                        modifier = Modifier.size(32.dp),
                    ) {
                        Icon(
                            imageVector = Icons.Default.ContentCopy,
                            contentDescription = "Copy seed",
                            tint = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.size(18.dp),
                        )
                    }
                }

                // Retry button
                OutlinedButton(
                    onClick = {
                        onIntent(GameIntent.RetryGame)
                        onSelectedCardsChange(emptyList())
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = buttonShape,
                    colors = outlinedButtonColors,
                ) {
                    Text(
                        text = "Retry",
                        style = MaterialTheme.typography.titleLarge,
                    )
                }

                // New Game button
                Button(
                    onClick = {
                        onIntent(GameIntent.NewGame)
                        onSelectedCardsChange(emptyList())
                    },
                    modifier = Modifier.fillMaxWidth(),
                    shape = buttonShape,
                    colors = primaryButtonColors,
                    elevation = primaryButtonElevation,
                ) {
                    Text(
                        text = "New Game",
                        style = MaterialTheme.typography.titleLarge,
                    )
                }

                // Play Custom Seed button
                TextButton(
                    onClick = onPlaySeed,
                    modifier = Modifier.fillMaxWidth(),
                ) {
                    Text(
                        text = "Play Custom Seed",
                        style = MaterialTheme.typography.bodyLarge,
                    )
                }
            }
        }
        is GameDisplayMode.ActiveGame -> {
            val currentRoom = mode.currentRoom
            when {
                currentRoom != null && currentRoom.size == 4 -> {
                    PreviewPanel(
                        previewEntries = simulateProcessing(selectedCards),
                    )
                    RoomActionButtons(
                        currentRoom = currentRoom,
                        selectedCards = selectedCards,
                        canAvoidRoom = mode.canAvoidRoom,
                        isGameOver = false,
                        isGameWon = false,
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
                currentRoom == null -> {
                    PreviewPanel(
                        previewEntries = emptyList(),
                        placeholderText = "Draw a room to see preview",
                    )
                    RoomActionButtons(
                        currentRoom = null,
                        selectedCards = selectedCards,
                        canAvoidRoom = false,
                        isGameOver = false,
                        isGameWon = false,
                        onAvoidRoom = {},
                        onProcessCards = {},
                        onDrawRoom = { onIntent(GameIntent.DrawRoom) },
                        onNewGame = {
                            onIntent(GameIntent.NewGame)
                            onSelectedCardsChange(emptyList())
                        },
                        onPlaySeed = onPlaySeed,
                    )
                }
                else -> {
                    // Room has 1 card remaining
                    PreviewPanel(
                        previewEntries = emptyList(),
                        placeholderText = "Draw next room to see preview",
                    )
                    RoomActionButtons(
                        currentRoom = currentRoom,
                        selectedCards = selectedCards,
                        canAvoidRoom = false,
                        isGameOver = false,
                        isGameWon = false,
                        onAvoidRoom = {},
                        onProcessCards = {},
                        onDrawRoom = { onIntent(GameIntent.DrawRoom) },
                        onNewGame = {
                            onIntent(GameIntent.NewGame)
                            onSelectedCardsChange(emptyList())
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun GameOverScreen(
    score: Int,
    highestScore: Int?,
    isNewHighScore: Boolean,
    gameSeed: Long,
    onNewGame: () -> Unit,
    onRetryGame: () -> Unit,
    onCopySeed: () -> Unit,
    onPlaySeed: () -> Unit,
    showButton: Boolean = true,
) {
    Column(
        modifier =
            Modifier
                .fillMaxWidth()
                .padding(24.dp)
                .testTag("game_over_screen"),
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
            color = Color.White,
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
                color = PurpleGrey80,
            )
        }

        if (showButton) {
            // Seed display with copy button
            Row(
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Seed: $gameSeed",
                    style = MaterialTheme.typography.bodyMedium,
                    color = PurpleGrey80,
                )
                IconButton(
                    onClick = onCopySeed,
                    modifier = Modifier.size(32.dp),
                ) {
                    Icon(
                        imageVector = Icons.Default.ContentCopy,
                        contentDescription = "Copy seed",
                        tint = PurpleGrey80,
                        modifier = Modifier.size(18.dp),
                    )
                }
            }

            val buttonShape = remember { RoundedCornerShape(12.dp) }
            val buttonColors = ButtonDefaults.buttonColors(containerColor = ButtonPrimary, contentColor = Color.White)
            val buttonElevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp, pressedElevation = 8.dp)
            val outlinedButtonColors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80)

            // Retry button
            OutlinedButton(
                onClick = onRetryGame,
                modifier = Modifier.fillMaxWidth(),
                shape = buttonShape,
                colors = outlinedButtonColors,
            ) {
                Text(
                    text = "Retry",
                    style = MaterialTheme.typography.titleLarge,
                )
            }

            // New Game button
            Button(
                onClick = onNewGame,
                modifier = Modifier.fillMaxWidth(),
                shape = buttonShape,
                colors = buttonColors,
                elevation = buttonElevation,
            ) {
                Text(
                    text = "New Game",
                    style = MaterialTheme.typography.titleLarge,
                )
            }

            // Play Custom Seed button
            TextButton(
                onClick = onPlaySeed,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "Play Custom Seed",
                    style = MaterialTheme.typography.bodyLarge,
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
    gameSeed: Long,
    onNewGame: () -> Unit,
    onRetryGame: () -> Unit,
    onCopySeed: () -> Unit,
    onPlaySeed: () -> Unit,
    showButton: Boolean = true,
) {
    Column(
        modifier =
            Modifier
                .fillMaxWidth()
                .padding(24.dp)
                .testTag("victory_screen"),
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
            color = Color.White,
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
                color = PurpleGrey80,
            )
        }

        if (showButton) {
            // Seed display with copy button
            Row(
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(
                    text = "Seed: $gameSeed",
                    style = MaterialTheme.typography.bodyMedium,
                    color = PurpleGrey80,
                )
                IconButton(
                    onClick = onCopySeed,
                    modifier = Modifier.size(32.dp),
                ) {
                    Icon(
                        imageVector = Icons.Default.ContentCopy,
                        contentDescription = "Copy seed",
                        tint = PurpleGrey80,
                        modifier = Modifier.size(18.dp),
                    )
                }
            }

            val buttonShape = remember { RoundedCornerShape(12.dp) }
            val buttonColors = ButtonDefaults.buttonColors(containerColor = ButtonPrimary, contentColor = Color.White)
            val buttonElevation = ButtonDefaults.buttonElevation(defaultElevation = 4.dp, pressedElevation = 8.dp)
            val outlinedButtonColors = ButtonDefaults.outlinedButtonColors(contentColor = Purple80)

            // Retry button
            OutlinedButton(
                onClick = onRetryGame,
                modifier = Modifier.fillMaxWidth(),
                shape = buttonShape,
                colors = outlinedButtonColors,
            ) {
                Text(
                    text = "Retry",
                    style = MaterialTheme.typography.titleLarge,
                )
            }

            // New Game button
            Button(
                onClick = onNewGame,
                modifier = Modifier.fillMaxWidth(),
                shape = buttonShape,
                colors = buttonColors,
                elevation = buttonElevation,
            ) {
                Text(
                    text = "New Game",
                    style = MaterialTheme.typography.titleLarge,
                )
            }

            // Play Custom Seed button
            TextButton(
                onClick = onPlaySeed,
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(
                    text = "Play Custom Seed",
                    style = MaterialTheme.typography.bodyLarge,
                )
            }
        }
    }
}

/**
 * Dialog for entering a custom seed to start a game.
 */
@Composable
private fun SeedEntryDialog(
    onDismiss: () -> Unit,
    onConfirm: (Long) -> Unit,
) {
    var seedText by remember { mutableStateOf("") }
    var isError by remember { mutableStateOf(false) }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Enter Seed") },
        text = {
            OutlinedTextField(
                value = seedText,
                onValueChange = {
                    seedText = it
                    isError = false
                },
                label = { Text("Seed") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                isError = isError,
                supportingText =
                    if (isError) {
                        { Text("Invalid seed - please enter a number") }
                    } else {
                        null
                    },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
        },
        confirmButton = {
            TextButton(
                onClick = {
                    val seed = seedText.toLongOrNull()
                    if (seed != null) {
                        onConfirm(seed)
                    } else {
                        isError = true
                    }
                },
            ) {
                Text("Start Game")
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("Cancel")
            }
        },
    )
}

@Preview(showBackground = true)
@Composable
fun GameScreenPreview() {
    ScoundroidTheme {
        GameScreen()
    }
}
