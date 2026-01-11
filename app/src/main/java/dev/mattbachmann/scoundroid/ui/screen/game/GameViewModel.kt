package dev.mattbachmann.scoundroid.ui.screen.game

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.LogEntry
import dev.mattbachmann.scoundroid.data.repository.HighScoreRepository
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlin.random.Random

/**
 * ViewModel for the game screen, managing game state and handling user intents.
 * Follows MVI (Model-View-Intent) pattern.
 *
 * @param highScoreRepository Repository for persisting high scores
 * @param randomSeed Optional seed for deterministic shuffling (useful for tests)
 */
class GameViewModel(
    private val highScoreRepository: HighScoreRepository? = null,
    private val randomSeed: Long? = null,
) : ViewModel() {
    // The seed used for the current game's deck shuffle
    private var currentGameSeed: Long = randomSeed ?: System.currentTimeMillis()

    private fun createRandom(): Random = Random(currentGameSeed)

    private val initialGameState = GameState.newGame(createRandom())
    private val mutableGameState = MutableStateFlow(initialGameState)
    private val gameState: StateFlow<GameState> = mutableGameState.asStateFlow()

    private val initialGameStarted = LogEntry.GameStarted(timestamp = System.currentTimeMillis())
    private val _uiState =
        MutableStateFlow(
            initialGameState
                .toUiState()
                .copy(actionLog = listOf(initialGameStarted)),
        )
    val uiState: StateFlow<GameUiState> = _uiState.asStateFlow()

    private var highestScore: Int? = null
    private val actionLogEntries = mutableListOf<LogEntry>(initialGameStarted)

    // State for paused card processing during combat choice
    private var pendingCardsToProcess: MutableList<Card> = mutableListOf()
    private var processingState: GameState? = null

    init {
        loadHighestScore()
    }

    private fun loadHighestScore() {
        viewModelScope.launch {
            highestScore = highScoreRepository?.getHighestScore()
            updateUiStateWithHighScore()
        }
    }

    private suspend fun updateUiStateWithHighScore() {
        val currentScore = gameState.value.calculateScore()
        val isNewHigh = highScoreRepository?.isNewHighScore(currentScore) ?: false
        val currentShowHelp = _uiState.value.showHelp
        val currentShowActionLog = _uiState.value.showActionLog
        // Issue #36: Preserve pendingCombatChoice to avoid resetting state mid-processing
        val currentPendingCombatChoice = _uiState.value.pendingCombatChoice
        _uiState.value =
            gameState.value.toUiState().copy(
                highestScore = highestScore,
                isNewHighScore = isNewHigh,
                showHelp = currentShowHelp,
                showActionLog = currentShowActionLog,
                actionLog = actionLogEntries.toList(),
                pendingCombatChoice = currentPendingCombatChoice,
            )
    }

    /**
     * Handles user intents and updates game state accordingly.
     */
    fun onIntent(intent: GameIntent) {
        viewModelScope.launch {
            when (intent) {
                is GameIntent.NewGame -> handleNewGame()
                is GameIntent.DrawRoom -> handleDrawRoom()
                is GameIntent.AvoidRoom -> handleAvoidRoom()
                is GameIntent.ProcessSelectedCards -> handleProcessSelectedCards(intent.selectedCards)
                is GameIntent.GameEnded -> handleGameEnded(intent.score, intent.won)
                is GameIntent.ShowHelp -> handleShowHelp()
                is GameIntent.HideHelp -> handleHideHelp()
                is GameIntent.ShowActionLog -> handleShowActionLog()
                is GameIntent.HideActionLog -> handleHideActionLog()
                is GameIntent.ResolveCombatChoice -> handleResolveCombatChoice(intent.useWeapon)
                is GameIntent.RetryGame -> handleRetryGame()
                is GameIntent.NewGameWithSeed -> handleNewGameWithSeed(intent.seed)
            }
        }
    }

    private fun handleNewGame() {
        // Generate a new seed for the new game
        currentGameSeed = System.currentTimeMillis()

        // Clear any pending combat state
        pendingCardsToProcess.clear()
        processingState = null
        _uiState.value = _uiState.value.copy(pendingCombatChoice = null)

        actionLogEntries.clear()
        actionLogEntries.add(LogEntry.GameStarted(timestamp = System.currentTimeMillis()))
        updateGameState(GameState.newGame(createRandom()))
    }

    private fun handleRetryGame() {
        // Keep the same seed - just reset the game state
        // Clear any pending combat state
        pendingCardsToProcess.clear()
        processingState = null
        _uiState.value = _uiState.value.copy(pendingCombatChoice = null)

        actionLogEntries.clear()
        actionLogEntries.add(LogEntry.GameStarted(timestamp = System.currentTimeMillis()))
        updateGameState(GameState.newGame(createRandom()))
    }

    private fun handleNewGameWithSeed(seed: Long) {
        // Use the provided seed
        currentGameSeed = seed

        // Clear any pending combat state
        pendingCardsToProcess.clear()
        processingState = null
        _uiState.value = _uiState.value.copy(pendingCombatChoice = null)

        actionLogEntries.clear()
        actionLogEntries.add(LogEntry.GameStarted(timestamp = System.currentTimeMillis()))
        updateGameState(GameState.newGame(createRandom()))
    }

    private fun handleDrawRoom() {
        val stateBefore = gameState.value
        val stateAfter = stateBefore.drawRoom()
        // Calculate actual cards drawn based on room size change
        val roomSizeBefore = stateBefore.currentRoom?.size ?: 0
        val roomSizeAfter = stateAfter.currentRoom?.size ?: 0
        val cardsDrawn = roomSizeAfter - roomSizeBefore
        actionLogEntries.add(
            LogEntry.RoomDrawn(
                timestamp = System.currentTimeMillis(),
                cardsDrawn = cardsDrawn,
                deckSizeAfter = stateAfter.deck.cards.size,
            ),
        )
        updateGameState(stateAfter)
    }

    private fun handleAvoidRoom() {
        val stateBefore = gameState.value
        val cardsReturned = stateBefore.currentRoom?.size ?: 0
        actionLogEntries.add(
            LogEntry.RoomAvoided(
                timestamp = System.currentTimeMillis(),
                cardsReturned = cardsReturned,
            ),
        )
        // Avoid the room, then auto-draw the next room
        val stateAfterAvoid = stateBefore.avoidRoom()
        val stateAfterDraw = stateAfterAvoid.drawRoom()

        // Log the auto-drawn room
        val cardsDrawn = stateAfterDraw.currentRoom?.size ?: 0
        actionLogEntries.add(
            LogEntry.RoomDrawn(
                timestamp = System.currentTimeMillis(),
                cardsDrawn = cardsDrawn,
                deckSizeAfter = stateAfterDraw.deck.cards.size,
            ),
        )
        updateGameState(stateAfterDraw)
    }

    private fun handleProcessSelectedCards(selectedCards: List<Card>) {
        // First, select the cards (leaves unselected card for next room)
        processingState = gameState.value.selectCards(selectedCards)
        pendingCardsToProcess = selectedCards.toMutableList()

        // Process cards, potentially pausing for combat choices
        processNextCard()
    }

    /**
     * Processes cards in the pending list iteratively.
     * May pause and return if a combat choice is needed.
     */
    private fun processNextCard() {
        while (true) {
            val state = processingState ?: return
            if (pendingCardsToProcess.isEmpty()) {
                // All cards processed, finalize
                updateGameState(state)
                processingState = null
                return
            }

            val card = pendingCardsToProcess.first()
            val healthBefore = state.health
            val weaponBefore = state.weaponState?.weapon
            val usedPotionBefore = state.usedPotionThisTurn

            when (card.type) {
                CardType.MONSTER -> {
                    val weaponState = state.weaponState

                    if (weaponState != null && weaponState.canDefeat(card)) {
                        // Player has a choice - pause for combat decision
                        val weapon = weaponState.weapon
                        val weaponDamage = (card.value - weapon.value).coerceAtLeast(0)
                        val barehandedDamage = card.value

                        val pendingChoice =
                            PendingCombatChoice(
                                monster = card,
                                weapon = weapon,
                                weaponDamage = weaponDamage,
                                barehandedDamage = barehandedDamage,
                                weaponDegradedTo = card.value,
                                remainingCards = pendingCardsToProcess.drop(1),
                            )

                        // Update UI to show combat choice with current processing state
                        // (health/weapon may have changed from earlier cards in this batch)
                        _uiState.value =
                            state.toUiState().copy(
                                highestScore = highestScore,
                                isNewHighScore =
                                    highestScore?.let { state.calculateScore() > it }
                                        ?: (highScoreRepository != null),
                                showHelp = _uiState.value.showHelp,
                                showActionLog = _uiState.value.showActionLog,
                                pendingCombatChoice = pendingChoice,
                                actionLog = actionLogEntries.toList(),
                            )
                        // Don't remove from pending yet - will be processed when choice is made
                        return
                    } else {
                        // No weapon or can't use it - fight barehanded automatically
                        val newState = state.fightMonsterBarehanded(card)
                        actionLogEntries.add(
                            LogEntry.MonsterFought(
                                timestamp = System.currentTimeMillis(),
                                monster = card,
                                weaponUsed = null,
                                damageBlocked = 0,
                                damageTaken = card.value,
                                healthBefore = healthBefore,
                                healthAfter = newState.health,
                            ),
                        )
                        processingState = newState
                        pendingCardsToProcess.removeAt(0)
                        // Continue loop to process next card
                    }
                }
                CardType.WEAPON -> {
                    val newState = state.equipWeapon(card)
                    actionLogEntries.add(
                        LogEntry.WeaponEquipped(
                            timestamp = System.currentTimeMillis(),
                            weapon = card,
                            replacedWeapon = weaponBefore,
                        ),
                    )
                    processingState = newState
                    pendingCardsToProcess.removeAt(0)
                    // Continue loop to process next card
                }
                CardType.POTION -> {
                    val wasDiscarded = usedPotionBefore
                    val newState = state.usePotion(card)
                    val healthRestored = if (wasDiscarded) 0 else newState.health - healthBefore

                    actionLogEntries.add(
                        LogEntry.PotionUsed(
                            timestamp = System.currentTimeMillis(),
                            potion = card,
                            healthRestored = healthRestored,
                            healthBefore = healthBefore,
                            healthAfter = newState.health,
                            wasDiscarded = wasDiscarded,
                        ),
                    )
                    processingState = newState
                    pendingCardsToProcess.removeAt(0)
                    // Continue loop to process next card
                }
            }
        }
    }

    /**
     * Handles the player's combat choice (weapon vs barehanded).
     */
    private fun handleResolveCombatChoice(useWeapon: Boolean) {
        val state = processingState ?: return
        val choice = _uiState.value.pendingCombatChoice ?: return
        val monster = choice.monster
        val healthBefore = state.health

        val newState: GameState
        val weaponUsed: Card?
        val damageBlocked: Int
        val damageTaken: Int

        if (useWeapon) {
            newState = state.fightMonsterWithWeapon(monster)
            weaponUsed = choice.weapon
            damageBlocked = choice.weapon.value.coerceAtMost(monster.value)
            damageTaken = choice.weaponDamage
        } else {
            newState = state.fightMonsterBarehanded(monster)
            weaponUsed = null
            damageBlocked = 0
            damageTaken = choice.barehandedDamage
        }

        actionLogEntries.add(
            LogEntry.MonsterFought(
                timestamp = System.currentTimeMillis(),
                monster = monster,
                weaponUsed = weaponUsed,
                damageBlocked = damageBlocked,
                damageTaken = damageTaken,
                healthBefore = healthBefore,
                healthAfter = newState.health,
            ),
        )

        // Clear the combat choice and continue processing
        processingState = newState
        pendingCardsToProcess.removeAt(0)
        _uiState.value =
            _uiState.value.copy(
                pendingCombatChoice = null,
                actionLog = actionLogEntries.toList(),
            )

        processNextCard()
    }

    private suspend fun handleGameEnded(
        score: Int,
        won: Boolean,
    ) {
        highScoreRepository?.saveScore(score = score, won = won)
        // Reload highest score after saving
        highestScore = highScoreRepository?.getHighestScore()
        updateUiStateWithHighScore()
    }

    private fun handleShowHelp() {
        _uiState.value = _uiState.value.copy(showHelp = true)
    }

    private fun handleHideHelp() {
        _uiState.value = _uiState.value.copy(showHelp = false)
    }

    private fun handleShowActionLog() {
        _uiState.value = _uiState.value.copy(showActionLog = true)
    }

    private fun handleHideActionLog() {
        _uiState.value = _uiState.value.copy(showActionLog = false)
    }

    /**
     * Simulates processing the selected cards and returns the log entries
     * that would be generated, without mutating any actual state.
     * Used for preview display before committing to card selection.
     */
    fun simulateProcessing(selectedCards: List<Card>): List<LogEntry> {
        if (selectedCards.isEmpty()) return emptyList()

        val previewEntries = mutableListOf<LogEntry>()
        var state = gameState.value

        selectedCards.forEach { card ->
            val healthBefore = state.health
            val weaponBefore = state.weaponState?.weapon
            val usedPotionBefore = state.usedPotionThisTurn

            val currentWeaponState = state.weaponState
            state =
                when (card.type) {
                    CardType.MONSTER -> {
                        // Compute weapon-related values based on whether we can use the weapon
                        val (weaponUsed, damageBlocked, damageTaken) =
                            if (currentWeaponState != null && currentWeaponState.canDefeat(card)) {
                                Triple(
                                    currentWeaponState.weapon,
                                    currentWeaponState.weapon.value.coerceAtMost(card.value),
                                    (card.value - currentWeaponState.weapon.value).coerceAtLeast(0),
                                )
                            } else {
                                Triple(null, 0, card.value)
                            }

                        val newState = state.fightMonster(card)

                        previewEntries.add(
                            LogEntry.MonsterFought(
                                timestamp = 0L,
                                monster = card,
                                weaponUsed = weaponUsed,
                                damageBlocked = damageBlocked,
                                damageTaken = damageTaken,
                                healthBefore = healthBefore,
                                healthAfter = newState.health,
                            ),
                        )
                        newState
                    }
                    CardType.WEAPON -> {
                        val newState = state.equipWeapon(card)
                        previewEntries.add(
                            LogEntry.WeaponEquipped(
                                timestamp = 0L,
                                weapon = card,
                                replacedWeapon = weaponBefore,
                            ),
                        )
                        newState
                    }
                    CardType.POTION -> {
                        val wasDiscarded = usedPotionBefore
                        val newState = state.usePotion(card)
                        val healthRestored = if (wasDiscarded) 0 else newState.health - healthBefore

                        previewEntries.add(
                            LogEntry.PotionUsed(
                                timestamp = 0L,
                                potion = card,
                                healthRestored = healthRestored,
                                healthBefore = healthBefore,
                                healthAfter = newState.health,
                                wasDiscarded = wasDiscarded,
                            ),
                        )
                        newState
                    }
                }
        }

        return previewEntries
    }

    private fun updateGameState(newState: GameState) {
        mutableGameState.value = newState
        // Update UI state immediately with cached high score info
        // isNewHighScore: true if no scores exist OR current score beats highest
        val currentScore = newState.calculateScore()
        val currentShowHelp = _uiState.value.showHelp
        val currentShowActionLog = _uiState.value.showActionLog
        _uiState.value =
            newState.toUiState().copy(
                highestScore = highestScore,
                isNewHighScore = highestScore?.let { currentScore > it } ?: (highScoreRepository != null),
                showHelp = currentShowHelp,
                showActionLog = currentShowActionLog,
                actionLog = actionLogEntries.toList(),
            )
    }

    /**
     * Converts GameState to GameUiState for UI consumption.
     */
    private fun GameState.toUiState(): GameUiState =
        GameUiState(
            health = health,
            deckSize = deck.cards.size,
            currentRoom = currentRoom,
            weaponState = weaponState,
            defeatedMonstersCount = defeatedMonsters.size,
            score = calculateScore(),
            isGameOver = isGameOver,
            isGameWon = isGameWon,
            lastRoomAvoided = lastRoomAvoided,
            canAvoidRoom = currentRoom != null && !lastRoomAvoided,
            gameSeed = currentGameSeed,
        )
}
