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

/**
 * ViewModel for the game screen, managing game state and handling user intents.
 * Follows MVI (Model-View-Intent) pattern.
 */
class GameViewModel(
    private val highScoreRepository: HighScoreRepository? = null,
) : ViewModel() {
    private val _gameState = MutableStateFlow(GameState.newGame())
    private val gameState: StateFlow<GameState> = _gameState.asStateFlow()

    private val _uiState =
        MutableStateFlow(
            GameUiState(
                health = 20,
                deckSize = 44,
                currentRoom = null,
                weaponState = null,
                defeatedMonstersCount = 0,
                score = 20,
                isGameOver = false,
                isGameWon = false,
                lastRoomAvoided = false,
                canAvoidRoom = false,
                actionLog = listOf(LogEntry.GameStarted(timestamp = System.currentTimeMillis())),
            ),
        )
    val uiState: StateFlow<GameUiState> = _uiState.asStateFlow()

    private var highestScore: Int? = null
    private val actionLogEntries = mutableListOf<LogEntry>(LogEntry.GameStarted(timestamp = System.currentTimeMillis()))

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
        _uiState.value =
            gameState.value.toUiState().copy(
                highestScore = highestScore,
                isNewHighScore = isNewHigh,
                showHelp = currentShowHelp,
                showActionLog = currentShowActionLog,
                actionLog = actionLogEntries.toList(),
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
            }
        }
    }

    private fun handleNewGame() {
        actionLogEntries.clear()
        actionLogEntries.add(LogEntry.GameStarted(timestamp = System.currentTimeMillis()))
        updateGameState(GameState.newGame())
    }

    private fun handleDrawRoom() {
        val stateBefore = gameState.value
        val stateAfter = stateBefore.drawRoom()
        val cardsDrawn = if (stateBefore.currentRoom == null) 4 else 3
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
        updateGameState(stateBefore.avoidRoom())
    }

    private fun handleProcessSelectedCards(selectedCards: List<Card>) {
        // First, select the cards (leaves unselected card for next room)
        var state = gameState.value.selectCards(selectedCards)

        // Then process each selected card, generating log entries
        selectedCards.forEach { card ->
            val healthBefore = state.health
            val weaponBefore = state.weaponState?.weapon
            val usedPotionBefore = state.usedPotionThisTurn

            state =
                when (card.type) {
                    CardType.MONSTER -> {
                        val canUseWeapon = state.weaponState?.canDefeat(card) == true
                        val weaponUsed = if (canUseWeapon) state.weaponState?.weapon else null
                        val damageBlocked =
                            if (canUseWeapon) {
                                state.weaponState!!.weapon.value.coerceAtMost(card.value)
                            } else {
                                0
                            }
                        val damageTaken =
                            if (canUseWeapon) {
                                (card.value - state.weaponState!!.weapon.value).coerceAtLeast(0)
                            } else {
                                card.value
                            }

                        val newState = state.fightMonster(card)

                        actionLogEntries.add(
                            LogEntry.MonsterFought(
                                timestamp = System.currentTimeMillis(),
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
                        actionLogEntries.add(
                            LogEntry.WeaponEquipped(
                                timestamp = System.currentTimeMillis(),
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
                        newState
                    }
                }
        }

        updateGameState(state)
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

    private fun updateGameState(newState: GameState) {
        _gameState.value = newState
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
    private fun GameState.toUiState(): GameUiState {
        return GameUiState(
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
        )
    }
}
