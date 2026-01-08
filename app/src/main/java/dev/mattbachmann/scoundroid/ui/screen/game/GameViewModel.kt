package dev.mattbachmann.scoundroid.ui.screen.game

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * ViewModel for the game screen, managing game state and handling user intents.
 * Follows MVI (Model-View-Intent) pattern.
 */
class GameViewModel : ViewModel() {
    private val _gameState = MutableStateFlow(GameState.newGame())
    private val gameState: StateFlow<GameState> = _gameState.asStateFlow()

    private val _uiState = MutableStateFlow(gameState.value.toUiState())
    val uiState: StateFlow<GameUiState> = _uiState.asStateFlow()

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
            }
        }
    }

    private fun handleNewGame() {
        updateGameState(GameState.newGame())
    }

    private fun handleDrawRoom() {
        updateGameState(gameState.value.drawRoom())
    }

    private fun handleAvoidRoom() {
        updateGameState(gameState.value.avoidRoom())
    }

    private fun handleProcessSelectedCards(selectedCards: List<dev.mattbachmann.scoundroid.data.model.Card>) {
        // First, select the cards (leaves unselected card for next room)
        var state = gameState.value.selectCards(selectedCards)

        // Then process each selected card
        selectedCards.forEach { card ->
            state =
                when (card.type) {
                    CardType.MONSTER -> state.fightMonster(card)
                    CardType.WEAPON -> state.equipWeapon(card)
                    CardType.POTION -> state.usePotion(card)
                }
        }

        updateGameState(state)
    }

    private fun updateGameState(newState: GameState) {
        _gameState.value = newState
        _uiState.value = newState.toUiState()
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
