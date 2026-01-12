package dev.mattbachmann.scoundroid.ui.screen.game

import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import dev.mattbachmann.scoundroid.data.repository.HighScoreRepository
import dev.mattbachmann.scoundroid.data.repository.WinningGameRepository

class GameViewModelFactory(
    private val highScoreRepository: HighScoreRepository,
    private val winningGameRepository: WinningGameRepository,
) : ViewModelProvider.Factory {
    @Suppress("UNCHECKED_CAST")
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        if (modelClass.isAssignableFrom(GameViewModel::class.java)) {
            return GameViewModel(highScoreRepository, winningGameRepository) as T
        }
        throw IllegalArgumentException("Unknown ViewModel class")
    }
}
