package dev.mattbachmann.scoundroid

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalConfiguration
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.mattbachmann.scoundroid.data.persistence.AppDatabase
import dev.mattbachmann.scoundroid.data.repository.HighScoreRepository
import dev.mattbachmann.scoundroid.ui.screen.game.GameScreen
import dev.mattbachmann.scoundroid.ui.screen.game.GameViewModelFactory
import dev.mattbachmann.scoundroid.ui.screen.game.ScreenSizeClass
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

class MainActivity : ComponentActivity() {
    private val database by lazy { AppDatabase.getDatabase(this) }
    private val repository by lazy { HighScoreRepository(database.highScoreDao()) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val screenSizeClass = getScreenSizeClass()
            ScoundroidTheme {
                GameScreen(
                    viewModel = viewModel(factory = GameViewModelFactory(repository)),
                    screenSizeClass = screenSizeClass,
                )
            }
        }
    }
}

/**
 * Determines the screen size class based on screen dimensions.
 * - COMPACT: Small phones (height < 700dp) - aggressive space saving
 * - MEDIUM: Fold cover screens, regular phones (height >= 700dp, width < 600dp)
 * - EXPANDED: Tablets, unfolded foldables (width >= 600dp)
 */
@Composable
private fun getScreenSizeClass(): ScreenSizeClass {
    val configuration = LocalConfiguration.current
    return when {
        configuration.screenWidthDp >= 600 -> ScreenSizeClass.EXPANDED
        configuration.screenHeightDp < 700 -> ScreenSizeClass.COMPACT
        else -> ScreenSizeClass.MEDIUM
    }
}
