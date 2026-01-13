package dev.mattbachmann.scoundroid

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.platform.LocalWindowInfo
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
 * Determines the screen size class based on screen dimensions and orientation.
 * - COMPACT: Small phones (height < 700dp) - aggressive space saving
 * - MEDIUM: Fold cover screens, regular phones (height >= 700dp, width < height)
 * - LANDSCAPE: Phones in landscape (width > height, height < 500dp) - compact horizontal layout
 * - TABLET: Unfolded foldables, tablets in landscape - spacious two-column layout
 * - TABLET_PORTRAIT: Unfolded foldables, tablets in portrait - vertical layout with large elements
 */
@Composable
private fun getScreenSizeClass(): ScreenSizeClass {
    val windowInfo = LocalWindowInfo.current
    val density = LocalDensity.current
    val containerSize = windowInfo.containerSize
    val widthDp = with(density) { containerSize.width.toDp() }
    val heightDp = with(density) { containerSize.height.toDp() }

    val minDimension = minOf(widthDp.value, heightDp.value)
    val maxDimension = maxOf(widthDp.value, heightDp.value)
    val isLargeScreen = minDimension >= 550 && maxDimension >= 800

    return when {
        // Landscape phones (wider than tall with limited height - includes folded phone landscape)
        // Must check this BEFORE tablet to catch folded landscape correctly
        widthDp.value > heightDp.value && heightDp.value < 500 -> ScreenSizeClass.LANDSCAPE

        // Large screens in landscape (tablets, unfolded foldables)
        isLargeScreen && widthDp.value > heightDp.value -> ScreenSizeClass.TABLET

        // Large screens in portrait (tablets, unfolded foldables)
        isLargeScreen && heightDp.value >= widthDp.value -> ScreenSizeClass.TABLET_PORTRAIT

        // Landscape phones (wider than tall, moderate height)
        widthDp.value > heightDp.value -> ScreenSizeClass.LANDSCAPE

        // Small portrait phones
        heightDp.value < 700 -> ScreenSizeClass.COMPACT

        // Regular portrait phones
        else -> ScreenSizeClass.MEDIUM
    }
}
