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
import dev.mattbachmann.scoundroid.data.repository.WinningGameRepository
import dev.mattbachmann.scoundroid.ui.screen.game.GameScreen
import dev.mattbachmann.scoundroid.ui.screen.game.GameViewModelFactory
import dev.mattbachmann.scoundroid.ui.screen.game.ScreenSizeClass
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

class MainActivity : ComponentActivity() {
    private val database by lazy { AppDatabase.getDatabase(this) }
    private val highScoreRepository by lazy { HighScoreRepository(database.highScoreDao()) }
    private val winningGameRepository by lazy { WinningGameRepository(database.winningGameDao()) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            val screenSizeClass = getScreenSizeClass()
            ScoundroidTheme {
                GameScreen(
                    viewModel =
                        viewModel(
                            factory = GameViewModelFactory(highScoreRepository, winningGameRepository),
                        ),
                    screenSizeClass = screenSizeClass,
                )
            }
        }
    }
}

/**
 * Determines the screen size class based on screen dimensions and orientation.
 * - COMPACT: Small phones (height < 780dp) - aggressive space saving
 * - MEDIUM: Fold cover screens, regular phones (height >= 780dp, width < height)
 * - LANDSCAPE: Phones in landscape (width > height, height < 500dp) - compact horizontal layout
 * - TABLET: Unfolded foldables (nearly-square screens), tablets in landscape - spacious two-column layout
 * - TABLET_PORTRAIT: Tablets in portrait (non-square) - vertical layout with large elements
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
    val aspectRatio = maxDimension / minDimension
    val isNearlySquare = aspectRatio < 1.1f // Less than 10% difference between dimensions

    return when {
        // Landscape phones (wider than tall with limited height - includes folded phone landscape)
        // Must check this BEFORE tablet to catch folded landscape correctly
        widthDp.value > heightDp.value && heightDp.value < 500 -> ScreenSizeClass.LANDSCAPE

        // Nearly-square large screens (like foldable inner displays) - use TABLET layout regardless of orientation
        isLargeScreen && isNearlySquare -> ScreenSizeClass.TABLET

        // Large screens in landscape (tablets, unfolded foldables)
        isLargeScreen && widthDp.value > heightDp.value -> ScreenSizeClass.TABLET

        // Large screens in portrait (tablets, unfolded foldables)
        isLargeScreen && heightDp.value >= widthDp.value -> ScreenSizeClass.TABLET_PORTRAIT

        // Landscape phones (wider than tall, moderate height)
        widthDp.value > heightDp.value -> ScreenSizeClass.LANDSCAPE

        // Small portrait phones (includes budget phones like Galaxy A01 at ~760dp)
        heightDp.value < 780 -> ScreenSizeClass.COMPACT

        // Regular portrait phones
        else -> ScreenSizeClass.MEDIUM
    }
}
