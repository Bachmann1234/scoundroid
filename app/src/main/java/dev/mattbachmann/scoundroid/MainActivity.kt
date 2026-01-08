package dev.mattbachmann.scoundroid

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import dev.mattbachmann.scoundroid.ui.screen.game.GameScreen
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            ScoundroidTheme {
                GameScreen()
            }
        }
    }
}
