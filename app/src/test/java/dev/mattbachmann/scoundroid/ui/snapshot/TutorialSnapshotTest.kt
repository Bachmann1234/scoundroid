package dev.mattbachmann.scoundroid.ui.snapshot

import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import app.cash.paparazzi.Paparazzi
import dev.mattbachmann.scoundroid.ui.component.TutorialContent
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import org.junit.Rule
import org.junit.Test

/**
 * Snapshot tests for TutorialContent in portrait mode.
 * Tests the tutorial slideshow renders correctly on a regular phone.
 */
class TutorialSnapshotPortraitTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PIXEL_7,
        )

    @Test
    fun tutorial_portrait() {
        paparazzi.snapshot {
            ScoundroidTheme {
                TutorialContent(
                    onDismiss = {},
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
    }
}

/**
 * Snapshot tests for TutorialContent in landscape mode.
 * Tests the landscape-optimized layout renders correctly.
 */
class TutorialSnapshotLandscapeTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PHONE_LANDSCAPE,
        )

    @Test
    fun tutorial_landscape() {
        paparazzi.snapshot {
            ScoundroidTheme {
                TutorialContent(
                    onDismiss = {},
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
    }
}

/**
 * Snapshot tests for TutorialContent on compact device.
 * Tests tutorial renders on small phones.
 */
class TutorialSnapshotCompactTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.GALAXY_A01,
        )

    @Test
    fun tutorial_compact() {
        paparazzi.snapshot {
            ScoundroidTheme {
                TutorialContent(
                    onDismiss = {},
                    modifier = Modifier.fillMaxSize(),
                )
            }
        }
    }
}
