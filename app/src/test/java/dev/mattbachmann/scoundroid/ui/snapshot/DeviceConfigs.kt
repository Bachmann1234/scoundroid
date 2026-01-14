package dev.mattbachmann.scoundroid.ui.snapshot

import app.cash.paparazzi.DeviceConfig
import com.android.resources.Density
import com.android.resources.ScreenOrientation

/**
 * Device configurations for snapshot testing.
 * These match the ScreenSizeClass breakpoints defined in MainActivity.kt:
 * - COMPACT: height < 780dp
 * - MEDIUM: height >= 780dp, portrait
 * - LANDSCAPE: width > height, height < 500dp
 * - TABLET: large screen (min >= 550dp, max >= 800dp), landscape or nearly-square
 * - TABLET_PORTRAIT: large screen, portrait
 */
object DeviceConfigs {
    /**
     * Samsung Galaxy A01 - Budget small phone
     * 720x1520 @ 320dpi = 360x760dp -> COMPACT (760 < 780)
     */
    val GALAXY_A01 =
        DeviceConfig(
            screenWidth = 720,
            screenHeight = 1520,
            density = Density.XHIGH, // 320 dpi
            orientation = ScreenOrientation.PORTRAIT,
        )

    /**
     * Pixel 7 - Regular phone
     * 1080x2400 @ 420dpi = 411x914dp -> MEDIUM (914 >= 780)
     */
    val PIXEL_7 =
        DeviceConfig(
            screenWidth = 1080,
            screenHeight = 2400,
            density = Density.create(420),
            orientation = ScreenOrientation.PORTRAIT,
        )

    /**
     * Pixel 10 Pro Fold - Cover screen (folded)
     * 1080x2092 @ 420dpi = ~411x796dp -> MEDIUM
     */
    val PIXEL_FOLD_COVER =
        DeviceConfig(
            screenWidth = 1080,
            screenHeight = 2092,
            density = Density.create(420),
            orientation = ScreenOrientation.PORTRAIT,
        )

    /**
     * Pixel 10 Pro Fold - Inner screen (unfolded)
     * 2208x1840 @ 420dpi = ~840x700dp -> TABLET (nearly square, large)
     */
    val PIXEL_FOLD_UNFOLDED =
        DeviceConfig(
            screenWidth = 2208,
            screenHeight = 1840,
            density = Density.create(420),
            orientation = ScreenOrientation.LANDSCAPE,
        )

    /**
     * Regular phone in landscape
     * 1080x2400 @ 420dpi rotated = 914x411dp -> LANDSCAPE (411 < 500)
     */
    val PHONE_LANDSCAPE =
        DeviceConfig(
            screenWidth = 2400,
            screenHeight = 1080,
            density = Density.create(420),
            orientation = ScreenOrientation.LANDSCAPE,
        )

    /**
     * Tablet in portrait mode
     * 1600x2560 @ 320dpi = 800x1280dp -> TABLET_PORTRAIT
     */
    val TABLET_PORTRAIT =
        DeviceConfig(
            screenWidth = 1600,
            screenHeight = 2560,
            density = Density.XHIGH,
            orientation = ScreenOrientation.PORTRAIT,
        )

    /**
     * All device configs for parameterized testing
     */
    val ALL_DEVICES =
        listOf(
            "Galaxy_A01_COMPACT" to GALAXY_A01,
            "Pixel_7_MEDIUM" to PIXEL_7,
            "Pixel_Fold_Cover_MEDIUM" to PIXEL_FOLD_COVER,
            "Pixel_Fold_Unfolded_TABLET" to PIXEL_FOLD_UNFOLDED,
            "Phone_LANDSCAPE" to PHONE_LANDSCAPE,
            "Tablet_PORTRAIT" to TABLET_PORTRAIT,
        )
}
