package dev.mattbachmann.scoundroid.ui.snapshot

import app.cash.paparazzi.Paparazzi
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.ui.component.RoomDisplay
import dev.mattbachmann.scoundroid.ui.screen.game.ScreenSizeClass
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import org.junit.Rule
import org.junit.Test

private val testRoom =
    listOf(
        Card(Suit.CLUBS, Rank.QUEEN), // Monster
        Card(Suit.DIAMONDS, Rank.FIVE), // Weapon
        Card(Suit.HEARTS, Rank.SEVEN), // Potion
        Card(Suit.SPADES, Rank.TEN), // Monster
    )

/**
 * Snapshot tests for RoomDisplay with COMPACT layout (Galaxy A01 - small phone).
 */
class RoomDisplayCompactSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.GALAXY_A01,
        )

    @Test
    fun roomDisplay_placeholders() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = emptyList(),
                    selectedCards = emptyList(),
                    onCardClick = null,
                    screenSizeClass = ScreenSizeClass.COMPACT,
                    showPlaceholders = true,
                )
            }
        }
    }

    @Test
    fun roomDisplay_fullRoom() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = emptyList(),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.COMPACT,
                )
            }
        }
    }

    @Test
    fun roomDisplay_withSelections() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = listOf(testRoom[0], testRoom[2], testRoom[3]),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.COMPACT,
                )
            }
        }
    }

    @Test
    fun roomDisplay_leftoverCard() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = listOf(testRoom[1]),
                    selectedCards = emptyList(),
                    onCardClick = null,
                    screenSizeClass = ScreenSizeClass.COMPACT,
                )
            }
        }
    }
}

/**
 * Snapshot tests for RoomDisplay with MEDIUM layout (Pixel 7 - regular phone).
 */
class RoomDisplayMediumSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PIXEL_7,
        )

    @Test
    fun roomDisplay_placeholders() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = emptyList(),
                    selectedCards = emptyList(),
                    onCardClick = null,
                    screenSizeClass = ScreenSizeClass.MEDIUM,
                    showPlaceholders = true,
                )
            }
        }
    }

    @Test
    fun roomDisplay_fullRoom() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = emptyList(),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.MEDIUM,
                )
            }
        }
    }

    @Test
    fun roomDisplay_withSelections() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = listOf(testRoom[0], testRoom[2], testRoom[3]),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.MEDIUM,
                )
            }
        }
    }
}

/**
 * Snapshot tests for RoomDisplay with LANDSCAPE layout (phone rotated).
 */
class RoomDisplayLandscapeSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PHONE_LANDSCAPE,
        )

    @Test
    fun roomDisplay_fullRoom() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = emptyList(),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.LANDSCAPE,
                )
            }
        }
    }

    @Test
    fun roomDisplay_withSelections() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = listOf(testRoom[0], testRoom[2], testRoom[3]),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.LANDSCAPE,
                )
            }
        }
    }
}

/**
 * Snapshot tests for RoomDisplay with TABLET layout (unfolded foldable).
 */
class RoomDisplayTabletSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PIXEL_FOLD_UNFOLDED,
        )

    @Test
    fun roomDisplay_placeholders() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = emptyList(),
                    selectedCards = emptyList(),
                    onCardClick = null,
                    screenSizeClass = ScreenSizeClass.TABLET,
                    showPlaceholders = true,
                )
            }
        }
    }

    @Test
    fun roomDisplay_fullRoom() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = emptyList(),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.TABLET,
                )
            }
        }
    }

    @Test
    fun roomDisplay_withSelections() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = listOf(testRoom[0], testRoom[2], testRoom[3]),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.TABLET,
                )
            }
        }
    }
}

/**
 * Snapshot tests for RoomDisplay with TABLET_PORTRAIT layout.
 */
class RoomDisplayTabletPortraitSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.TABLET_PORTRAIT,
        )

    @Test
    fun roomDisplay_fullRoom() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = emptyList(),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.TABLET_PORTRAIT,
                )
            }
        }
    }

    @Test
    fun roomDisplay_withSelections() {
        paparazzi.snapshot {
            ScoundroidTheme {
                RoomDisplay(
                    cards = testRoom,
                    selectedCards = listOf(testRoom[0], testRoom[2], testRoom[3]),
                    onCardClick = {},
                    screenSizeClass = ScreenSizeClass.TABLET_PORTRAIT,
                )
            }
        }
    }
}
