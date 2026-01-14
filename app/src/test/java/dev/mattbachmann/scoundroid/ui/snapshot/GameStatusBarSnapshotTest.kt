package dev.mattbachmann.scoundroid.ui.snapshot

import app.cash.paparazzi.Paparazzi
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.data.model.WeaponState
import dev.mattbachmann.scoundroid.ui.component.GameStatusBar
import dev.mattbachmann.scoundroid.ui.component.StatusBarLayout
import dev.mattbachmann.scoundroid.ui.theme.ScoundroidTheme
import org.junit.Rule
import org.junit.Test

/**
 * Snapshot tests for GameStatusBar with COMPACT layout (small phones like Galaxy A01).
 */
class GameStatusBarCompactSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.GALAXY_A01,
        )

    @Test
    fun statusBar_noWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 20,
                    score = 20,
                    deckSize = 44,
                    weaponState = null,
                    defeatedMonstersCount = 0,
                    layout = StatusBarLayout.COMPACT,
                )
            }
        }
    }

    @Test
    fun statusBar_withFreshWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 15,
                    score = 15,
                    deckSize = 30,
                    weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.SEVEN)),
                    defeatedMonstersCount = 5,
                    layout = StatusBarLayout.COMPACT,
                )
            }
        }
    }

    @Test
    fun statusBar_withDegradedWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 10,
                    score = 10,
                    deckSize = 20,
                    weaponState =
                        WeaponState(
                            weapon = Card(Suit.DIAMONDS, Rank.NINE),
                            maxMonsterValue = 6,
                        ),
                    defeatedMonstersCount = 10,
                    layout = StatusBarLayout.COMPACT,
                )
            }
        }
    }

    @Test
    fun statusBar_lowHealth() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 3,
                    score = 3,
                    deckSize = 10,
                    weaponState = null,
                    defeatedMonstersCount = 15,
                    layout = StatusBarLayout.COMPACT,
                )
            }
        }
    }
}

/**
 * Snapshot tests for GameStatusBar with MEDIUM layout (regular phones).
 */
class GameStatusBarMediumSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PIXEL_7,
        )

    @Test
    fun statusBar_noWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 20,
                    score = 20,
                    deckSize = 44,
                    weaponState = null,
                    defeatedMonstersCount = 0,
                    layout = StatusBarLayout.MEDIUM,
                )
            }
        }
    }

    @Test
    fun statusBar_withFreshWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 15,
                    score = 15,
                    deckSize = 30,
                    weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.SEVEN)),
                    defeatedMonstersCount = 5,
                    layout = StatusBarLayout.MEDIUM,
                )
            }
        }
    }

    @Test
    fun statusBar_withDegradedWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 10,
                    score = 10,
                    deckSize = 20,
                    weaponState =
                        WeaponState(
                            weapon = Card(Suit.DIAMONDS, Rank.NINE),
                            maxMonsterValue = 6,
                        ),
                    defeatedMonstersCount = 10,
                    layout = StatusBarLayout.MEDIUM,
                )
            }
        }
    }
}

/**
 * Snapshot tests for GameStatusBar with INLINE layout (tablets/landscape).
 */
class GameStatusBarInlineSnapshotTest {
    @get:Rule
    val paparazzi =
        Paparazzi(
            deviceConfig = DeviceConfigs.PIXEL_FOLD_UNFOLDED,
        )

    @Test
    fun statusBar_noWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 20,
                    score = 20,
                    deckSize = 44,
                    weaponState = null,
                    defeatedMonstersCount = 0,
                    layout = StatusBarLayout.INLINE,
                )
            }
        }
    }

    @Test
    fun statusBar_withFreshWeapon() {
        paparazzi.snapshot {
            ScoundroidTheme {
                GameStatusBar(
                    health = 15,
                    score = 15,
                    deckSize = 30,
                    weaponState = WeaponState(Card(Suit.DIAMONDS, Rank.SEVEN)),
                    defeatedMonstersCount = 5,
                    layout = StatusBarLayout.INLINE,
                )
            }
        }
    }
}
