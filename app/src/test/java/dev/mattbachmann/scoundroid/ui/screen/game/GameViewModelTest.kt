package dev.mattbachmann.scoundroid.ui.screen.game

import app.cash.turbine.test
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Before
import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

@OptIn(ExperimentalCoroutinesApi::class)
class GameViewModelTest {
    private val testDispatcher = StandardTestDispatcher()

    @Before
    fun setup() {
        Dispatchers.setMain(testDispatcher)
    }

    @After
    fun tearDown() {
        Dispatchers.resetMain()
    }

    // Test helper functions
    private fun testMonster(value: Int) = Card(Suit.CLUBS, Rank.fromValue(value))

    private fun testWeapon(value: Int) = Card(Suit.DIAMONDS, Rank.fromValue(value))

    private fun testPotion(value: Int) = Card(Suit.HEARTS, Rank.fromValue(value))

    // ========== Initialization Tests ==========

    @Test
    fun `initial state starts new game with 20 health`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(20, state.health)
                assertNull(state.currentRoom)
                assertNull(state.weaponState)
                assertFalse(state.isGameOver)
                assertFalse(state.isGameWon)
            }
        }

    @Test
    fun `initial state has full 44-card deck`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(44, state.deckSize)
            }
        }

    @Test
    fun `initial state has no room drawn`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                val state = awaitItem()
                assertNull(state.currentRoom)
            }
        }

    // ========== Room Drawing Tests ==========

    @Test
    fun `drawRoom intent draws 4 cards from deck`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertNotNull(state.currentRoom)
                assertEquals(4, state.currentRoom!!.size)
                assertEquals(40, state.deckSize) // 44 - 4 = 40
            }
        }

    @Test
    fun `drawRoom after card selection draws 3 more cards`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                // First room
                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                val state = awaitItem()
                val cardsToSelect = state.currentRoom!!.take(3)

                // Select 3 cards, leaving 1
                viewModel.onIntent(GameIntent.SelectCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()
                awaitItem()

                // Draw next room (should draw 3 more + 1 remaining = 4 total)
                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()

                val newState = awaitItem()
                assertEquals(4, newState.currentRoom!!.size)
            }
        }

    // ========== Room Avoidance Tests ==========

    @Test
    fun `avoidRoom intent moves all 4 cards to bottom of deck`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            val initialDeckSize = 40 // 44 - 4 cards drawn

            viewModel.onIntent(GameIntent.AvoidRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertNull(state.currentRoom)
                assertEquals(44, state.deckSize) // Cards returned to deck
                assertTrue(state.lastRoomAvoided)
            }
        }

    @Test
    fun `cannot avoid room twice in a row`() =
        runTest {
            val viewModel = GameViewModel()

            // Draw and avoid first room
            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()
            viewModel.onIntent(GameIntent.AvoidRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            // Draw second room
            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertFalse(state.canAvoidRoom)
            }
        }

    @Test
    fun `can avoid room after processing previous room`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                // Draw first room
                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                val state = awaitItem()
                assertTrue(state.canAvoidRoom)

                // Select cards (process room)
                val cardsToSelect = state.currentRoom!!.take(3)
                viewModel.onIntent(GameIntent.SelectCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()
                awaitItem()

                // Draw next room
                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()

                // Should be able to avoid this room
                val newState = awaitItem()
                assertTrue(newState.canAvoidRoom)
            }
        }

    // ========== Card Selection Tests ==========

    @Test
    fun `selectCards intent selects 3 of 4 cards`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                val cardsToSelect = state.currentRoom!!.take(3)

                viewModel.onIntent(GameIntent.SelectCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()

                val newState = awaitItem()
                assertEquals(1, newState.currentRoom!!.size) // 1 card left for next room
            }
        }

    // ========== Card Processing Tests ==========

    @Test
    fun `processCard with monster reduces health`() =
        runTest {
            val viewModel = GameViewModel()
            val monster = testMonster(10)

            viewModel.onIntent(GameIntent.ProcessCard(monster))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(10, state.health) // 20 - 10 = 10
            }
        }

    @Test
    fun `processCard with weapon equips it`() =
        runTest {
            val viewModel = GameViewModel()
            val weapon = testWeapon(5)

            viewModel.onIntent(GameIntent.ProcessCard(weapon))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertNotNull(state.weaponState)
                assertEquals(5, state.weaponState!!.weapon.value)
            }
        }

    @Test
    fun `processCard with potion restores health`() =
        runTest {
            val viewModel = GameViewModel()

            // First take damage
            viewModel.onIntent(GameIntent.ProcessCard(testMonster(10)))
            testDispatcher.scheduler.advanceUntilIdle()

            // Then use potion
            viewModel.onIntent(GameIntent.ProcessCard(testPotion(7)))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(17, state.health) // 10 + 7 = 17
            }
        }

    @Test
    fun `processCard with weapon reduces monster damage`() =
        runTest {
            val viewModel = GameViewModel()

            // Equip weapon first
            viewModel.onIntent(GameIntent.ProcessCard(testWeapon(5)))
            testDispatcher.scheduler.advanceUntilIdle()

            // Fight monster
            viewModel.onIntent(GameIntent.ProcessCard(testMonster(8)))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(17, state.health) // 20 - (8 - 5) = 17
            }
        }

    // ========== Game Over Tests ==========

    @Test
    fun `game over when health reaches 0`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                // Deal damage to reduce health
                viewModel.onIntent(GameIntent.ProcessCard(testMonster(14))) // Ace
                testDispatcher.scheduler.advanceUntilIdle()
                awaitItem()

                // Deal lethal damage (already at 6, need 6 more to reach 0)
                viewModel.onIntent(GameIntent.ProcessCard(testMonster(6)))
                testDispatcher.scheduler.advanceUntilIdle()

                val state = awaitItem()
                assertEquals(0, state.health)
                assertTrue(state.isGameOver)
                assertFalse(state.isGameWon)
            }
        }

    @Test
    fun `game won when deck is empty with health greater than 0`() =
        runTest {
            // This test would require processing all 44 cards, which is impractical
            // Instead, we'll test the logic by directly checking the state calculation
            val gameState =
                GameState.newGame().copy(
                    deck = dev.mattbachmann.scoundroid.data.model.Deck(emptyList()),
                    health = 15,
                )

            assertTrue(gameState.isGameWon)
            assertFalse(gameState.isGameOver)
        }

    // ========== Scoring Tests ==========

    @Test
    fun `score reflects current health when alive`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.onIntent(GameIntent.ProcessCard(testMonster(7)))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(13, state.health)
                assertEquals(13, state.score)
            }
        }

    @Test
    fun `score is negative when dead with remaining monsters`() =
        runTest {
            val gameState =
                GameState.newGame().copy(
                    health = 0,
                    // Remaining monsters sum to 50
                    deck =
                        dev.mattbachmann.scoundroid.data.model.Deck(
                            listOf(
                                testMonster(10),
                                testMonster(10),
                                testMonster(10),
                                testMonster(10),
                                testMonster(10),
                            ),
                        ),
                )

            assertEquals(-50, gameState.calculateScore())
        }

    // ========== UI State Mapping Tests ==========

    @Test
    fun `uiState exposes all necessary game information`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                val state = awaitItem()

                // Verify all UI-necessary fields are present
                assertNotNull(state.health)
                assertNotNull(state.deckSize)
                assertNotNull(state.defeatedMonstersCount)
                assertNotNull(state.score)
                assertNotNull(state.isGameOver)
                assertNotNull(state.isGameWon)
                assertNotNull(state.canAvoidRoom)
            }
        }

    @Test
    fun `defeated monsters count increases when monster is defeated`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.onIntent(GameIntent.ProcessCard(testMonster(5)))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(1, state.defeatedMonstersCount)
            }
        }

    // ========== New Game Tests ==========

    @Test
    fun `newGame intent resets game to initial state`() =
        runTest {
            val viewModel = GameViewModel()

            // Play some of the game
            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()
            viewModel.onIntent(GameIntent.ProcessCard(testMonster(10)))
            testDispatcher.scheduler.advanceUntilIdle()

            // Start new game
            viewModel.onIntent(GameIntent.NewGame)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(20, state.health)
                assertEquals(44, state.deckSize)
                assertNull(state.currentRoom)
                assertNull(state.weaponState)
                assertEquals(0, state.defeatedMonstersCount)
            }
        }

    // ========== Edge Cases ==========

    @Test
    fun `weapon degradation is tracked correctly`() =
        runTest {
            val viewModel = GameViewModel()

            // Equip weapon
            viewModel.onIntent(GameIntent.ProcessCard(testWeapon(5)))
            testDispatcher.scheduler.advanceUntilIdle()

            // Fight high-value monster
            viewModel.onIntent(GameIntent.ProcessCard(testMonster(12)))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertNotNull(state.weaponState)
                assertEquals(12, state.weaponState!!.maxMonsterValue)
            }
        }

    @Test
    fun `potion cannot exceed max health`() =
        runTest {
            val viewModel = GameViewModel()

            // Use potion at full health
            viewModel.onIntent(GameIntent.ProcessCard(testPotion(10)))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(20, state.health) // Still 20, not 30
            }
        }

    @Test
    fun `only one potion per turn is applied`() =
        runTest {
            val viewModel = GameViewModel()

            // Take damage first
            viewModel.onIntent(GameIntent.ProcessCard(testMonster(10)))
            testDispatcher.scheduler.advanceUntilIdle()

            // Use first potion
            viewModel.onIntent(GameIntent.ProcessCard(testPotion(5)))
            testDispatcher.scheduler.advanceUntilIdle()

            // Try to use second potion in same turn
            viewModel.onIntent(GameIntent.ProcessCard(testPotion(5)))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(15, state.health) // 10 + 5 (first potion only), not 10 + 5 + 5
            }
        }
}
