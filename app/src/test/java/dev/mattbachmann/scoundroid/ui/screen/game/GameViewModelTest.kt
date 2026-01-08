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

                // Select and process 3 cards, leaving 1
                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
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

                // Select and process cards
                val cardsToSelect = state.currentRoom!!.take(3)
                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
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

    // ========== Card Selection and Processing Tests ==========

    @Test
    fun `processSelectedCards intent selects and processes 3 of 4 cards`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                val cardsToSelect = state.currentRoom!!.take(3)

                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()

                val newState = awaitItem()
                assertEquals(1, newState.currentRoom!!.size) // 1 card left for next room
            }
        }

    // ========== Card Processing Logic Tests (via GameState) ==========

    @Test
    fun `fighting monster reduces health`() =
        runTest {
            val monster = testMonster(10)
            val initialState = GameState.newGame()

            val newState = initialState.fightMonster(monster)

            assertEquals(10, newState.health) // 20 - 10 = 10
        }

    @Test
    fun `equipping weapon updates weapon state`() =
        runTest {
            val weapon = testWeapon(5)
            val initialState = GameState.newGame()

            val newState = initialState.equipWeapon(weapon)

            assertNotNull(newState.weaponState)
            assertEquals(5, newState.weaponState!!.weapon.value)
        }

    @Test
    fun `using potion restores health`() =
        runTest {
            val potion = testPotion(7)
            val initialState = GameState.newGame().fightMonster(testMonster(10)) // Take damage first

            val newState = initialState.usePotion(potion)

            assertEquals(17, newState.health) // 10 + 7 = 17
        }

    @Test
    fun `weapon reduces monster damage`() =
        runTest {
            val initialState =
                GameState.newGame()
                    .equipWeapon(testWeapon(5))
                    .fightMonster(testMonster(8))

            assertEquals(17, initialState.health) // 20 - (8 - 5) = 17
        }

    // ========== Game Over Tests ==========

    @Test
    fun `game over when health reaches 0`() =
        runTest {
            val state =
                GameState.newGame()
                    .fightMonster(testMonster(14)) // 20 - 14 = 6
                    .fightMonster(testMonster(6)) // 6 - 6 = 0

            assertEquals(0, state.health)
            assertTrue(state.isGameOver)
            assertFalse(state.isGameWon)
        }

    @Test
    fun `game won when deck is empty with health greater than 0`() =
        runTest {
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
            val state = GameState.newGame().fightMonster(testMonster(7))

            assertEquals(13, state.health)
            assertEquals(13, state.calculateScore())
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
            val state = GameState.newGame().fightMonster(testMonster(5))

            assertEquals(1, state.defeatedMonsters.size)
        }

    // ========== New Game Tests ==========

    @Test
    fun `newGame intent resets game to initial state`() =
        runTest {
            val viewModel = GameViewModel()

            // Play some of the game
            viewModel.onIntent(GameIntent.DrawRoom)
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
            val state =
                GameState.newGame()
                    .equipWeapon(testWeapon(5))
                    .fightMonster(testMonster(12))

            assertNotNull(state.weaponState)
            assertEquals(12, state.weaponState!!.maxMonsterValue)
        }

    @Test
    fun `potion cannot exceed max health`() =
        runTest {
            val state = GameState.newGame().usePotion(testPotion(10))

            assertEquals(20, state.health) // Still 20, not 30
        }

    @Test
    fun `only first potion per room processing sequence is applied`() =
        runTest {
            // This tests that when processing multiple cards without drawing a new room,
            // only the first potion takes effect (usedPotionThisTurn flag)
            val state =
                GameState.newGame()
                    .fightMonster(testMonster(10)) // Health: 10
                    .usePotion(testPotion(5)) // Health: 15 (first potion applied)
                    .usePotion(testPotion(5)) // Health: 15 (second potion ignored)

            assertEquals(15, state.health) // 10 + 5 (first potion only), not 10 + 5 + 5
        }
}
