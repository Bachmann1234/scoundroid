package dev.mattbachmann.scoundroid.ui.screen.game

import app.cash.turbine.test
import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.LogEntry
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.data.repository.HighScoreRepository
import io.mockk.coEvery
import io.mockk.coVerify
import io.mockk.mockk
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.test.StandardTestDispatcher
import kotlinx.coroutines.test.resetMain
import kotlinx.coroutines.test.runTest
import kotlinx.coroutines.test.setMain
import org.junit.After
import org.junit.Assume.assumeTrue
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
            // Seed 33 produces first room [8♦, 6♦, 4♦, 2♥] - all weapons/potions, no monsters
            val viewModel = GameViewModel(randomSeed = 33L)

            viewModel.uiState.test {
                awaitItem() // Initial state

                // First room
                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                val state = awaitItem()
                assertEquals(4, state.currentRoom!!.size)
                val cardsToSelect = state.currentRoom!!.take(3)

                // Select and process 3 cards (all weapons/potions, no combat choices)
                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()

                val afterProcessing = awaitItem()
                assertEquals(1, afterProcessing.currentRoom!!.size)

                // Draw next room (should draw 3 more + 1 remaining = 4 total)
                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()

                val newState = awaitItem()
                assertEquals(4, newState.currentRoom!!.size)
            }
        }

    // ========== Room Avoidance Tests ==========

    @Test
    fun `avoidRoom intent moves all 4 cards to bottom of deck and auto-draws new room`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.onIntent(GameIntent.AvoidRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                // After avoid, a new room is auto-drawn
                assertNotNull(state.currentRoom)
                assertEquals(4, state.currentRoom!!.size)
                // Avoided 4 cards go to bottom, new 4 drawn from top
                assertEquals(40, state.deckSize)
                assertTrue(state.lastRoomAvoided)
            }
        }

    @Test
    fun `cannot avoid room twice in a row`() =
        runTest {
            val viewModel = GameViewModel()

            // Draw and avoid first room (auto-draws second room)
            viewModel.onIntent(GameIntent.DrawRoom)
            testDispatcher.scheduler.advanceUntilIdle()
            viewModel.onIntent(GameIntent.AvoidRoom)
            testDispatcher.scheduler.advanceUntilIdle()

            // Second room is already auto-drawn after avoid
            viewModel.uiState.test {
                val state = awaitItem()
                // Room exists (auto-drawn) but cannot avoid because last room was avoided
                assertNotNull(state.currentRoom)
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

            viewModel.uiState.test {
                awaitItem() // Skip initial state

                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                val roomState = awaitItem()
                val cardsToSelect = roomState.currentRoom!!.take(3)

                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()

                // Process may pause for combat choices. Loop until room has 1 card (processing complete)
                var currentState = awaitItem()
                var safetyCounter = 0
                val maxIterations = 100
                while (currentState.currentRoom?.size != 1 && safetyCounter < maxIterations) {
                    if (currentState.pendingCombatChoice != null) {
                        viewModel.onIntent(GameIntent.ResolveCombatChoice(useWeapon = true))
                        testDispatcher.scheduler.advanceUntilIdle()
                        safetyCounter++
                    }
                    currentState = awaitItem()
                }
                assertTrue(safetyCounter < maxIterations, "Exceeded max iterations; possible infinite loop")

                assertEquals(1, currentState.currentRoom!!.size) // 1 card left for next room
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

    // ========== High Score Tests ==========

    @Test
    fun `initial state loads highest score from repository`() =
        runTest {
            val mockRepo = mockk<HighScoreRepository>(relaxed = true)
            coEvery { mockRepo.getHighestScore() } returns 15

            val viewModel = GameViewModel(mockRepo)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(15, state.highestScore)
            }
        }

    @Test
    fun `initial state shows null highest score when no scores exist`() =
        runTest {
            val mockRepo = mockk<HighScoreRepository>(relaxed = true)
            coEvery { mockRepo.getHighestScore() } returns null

            val viewModel = GameViewModel(mockRepo)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertNull(state.highestScore)
            }
        }

    @Test
    fun `isNewHighScore is true when current score beats existing high score`() =
        runTest {
            val mockRepo = mockk<HighScoreRepository>(relaxed = true)
            coEvery { mockRepo.getHighestScore() } returns 10
            coEvery { mockRepo.isNewHighScore(any()) } returns true

            val viewModel = GameViewModel(mockRepo)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                // Initial health of 20 should beat high score of 10
                assertTrue(state.isNewHighScore)
            }
        }

    @Test
    fun `isNewHighScore is false when current score is below existing high score`() =
        runTest {
            val mockRepo = mockk<HighScoreRepository>(relaxed = true)
            coEvery { mockRepo.getHighestScore() } returns 20
            coEvery { mockRepo.isNewHighScore(any()) } returns false

            val viewModel = GameViewModel(mockRepo)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertFalse(state.isNewHighScore)
            }
        }

    @Test
    fun `score is saved when game is won`() =
        runTest {
            val mockRepo = mockk<HighScoreRepository>(relaxed = true)
            coEvery { mockRepo.getHighestScore() } returns null
            coEvery { mockRepo.isNewHighScore(any()) } returns true

            val viewModel = GameViewModel(mockRepo)
            testDispatcher.scheduler.advanceUntilIdle()

            // Simulate winning by ending game (empty deck with health > 0)
            viewModel.onIntent(GameIntent.GameEnded(score = 15, won = true))
            testDispatcher.scheduler.advanceUntilIdle()

            coVerify { mockRepo.saveScore(score = 15, won = true) }
        }

    @Test
    fun `score is saved when game is lost`() =
        runTest {
            val mockRepo = mockk<HighScoreRepository>(relaxed = true)
            coEvery { mockRepo.getHighestScore() } returns null

            val viewModel = GameViewModel(mockRepo)
            testDispatcher.scheduler.advanceUntilIdle()

            // Simulate losing
            viewModel.onIntent(GameIntent.GameEnded(score = -30, won = false))
            testDispatcher.scheduler.advanceUntilIdle()

            coVerify { mockRepo.saveScore(score = -30, won = false) }
        }

    @Test
    fun `highest score is updated after saving new high score`() =
        runTest {
            val mockRepo = mockk<HighScoreRepository>(relaxed = true)
            coEvery { mockRepo.getHighestScore() } returns 10 andThen 18
            coEvery { mockRepo.isNewHighScore(any()) } returns true

            val viewModel = GameViewModel(mockRepo)
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.onIntent(GameIntent.GameEnded(score = 18, won = true))
            testDispatcher.scheduler.advanceUntilIdle()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(18, state.highestScore)
            }
        }

    // ========== Help Screen Tests ==========

    @Test
    fun `ShowHelp intent sets showHelp to true`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                val initialState = awaitItem()
                assertFalse(initialState.showHelp)

                viewModel.onIntent(GameIntent.ShowHelp)
                testDispatcher.scheduler.advanceUntilIdle()

                val newState = awaitItem()
                assertTrue(newState.showHelp)
            }
        }

    @Test
    fun `HideHelp intent sets showHelp to false`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                // Show help first
                viewModel.onIntent(GameIntent.ShowHelp)
                testDispatcher.scheduler.advanceUntilIdle()
                val shownState = awaitItem()
                assertTrue(shownState.showHelp)

                // Now hide it
                viewModel.onIntent(GameIntent.HideHelp)
                testDispatcher.scheduler.advanceUntilIdle()

                val hiddenState = awaitItem()
                assertFalse(hiddenState.showHelp)
            }
        }

    // ========== Action Log Tests ==========

    @Test
    fun `ShowActionLog intent sets showActionLog to true`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                val initialState = awaitItem()
                assertFalse(initialState.showActionLog)

                viewModel.onIntent(GameIntent.ShowActionLog)
                testDispatcher.scheduler.advanceUntilIdle()

                val newState = awaitItem()
                assertTrue(newState.showActionLog)
            }
        }

    @Test
    fun `HideActionLog intent sets showActionLog to false`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                viewModel.onIntent(GameIntent.ShowActionLog)
                testDispatcher.scheduler.advanceUntilIdle()
                val shownState = awaitItem()
                assertTrue(shownState.showActionLog)

                viewModel.onIntent(GameIntent.HideActionLog)
                testDispatcher.scheduler.advanceUntilIdle()

                val hiddenState = awaitItem()
                assertFalse(hiddenState.showActionLog)
            }
        }

    @Test
    fun `new game adds GameStarted entry to log`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                val state = awaitItem()
                assertEquals(1, state.actionLog.size)
                assertTrue(state.actionLog[0] is LogEntry.GameStarted)
            }
        }

    @Test
    fun `drawRoom adds RoomDrawn entry to log`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state (GameStarted)

                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()

                val state = awaitItem()
                // Should have GameStarted + RoomDrawn
                assertEquals(2, state.actionLog.size)
                val lastEntry = state.actionLog.last()
                assertTrue(lastEntry is LogEntry.RoomDrawn)
                val roomDrawn = lastEntry as LogEntry.RoomDrawn
                assertEquals(4, roomDrawn.cardsDrawn)
                assertEquals(40, roomDrawn.deckSizeAfter)
            }
        }

    @Test
    fun `avoidRoom adds RoomAvoided entry to log`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                awaitItem()

                viewModel.onIntent(GameIntent.AvoidRoom)
                testDispatcher.scheduler.advanceUntilIdle()

                val state = awaitItem()
                // Log should have RoomAvoided followed by RoomDrawn (auto-draw)
                val roomAvoidedEntry = state.actionLog.filterIsInstance<LogEntry.RoomAvoided>().last()
                assertEquals(4, roomAvoidedEntry.cardsReturned)
                val roomDrawnEntry = state.actionLog.filterIsInstance<LogEntry.RoomDrawn>().last()
                assertTrue(roomDrawnEntry.cardsDrawn > 0)
            }
        }

    @Test
    fun `avoidRoom auto-draws next room`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                awaitItem()

                viewModel.onIntent(GameIntent.AvoidRoom)
                testDispatcher.scheduler.advanceUntilIdle()

                val state = awaitItem()
                // After avoiding, a new room should be automatically drawn
                assertNotNull(state.currentRoom)
                assertEquals(4, state.currentRoom!!.size)
                // Deck should still be 40 (avoided 4 goes to bottom, new 4 drawn)
                assertEquals(40, state.deckSize)
            }
        }

    @Test
    fun `fighting monster barehanded creates log entry with full damage`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                skipItems(1) // Skip initial state

                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                val roomState = awaitItem()

                // Ensure room is drawn
                assertNotNull(roomState.currentRoom)
                val room = roomState.currentRoom!!

                // Find a monster in the room - skip test if none found
                val monster = room.find { it.type == CardType.MONSTER }
                assumeTrue(monster != null)

                // Select 3 cards with monster FIRST to ensure barehanded fighting
                // (weapons process in order, so monster first means no weapon equipped yet)
                val otherCards = room.filter { it != monster }.take(2)
                val cardsToSelect = listOf(monster!!) + otherCards
                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()

                // Process may pause for combat choices (if later cards include weapon then monster)
                var state = awaitItem()
                var safetyCounter = 0
                val maxIterations = 100
                while (state.pendingCombatChoice != null && safetyCounter < maxIterations) {
                    viewModel.onIntent(GameIntent.ResolveCombatChoice(useWeapon = true))
                    testDispatcher.scheduler.advanceUntilIdle()
                    state = awaitItem()
                    safetyCounter++
                }
                assertTrue(safetyCounter < maxIterations, "Exceeded max iterations; possible infinite loop")

                // Find the MonsterFought entry for our specific monster
                val monsterEntry =
                    state.actionLog.filterIsInstance<LogEntry.MonsterFought>()
                        .find { it.monster == monster }
                assertNotNull(monsterEntry)
                assertNull(monsterEntry!!.weaponUsed)
                assertEquals(monster.value, monsterEntry.damageTaken)
                assertEquals(0, monsterEntry.damageBlocked)
                // Health after should be health before minus damage
                assertEquals(monsterEntry.healthBefore - monster.value, monsterEntry.healthAfter)
                cancelAndIgnoreRemainingEvents()
            }
        }

    @Test
    fun `equipping weapon creates log entry`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                skipItems(1) // Skip initial state

                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                val roomState = awaitItem()

                assertNotNull(roomState.currentRoom)
                val room = roomState.currentRoom!!

                // Find a weapon in the room - skip test if none found
                val weapon = room.find { it.type == CardType.WEAPON }
                assumeTrue(weapon != null)

                val cardsToSelect = room.take(3).toMutableList()
                if (weapon !in cardsToSelect) {
                    cardsToSelect[0] = weapon!!
                }
                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()

                // Process may pause for combat choices if weapon is equipped then monster is fought
                var state = awaitItem()
                var safetyCounter = 0
                val maxIterations = 100
                while (state.pendingCombatChoice != null && safetyCounter < maxIterations) {
                    viewModel.onIntent(GameIntent.ResolveCombatChoice(useWeapon = true))
                    testDispatcher.scheduler.advanceUntilIdle()
                    state = awaitItem()
                    safetyCounter++
                }
                assertTrue(safetyCounter < maxIterations, "Exceeded max iterations; possible infinite loop")

                val weaponEntry = state.actionLog.filterIsInstance<LogEntry.WeaponEquipped>().firstOrNull()
                assertNotNull(weaponEntry)
                assertEquals(weapon, weaponEntry!!.weapon)
                assertNull(weaponEntry.replacedWeapon)
                cancelAndIgnoreRemainingEvents()
            }
        }

    @Test
    fun `using potion creates log entry`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                skipItems(1) // Skip initial state

                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                val roomState = awaitItem()

                assertNotNull(roomState.currentRoom)
                val room = roomState.currentRoom!!

                // Find a potion in the room - skip test if none found
                val potion = room.find { it.type == CardType.POTION }
                assumeTrue(potion != null)

                val cardsToSelect = room.take(3).toMutableList()
                if (potion !in cardsToSelect) {
                    cardsToSelect[0] = potion!!
                }
                viewModel.onIntent(GameIntent.ProcessSelectedCards(cardsToSelect))
                testDispatcher.scheduler.advanceUntilIdle()

                // Process may pause for combat choices
                var state = awaitItem()
                var safetyCounter = 0
                val maxIterations = 100
                while (state.pendingCombatChoice != null && safetyCounter < maxIterations) {
                    viewModel.onIntent(GameIntent.ResolveCombatChoice(useWeapon = true))
                    testDispatcher.scheduler.advanceUntilIdle()
                    state = awaitItem()
                    safetyCounter++
                }
                assertTrue(safetyCounter < maxIterations, "Exceeded max iterations; possible infinite loop")

                val potionEntry = state.actionLog.filterIsInstance<LogEntry.PotionUsed>().firstOrNull()
                assertNotNull(potionEntry)
                assertEquals(potion, potionEntry!!.potion)
                assertFalse(potionEntry.wasDiscarded)
                // healthAfter should be >= healthBefore (healing or capped)
                assertTrue(potionEntry.healthAfter >= potionEntry.healthBefore)
                cancelAndIgnoreRemainingEvents()
            }
        }

    @Test
    fun `NewGame intent clears log and adds GameStarted`() =
        runTest {
            val viewModel = GameViewModel()

            viewModel.uiState.test {
                awaitItem() // Initial state

                // Play some of the game to build up log
                viewModel.onIntent(GameIntent.DrawRoom)
                testDispatcher.scheduler.advanceUntilIdle()
                awaitItem()

                // Start new game
                viewModel.onIntent(GameIntent.NewGame)
                testDispatcher.scheduler.advanceUntilIdle()

                val state = awaitItem()
                // Log should have only GameStarted
                assertEquals(1, state.actionLog.size)
                assertTrue(state.actionLog[0] is LogEntry.GameStarted)
            }
        }

    // ========== Simulate Processing (Preview) Tests ==========

    @Test
    fun `simulateProcessing returns empty list for empty selection`() =
        runTest {
            val viewModel = GameViewModel()

            val result = viewModel.simulateProcessing(emptyList())

            assertTrue(result.isEmpty())
        }

    @Test
    fun `simulateProcessing returns MonsterFought entry for monster card`() =
        runTest {
            val viewModel = GameViewModel()
            val monster = testMonster(8)

            val result = viewModel.simulateProcessing(listOf(monster))

            assertEquals(1, result.size)
            assertTrue(result[0] is LogEntry.MonsterFought)
            val entry = result[0] as LogEntry.MonsterFought
            assertEquals(monster, entry.monster)
            assertEquals(8, entry.damageTaken)
            assertNull(entry.weaponUsed)
            assertEquals(20, entry.healthBefore)
            assertEquals(12, entry.healthAfter)
        }

    @Test
    fun `simulateProcessing returns WeaponEquipped entry for weapon card`() =
        runTest {
            val viewModel = GameViewModel()
            val weapon = testWeapon(5)

            val result = viewModel.simulateProcessing(listOf(weapon))

            assertEquals(1, result.size)
            assertTrue(result[0] is LogEntry.WeaponEquipped)
            val entry = result[0] as LogEntry.WeaponEquipped
            assertEquals(weapon, entry.weapon)
            assertNull(entry.replacedWeapon)
        }

    @Test
    fun `simulateProcessing returns PotionUsed entry for potion card`() =
        runTest {
            val viewModel = GameViewModel()
            val potion = testPotion(7)

            val result = viewModel.simulateProcessing(listOf(potion))

            assertEquals(1, result.size)
            assertTrue(result[0] is LogEntry.PotionUsed)
            val entry = result[0] as LogEntry.PotionUsed
            assertEquals(potion, entry.potion)
            assertEquals(0, entry.healthRestored) // Already at max health
            assertFalse(entry.wasDiscarded)
        }

    @Test
    fun `simulateProcessing shows weapon reducing monster damage`() =
        runTest {
            val viewModel = GameViewModel()
            val weapon = testWeapon(5)
            val monster = testMonster(8)

            // Weapon first, then monster - weapon should reduce damage
            val result = viewModel.simulateProcessing(listOf(weapon, monster))

            assertEquals(2, result.size)
            assertTrue(result[0] is LogEntry.WeaponEquipped)
            assertTrue(result[1] is LogEntry.MonsterFought)
            val monsterEntry = result[1] as LogEntry.MonsterFought
            assertEquals(weapon, monsterEntry.weaponUsed)
            assertEquals(5, monsterEntry.damageBlocked)
            assertEquals(3, monsterEntry.damageTaken) // 8 - 5 = 3
            assertEquals(17, monsterEntry.healthAfter) // 20 - 3 = 17
        }

    @Test
    fun `simulateProcessing shows second potion as discarded`() =
        runTest {
            val viewModel = GameViewModel()
            val potion1 = testPotion(5)
            val potion2 = testPotion(7)

            val result = viewModel.simulateProcessing(listOf(potion1, potion2))

            assertEquals(2, result.size)
            val firstPotionEntry = result[0] as LogEntry.PotionUsed
            assertFalse(firstPotionEntry.wasDiscarded)

            val secondPotionEntry = result[1] as LogEntry.PotionUsed
            assertTrue(secondPotionEntry.wasDiscarded)
            assertEquals(0, secondPotionEntry.healthRestored)
        }

    @Test
    fun `simulateProcessing does not mutate actual game state`() =
        runTest {
            val viewModel = GameViewModel()
            val monster = testMonster(10)

            viewModel.uiState.test {
                val initialState = awaitItem()
                val initialHealth = initialState.health

                // Simulate processing monster
                viewModel.simulateProcessing(listOf(monster))

                // Health should not change
                val currentState = viewModel.uiState.value
                assertEquals(initialHealth, currentState.health)
            }
        }

    @Test
    fun `simulateProcessing reflects order-dependent weapon degradation`() =
        runTest {
            val viewModel = GameViewModel()
            val weapon = testWeapon(5)
            val smallMonster = testMonster(3)
            val bigMonster = testMonster(12)

            // Equip weapon, fight small monster (weapon degrades to max 3), then big monster
            val result = viewModel.simulateProcessing(listOf(weapon, smallMonster, bigMonster))

            assertEquals(3, result.size)

            // Small monster should use weapon
            val smallFight = result[1] as LogEntry.MonsterFought
            assertEquals(weapon, smallFight.weaponUsed)

            // Big monster should NOT use weapon (12 > 3 after degradation)
            val bigFight = result[2] as LogEntry.MonsterFought
            assertNull(bigFight.weaponUsed)
            assertEquals(12, bigFight.damageTaken) // Full damage
        }
}
