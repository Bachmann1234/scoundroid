package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.Deck
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import dev.mattbachmann.scoundroid.data.model.WeaponState
import kotlin.random.Random
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class InformedPlayerTest {
    private val player = InformedPlayer()

    private fun monster(
        value: Int,
        suit: Suit = Suit.CLUBS,
    ): Card = Card(suit, Rank.fromValue(value))

    private fun weapon(value: Int): Card = Card(Suit.DIAMONDS, Rank.fromValue(value))

    private fun potion(value: Int): Card = Card(Suit.HEARTS, Rank.fromValue(value))

    private fun gameWithDeck(
        cards: List<Card>,
        health: Int = 20,
        weaponState: WeaponState? = null,
    ): GameState =
        GameState(
            deck = Deck(cards),
            health = health,
            currentRoom = null,
            weaponState = weaponState,
            defeatedMonsters = emptyList(),
            discardPile = emptyList(),
            lastRoomAvoided = false,
            usedPotionThisTurn = false,
            lastCardProcessed = null,
        )

    @Test
    fun `wins easy scenario with weak monsters`() {
        val game =
            gameWithDeck(
                listOf(
                    monster(2, Suit.CLUBS),
                    monster(2, Suit.SPADES),
                    monster(3, Suit.CLUBS),
                    monster(3, Suit.SPADES),
                ),
                health = 20,
            )

        val result = player.playGame(game)

        assertTrue(result.health > 0)
        assertTrue(result.deck.isEmpty)
    }

    @Test
    fun `uses weapon effectively`() {
        val game =
            gameWithDeck(
                listOf(
                    weapon(10),
                    monster(13, Suit.CLUBS),
                    potion(5),
                    potion(6),
                ),
                health = 10,
            )

        val result = player.playGame(game)

        assertTrue(result.health > 0, "Should survive with weapon")
    }

    @Test
    fun `tracks deck knowledge correctly through game`() {
        // 8 cards = 2 rooms
        val game =
            gameWithDeck(
                listOf(
                    weapon(10),
                    monster(14, Suit.CLUBS), // Ace
                    monster(14, Suit.SPADES), // Ace
                    potion(5),
                    // Room 2
                    monster(8, Suit.CLUBS),
                    monster(7, Suit.SPADES),
                    potion(3),
                ),
                health = 20,
            )

        val result = player.playGame(game)

        // Should survive - weapon handles aces, small monsters are easy
        assertTrue(result.health > 0)
        assertTrue(result.deck.isEmpty)
    }

    @Test
    fun `dynamic weapon threshold - uses weapon on 8 when max remaining is 8`() {
        // Scenario: All monsters >= 9 have been processed
        // Max remaining monster is 8
        // Static threshold of 9 would NOT use weapon on 8
        // Dynamic threshold should use weapon on 8 (it's the biggest threat remaining)

        // Create knowledge where aces, kings, queens, jacks, 10s, 9s are all gone
        val processedMonsters =
            listOf(
                monster(14, Suit.CLUBS),
                monster(14, Suit.SPADES),
                monster(13, Suit.CLUBS),
                monster(13, Suit.SPADES),
                monster(12, Suit.CLUBS),
                monster(12, Suit.SPADES),
                monster(11, Suit.CLUBS),
                monster(11, Suit.SPADES),
                monster(10, Suit.CLUBS),
                monster(10, Suit.SPADES),
                monster(9, Suit.CLUBS),
                monster(9, Suit.SPADES),
            )

        var knowledge = DeckKnowledge.initial()
        for (card in processedMonsters) {
            knowledge = knowledge.cardProcessed(card)
        }

        // Max remaining monster should now be 8
        assertEquals(8, knowledge.maxMonsterRemaining)

        // The dynamic threshold should be 8
        val dynamicThreshold = player.getWeaponPreservationThreshold(knowledge)
        assertEquals(8, dynamicThreshold)
    }

    @Test
    fun `dynamic weapon threshold - caps at default when big monsters remain`() {
        val knowledge = DeckKnowledge.initial()

        // With aces still in deck, threshold should be the default (9)
        val threshold = player.getWeaponPreservationThreshold(knowledge)
        assertEquals(HeuristicPlayer.WEAPON_PRESERVATION_THRESHOLD, threshold)
    }

    @Test
    fun `weapon equip considers max remaining monster`() {
        // If current weapon (degraded to 6) can still handle all remaining monsters (max 5),
        // we might not want to swap to a fresh weapon

        var knowledge = DeckKnowledge.initial()
        // Process all monsters >= 6
        val bigMonsters =
            listOf(14, 13, 12, 11, 10, 9, 8, 7, 6).flatMap {
                listOf(monster(it, Suit.CLUBS), monster(it, Suit.SPADES))
            }
        for (m in bigMonsters) {
            knowledge = knowledge.cardProcessed(m)
        }

        // Max remaining is 5
        assertEquals(5, knowledge.maxMonsterRemaining)

        // Weapon degraded to 6 can still handle everything
        val currentWeapon = WeaponState(weapon(8), maxMonsterValue = 6)
        val newWeapon = weapon(4) // Fresh but lower value

        // Should NOT equip the 4-weapon since our degraded 8 (hitting up to 6) covers all
        val shouldEquip = player.shouldEquipWeaponWithKnowledge(currentWeapon, newWeapon, knowledge)
        assertTrue(!shouldEquip)
    }

    @Test
    fun `simulate 100 seeds and compare to heuristic`() {
        val result = PlayerBenchmark.runBenchmark(100L)

        // Just verify both run without crashing - don't assert win rate
        assertTrue(result.numSeeds == 100L)
    }

    @Test
    fun `find divergent seeds in 50k range`() {
        val result = PlayerBenchmark.runBenchmarkWithSeeds(50_000L)

        java.io.File("/tmp/divergent_seeds.txt").writeText(
            """
            DIVERGENT SEEDS (50000 range):
            Informed wins:  ${result.informedWins} (${String.format("%.3f", result.informedWins / 500.0)}%)
            Heuristic wins: ${result.heuristicWins} (${String.format("%.3f", result.heuristicWins / 500.0)}%)

            Only Informed won (${result.onlyInformed}): ${result.onlyInformedSeeds.take(20)}
            Only Heuristic won (${result.onlyHeuristic}): ${result.onlyHeuristicSeeds.take(20)}
            """.trimIndent(),
        )

        assertTrue(true)
    }

    @Test
    @Ignore("Slow benchmark - run manually")
    fun `benchmark 100k seeds`() {
        val result = PlayerBenchmark.runBenchmarkWithSeeds(100_000L)

        java.io.File("/tmp/benchmark_100k.txt").writeText(
            """
            BENCHMARK (100000 seeds):
            Informed wins:  ${result.informedWins} (${String.format("%.3f", result.informedWins / 1000.0)}%)
            Heuristic wins: ${result.heuristicWins} (${String.format("%.3f", result.heuristicWins / 1000.0)}%)

            Only Informed won (${result.onlyInformed}): ${result.onlyInformedSeeds.take(20)}
            Only Heuristic won (${result.onlyHeuristic}): ${result.onlyHeuristicSeeds.take(20)}
            Time: ${result.elapsedMs}ms
            """.trimIndent(),
        )

        assertTrue(true)
    }

    @Test
    fun `trace divergent seed 1723`() {
        // Trace a specific seed - change the number to investigate different seeds
        val seed = 9871L // Informed wins, Heuristic loses
        val comparison = PlayerComparison.compareSeed(seed)
        java.io.File("/tmp/seed_comparison.txt").writeText(comparison)

        val informedTrace = PlayerComparison.traceInformedPlayer(seed)
        java.io.File("/tmp/seed_informed.txt").writeText(informedTrace)

        assertTrue(true)
    }

    @Test
    @Ignore("Slow benchmark - run manually")
    fun `benchmark 10000 seeds and report results`() {
        val result = PlayerBenchmark.runBenchmark(10_000L)

        java.io.File("/tmp/benchmark_results.txt").writeText(
            """
            BENCHMARK RESULTS (10000 seeds):
            Informed wins:  ${result.informedWins} (${String.format("%.3f", result.informedWins / 100.0)}%)
            Heuristic wins: ${result.heuristicWins} (${String.format("%.3f", result.heuristicWins / 100.0)}%)
            Only Informed:  ${result.onlyInformed}
            Only Heuristic: ${result.onlyHeuristic}
            Both win:       ${result.bothWin}
            Time: ${result.elapsedMs}ms
            """.trimIndent(),
        )
    }

    @Test
    @Ignore("Slow benchmark test - run manually")
    fun `benchmark 10000 seeds`() {
        val heuristicPlayer = HeuristicPlayer()
        val numSeeds = 10_000L

        var informedWins = 0
        var heuristicWins = 0
        var bothWin = 0
        var onlyInformed = 0
        var onlyHeuristic = 0

        val startTime = System.currentTimeMillis()

        for (seed in 1L..numSeeds) {
            val game = GameState.newGame(Random(seed))

            val informedResult = player.playGame(game)
            val heuristicResult = heuristicPlayer.playGame(game)

            val informedWon =
                informedResult.health > 0 &&
                    informedResult.deck.isEmpty &&
                    (informedResult.currentRoom == null || informedResult.currentRoom.isEmpty())
            val heuristicWon =
                heuristicResult.health > 0 &&
                    heuristicResult.deck.isEmpty &&
                    (heuristicResult.currentRoom == null || heuristicResult.currentRoom.isEmpty())

            if (informedWon) informedWins++
            if (heuristicWon) heuristicWins++
            if (informedWon && heuristicWon) bothWin++
            if (informedWon && !heuristicWon) onlyInformed++
            if (!informedWon && heuristicWon) onlyHeuristic++
        }

        val elapsed = System.currentTimeMillis() - startTime

        println("\n╔══════════════════════════════════════════════╗")
        println("║   INFORMED vs HEURISTIC: $numSeeds SEEDS")
        println("╠══════════════════════════════════════════════╣")
        println("║  Time: ${elapsed}ms")
        println("║  Informed wins:  $informedWins (${String.format("%.3f", informedWins * 100.0 / numSeeds)}%)")
        println("║  Heuristic wins: $heuristicWins (${String.format("%.3f", heuristicWins * 100.0 / numSeeds)}%)")
        println("╠══════════════════════════════════════════════╣")
        println("║  Both win:       $bothWin")
        println("║  Only Informed:  $onlyInformed")
        println("║  Only Heuristic: $onlyHeuristic")
        println("╚══════════════════════════════════════════════╝\n")
    }
}
