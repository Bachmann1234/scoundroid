package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.Deck
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.Rank
import dev.mattbachmann.scoundroid.data.model.Suit
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertTrue

class HeuristicPlayerTest {

    private val player = HeuristicPlayer()
    private val simulator = HeuristicSimulator()

    private fun gameWithDeck(cards: List<Card>, health: Int = 20): GameState {
        return GameState(
            deck = Deck(cards),
            health = health,
            currentRoom = null,
            weaponState = null,
            defeatedMonsters = emptyList(),
            discardPile = emptyList(),
            lastRoomAvoided = false,
            usedPotionThisTurn = false,
            lastCardProcessed = null,
        )
    }

    @Test
    fun `wins easy scenario with weak monsters`() {
        val game = gameWithDeck(
            listOf(
                Card(Suit.CLUBS, Rank.TWO),
                Card(Suit.SPADES, Rank.TWO),
                Card(Suit.CLUBS, Rank.THREE),
                Card(Suit.SPADES, Rank.THREE),
            ),
            health = 20
        )

        val result = player.playGame(game)

        println("Easy scenario: health=${result.health}, deck=${result.deck.size}, room=${result.currentRoom?.size}")
        assertTrue(result.health > 0)
        assertTrue(result.deck.isEmpty)
        println("Easy scenario: final health = ${result.health}")
    }

    @Test
    fun `manual trace of 8 card game`() {
        // 8 cards = 2 rooms
        val game = gameWithDeck(
            listOf(
                // Room 1
                Card(Suit.DIAMONDS, Rank.TEN), // Best weapon
                Card(Suit.CLUBS, Rank.TEN),    // Monster 10
                Card(Suit.HEARTS, Rank.TEN),   // Potion
                Card(Suit.CLUBS, Rank.FIVE),   // Monster 5
                // Room 2 (3 cards + 1 leftover)
                Card(Suit.SPADES, Rank.THREE), // Monster 3
                Card(Suit.HEARTS, Rank.FIVE),  // Potion
                Card(Suit.CLUBS, Rank.TWO),    // Monster 2
            ),
            health = 20
        )

        println("\n=== 8-Card Game Step-By-Step ===")

        // Step through manually
        var state = game.drawRoom()
        println("Drew room: ${state.currentRoom?.map { "${it.rank.displayName}${it.suit.symbol} (${it.type})" }}")

        // Simulate what HeuristicPlayer would do
        val room = state.currentRoom!!

        // What would be left?
        println("\nChoosing card to leave...")
        val monsters = room.filter { it.type == CardType.MONSTER }
        val weapons = room.filter { it.type == CardType.WEAPON }
        val potions = room.filter { it.type == CardType.POTION }
        println("  Monsters: ${monsters.map { it.value }}")
        println("  Weapons: ${weapons.map { it.value }}")
        println("  Potions: ${potions.map { it.value }}")

        // Now use the actual player
        val result = player.playGame(game)

        println("\nActual result:")
        println("  Health: ${result.health}")
        println("  Deck empty: ${result.deck.isEmpty}")
        println("  Game over: ${result.isGameOver}")

        // Expected: with 10-weapon, all monsters take 0 damage except none
        // 10-10=0, 5-10=0, 3-10=0, 2-10=0 = 0 total damage
        println("Expected 0 damage, actual = ${20 - result.health}")
        println("================================\n")

        assertTrue(result.health >= 15, "Should take minimal damage with weapon")
    }

    @Test
    fun `uses weapon effectively`() {
        val game = gameWithDeck(
            listOf(
                Card(Suit.DIAMONDS, Rank.TEN), // Weapon
                Card(Suit.CLUBS, Rank.KING),   // 13 damage, 3 with weapon
                Card(Suit.HEARTS, Rank.FIVE),
                Card(Suit.HEARTS, Rank.SIX),
            ),
            health = 10
        )

        val result = player.playGame(game)

        assertTrue(result.health > 0, "Should survive with weapon")
        println("Weapon scenario: final health = ${result.health}")
    }

    @Test
    fun `simulate 100 seeds with heuristic player`() {
        val results = simulator.simulateSeeds(1L..100L)

        val wins = results.values.count { it.wins > 0 }
        val avgWinScore = results.values
            .filter { it.wins > 0 }
            .mapNotNull { it.averageWinScore }
            .average()
            .takeIf { !it.isNaN() }

        println("\n=== Heuristic Player: 100 Seeds ===")
        println("Wins: $wins / 100 (${wins}%)")
        if (avgWinScore != null) {
            println("Average winning score: ${String.format("%.1f", avgWinScore)}")
        }

        // Show some individual results
        println("\nSample results:")
        results.entries.take(20).forEach { (seed, result) ->
            val status = if (result.wins > 0) "WIN" else "LOSS"
            println("  Seed $seed: $status (score: ${result.maxScore})")
        }
        println("===================================\n")
    }

    @Test
    fun `simulate 1000 seeds for accurate win rate`() {
        val numSeeds = 100_000L
        val startTime = System.currentTimeMillis()
        val results = simulator.simulateSeeds(1L..numSeeds)
        val elapsed = System.currentTimeMillis() - startTime

        val wins = results.values.count { it.wins > 0 }
        val winRate = wins.toDouble() / numSeeds * 100

        val winningScores = results.values
            .filter { it.wins > 0 }
            .map { it.maxScore }

        val avgWinScore = winningScores.average().takeIf { !it.isNaN() }
        val maxWinScore = winningScores.maxOrNull()
        val minWinScore = winningScores.minOrNull()

        val losingScores = results.values
            .filter { it.wins == 0 }
            .map { it.maxScore }
        val avgLossScore = losingScores.average().takeIf { !it.isNaN() }

        println("\n╔══════════════════════════════════════════════╗")
        println("║   HEURISTIC PLAYER: $numSeeds SEEDS")
        println("╠══════════════════════════════════════════════╣")
        println("║  Time: ${elapsed}ms (${String.format("%.2f", elapsed/1000.0)}s)")
        println("║  Speed: ${String.format("%.0f", numSeeds * 1000.0 / elapsed)} games/sec")
        println("║  Win Rate: ${String.format("%.3f", winRate)}%")
        println("║  Wins: $wins / $numSeeds")
        println("╠══════════════════════════════════════════════╣")
        if (avgWinScore != null) {
            println("║  Avg Win Score:  ${String.format("%.1f", avgWinScore)}")
            println("║  Best Win Score: $maxWinScore")
            println("║  Worst Win Score: $minWinScore")
        }
        if (avgLossScore != null) {
            println("║  Avg Loss Score: ${String.format("%.1f", avgLossScore)}")
        }
        println("╚══════════════════════════════════════════════╝\n")

        // Don't assert wins - game might be genuinely hard
        assertTrue(results.size == numSeeds.toInt(), "Should simulate all games")
    }

    @Test
    fun `find and trace winning seeds`() {
        println("\n=== Finding Winning Seeds ===")
        val results = simulator.simulateSeeds(1L..100_000L)
        val winningSeedsWithScores = results.entries
            .filter { it.value.wins > 0 }
            .map { it.key to it.value.maxScore }
            .sortedByDescending { it.second }

        println("Found ${winningSeedsWithScores.size} winning seeds:")
        winningSeedsWithScores.forEach { (seed, score) ->
            println("  Seed $seed: score $score")
        }

        // Trace the best winning seed
        if (winningSeedsWithScores.isNotEmpty()) {
            val bestSeed = winningSeedsWithScores.first().first
            println("\n=== Tracing Best Winning Seed: $bestSeed ===")
            traceGame(bestSeed)
        }
        println("==============================\n")
    }

    private fun traceGame(seed: Long) {
        val game = GameState.newGame(Random(seed))
        var state = game
        var roomNum = 0

        println("Starting health: ${state.health}, deck: ${state.deck.size}")

        while (!state.isGameOver) {
            // Check for win
            if (state.deck.isEmpty && (state.currentRoom == null || state.currentRoom.isEmpty()) && state.health > 0) {
                println("\n*** WON with ${state.health} health! ***")
                break
            }

            roomNum++
            if (roomNum > 25) {
                println("Safety break")
                break
            }

            // Draw room if needed
            while (state.currentRoom == null || state.currentRoom.isEmpty() ||
                (state.currentRoom.size < 4 && !state.deck.isEmpty)) {
                state = state.drawRoom()
            }

            val room = state.currentRoom ?: break
            val weapon = state.weaponState

            println("\n--- Room $roomNum ---")
            println("Health: ${state.health}, Deck: ${state.deck.size}")
            println("Weapon: ${weapon?.weapon?.value ?: "none"} (can hit: ${weapon?.maxMonsterValue ?: "any"})")
            println("Cards: ${room.map { "${it.rank.displayName}${it.suit.symbol}" }}")

            // Use the actual heuristic player for one step
            val oldHealth = state.health
            state = player.playGame(state.copy()) // Play to end of this room

            // For detailed trace, we'd need to expose internals, but this shows results
            println("After room: health ${oldHealth} -> ${state.health}")
        }

        println("\nFinal: health=${state.health}, deck=${state.deck.size}, room=${state.currentRoom?.size ?: 0}")
    }

    @Test
    fun `compare heuristic vs random play`() {
        val randomSimulator = MonteCarloSimulator()

        println("\n=== Heuristic vs Random: Seeds 1-50 ===")

        var heuristicWins = 0
        var randomWins = 0

        for (seed in 1L..50L) {
            val game = GameState.newGame(Random(seed))

            // Heuristic play (deterministic)
            val heuristicResult = player.playGame(game)
            val heuristicWon = heuristicResult.health > 0 && heuristicResult.deck.isEmpty

            // Random play (sample many times)
            val randomResult = randomSimulator.simulate(game, samples = 100, random = Random(seed * 1000))

            if (heuristicWon) heuristicWins++
            if (randomResult.wins > 0) randomWins++

            val hStatus = if (heuristicWon) "WIN" else "LOSS"
            val rStatus = "${randomResult.wins}%"

            if (seed <= 10) {
                println("Seed $seed: Heuristic=$hStatus, Random=$rStatus")
            }
        }

        println("...")
        println("Heuristic wins: $heuristicWins / 50")
        println("Random wins (any): $randomWins / 50")
        println("==========================================\n")
    }

    @Test
    fun `trace seed 1 step by step`() {
        println("\n=== SEED 1 MANUAL TRACE ===")
        val game = GameState.newGame(Random(1))

        var state = game
        var roomNum = 0

        while (!state.isGameOver && state.deck.size > 0 || (state.currentRoom?.isNotEmpty() == true)) {
            roomNum++
            if (roomNum > 20) break // Safety

            // Draw room
            if (state.currentRoom == null || state.currentRoom.isEmpty()) {
                state = state.drawRoom()
            }
            while ((state.currentRoom?.size ?: 0) < 4 && !state.deck.isEmpty) {
                state = state.drawRoom()
            }

            val room = state.currentRoom ?: break
            val weapon = state.weaponState

            println("\n--- Room $roomNum ---")
            println("Health: ${state.health}, Deck: ${state.deck.size}")
            println("Weapon: ${weapon?.weapon?.value ?: "none"} (can hit: ${weapon?.maxMonsterValue ?: "any"})")
            println("Cards: ${room.map { "${it.rank.displayName}${it.suit.symbol}" }}")

            if (room.size < 4 && state.deck.isEmpty) {
                // End game - process all
                println("END GAME - processing ${room.size} remaining cards")
                for (card in room) {
                    val oldHealth = state.health
                    state = when (card.type) {
                        CardType.MONSTER -> {
                            val canUse = state.weaponState?.canDefeat(card) == true
                            if (canUse) state.fightMonsterWithWeapon(card)
                            else state.fightMonsterBarehanded(card)
                        }
                        CardType.WEAPON -> state.equipWeapon(card)
                        CardType.POTION -> state.usePotion(card)
                    }
                    println("  ${card.rank.displayName}${card.suit.symbol}: ${oldHealth} -> ${state.health}")
                    if (state.isGameOver) break
                }
                state = state.copy(currentRoom = null)
                break
            }

            if (room.size == 4) {
                // Choose card to leave (smallest monster)
                val monsters = room.filter { it.type == CardType.MONSTER }
                val leave = if (monsters.isNotEmpty()) monsters.minBy { it.value } else room.first()
                val process = room.filter { it != leave }

                println("Leave: ${leave.rank.displayName}${leave.suit.symbol}")
                println("Process: ${process.map { "${it.rank.displayName}${it.suit.symbol}" }}")

                // Sort: weapons, big monsters, potions
                val ordered = process.sortedWith(compareBy(
                    { if (it.type == CardType.WEAPON) 0 else if (it.type == CardType.MONSTER) 1 else 2 },
                    { -it.value }
                ))

                state = state.copy(currentRoom = listOf(leave), usedPotionThisTurn = false)

                for (card in ordered) {
                    val oldHealth = state.health
                    val oldWeapon = state.weaponState?.weapon?.value
                    state = when (card.type) {
                        CardType.MONSTER -> {
                            val canUse = state.weaponState?.canDefeat(card) == true
                            if (canUse) state.fightMonsterWithWeapon(card)
                            else state.fightMonsterBarehanded(card)
                        }
                        CardType.WEAPON -> {
                            if (card.value > (state.weaponState?.weapon?.value ?: 0))
                                state.equipWeapon(card)
                            else state
                        }
                        CardType.POTION -> state.usePotion(card)
                    }
                    val action = when (card.type) {
                        CardType.MONSTER -> if (state.weaponState?.canDefeat(card) == true) "weapon" else "bare"
                        CardType.WEAPON -> if (card.value > (oldWeapon ?: 0)) "equip" else "skip"
                        CardType.POTION -> "heal"
                    }
                    println("  ${card.rank.displayName}${card.suit.symbol} ($action): ${oldHealth} -> ${state.health}")
                    if (state.isGameOver) {
                        println("  DIED!")
                        break
                    }
                }
            }

            if (state.isGameOver) break
        }

        println("\n=== FINAL ===")
        println("Health: ${state.health}")
        println("Deck: ${state.deck.size}")
        println("Room: ${state.currentRoom?.size ?: 0}")
        println("Won: ${state.health > 0 && state.deck.isEmpty && (state.currentRoom?.isEmpty() != false)}")
        println("=============\n")
    }

    @Test
    fun `debug single game`() {
        println("\n=== Debug: Seed 1 Game Trace ===")

        val game = GameState.newGame(Random(1))
        var state = game
        var turn = 0

        println("Starting health: ${state.health}")
        println("Deck size: ${state.deck.size}")

        while (!state.isGameOver && turn < 50) {
            turn++

            // Draw room if needed
            if (state.currentRoom == null || state.currentRoom.isEmpty()) {
                state = state.drawRoom()
                println("\nTurn $turn - Drew room:")
                state.currentRoom?.forEach { card ->
                    println("  ${card.rank.displayName}${card.suit.symbol} (${card.type}, value=${card.value})")
                }
                println("Health: ${state.health}, Deck: ${state.deck.size}")
            }

            // Fill room if needed
            if ((state.currentRoom?.size ?: 0) < 4 && !state.deck.isEmpty) {
                state = state.drawRoom()
                println("  Filled room to ${state.currentRoom?.size} cards")
            }

            // Process room
            val room = state.currentRoom ?: continue
            if (room.size == 4) {
                // Simple processing: equip weapons, use potions, fight monsters
                val weapons = room.filter { it.type == CardType.WEAPON }.sortedByDescending { it.value }
                val potions = room.filter { it.type == CardType.POTION }.sortedByDescending { it.value }
                val monsters = room.filter { it.type == CardType.MONSTER }.sortedBy { it.value }

                // Leave smallest value card
                val cardToLeave = room.minBy {
                    when (it.type) {
                        CardType.MONSTER -> -it.value // Keep big monsters to fight later? No, leave them
                        CardType.WEAPON -> it.value
                        CardType.POTION -> it.value
                    }
                }

                val toProcess = room.filter { it != cardToLeave }
                println("  Processing: ${toProcess.map { "${it.rank.displayName}${it.suit.symbol}" }}")
                println("  Leaving: ${cardToLeave.rank.displayName}${cardToLeave.suit.symbol}")

                state = state.copy(currentRoom = listOf(cardToLeave), usedPotionThisTurn = false)

                // Process in order: weapons, potions, monsters
                val ordered = weapons.filter { it != cardToLeave } +
                    potions.filter { it != cardToLeave } +
                    monsters.filter { it != cardToLeave }

                for (card in ordered) {
                    when (card.type) {
                        CardType.WEAPON -> {
                            state = state.equipWeapon(card)
                            println("    Equipped ${card.rank.displayName}${card.suit.symbol}")
                        }
                        CardType.POTION -> {
                            val oldHealth = state.health
                            state = state.usePotion(card)
                            println("    Used potion ${card.rank.displayName}${card.suit.symbol}: $oldHealth -> ${state.health}")
                        }
                        CardType.MONSTER -> {
                            val canUseWeapon = state.weaponState?.canDefeat(card) == true
                            val oldHealth = state.health
                            state = if (canUseWeapon) {
                                state.fightMonsterWithWeapon(card)
                            } else {
                                state.fightMonsterBarehanded(card)
                            }
                            val method = if (canUseWeapon) "with weapon" else "barehanded"
                            println("    Fought ${card.rank.displayName}${card.suit.symbol} $method: $oldHealth -> ${state.health}")

                            if (state.isGameOver) {
                                println("  DIED!")
                                break
                            }
                        }
                    }
                }
            } else if (room.size < 4 && state.deck.isEmpty) {
                // End game - process remaining
                println("  End game: processing ${room.size} remaining cards")
                for (card in room.sortedBy { if (it.type == CardType.MONSTER) 1 else 0 }) {
                    state = state.copy(currentRoom = null)
                    when (card.type) {
                        CardType.WEAPON -> state = state.equipWeapon(card)
                        CardType.POTION -> state = state.usePotion(card)
                        CardType.MONSTER -> {
                            val canUseWeapon = state.weaponState?.canDefeat(card) == true
                            state = if (canUseWeapon) {
                                state.fightMonsterWithWeapon(card)
                            } else {
                                state.fightMonsterBarehanded(card)
                            }
                        }
                    }
                    if (state.isGameOver) break
                }
            }

            if (state.isGameOver) break
        }

        println("\n=== Final State ===")
        println("Health: ${state.health}")
        println("Deck empty: ${state.deck.isEmpty}")
        println("Room empty: ${state.currentRoom?.isEmpty() ?: true}")
        println("Game over: ${state.isGameOver}")
        println("Score: ${state.calculateScore()}")
        println("===================================\n")
    }
}
