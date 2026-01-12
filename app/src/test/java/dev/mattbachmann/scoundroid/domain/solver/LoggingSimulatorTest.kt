package dev.mattbachmann.scoundroid.domain.solver

import dev.mattbachmann.scoundroid.data.model.Card
import dev.mattbachmann.scoundroid.data.model.CardType
import dev.mattbachmann.scoundroid.data.model.Deck
import dev.mattbachmann.scoundroid.data.model.GameState
import dev.mattbachmann.scoundroid.data.model.Rank
import org.junit.Ignore
import org.junit.Test
import java.io.File
import kotlin.random.Random

class LoggingSimulatorTest {
    @Test
    fun `analyze heuristic player performance`() {
        val simulator = LoggingSimulator()
        val outputFile = File("build/analysis-output.txt")

        val output = StringBuilder()

        fun log(s: String = "") {
            output.appendLine(s)
        }

        log("Running 10,000 games with logging...")
        val analysis = simulator.simulate(1L..10_000L, collectDetailedLogs = 50)

        log()
        log(analysis.stats.prettyPrint())

        // Analyze the detailed logs
        log()
        log("=== DETAILED LOSS ANALYSIS ===")

        analyzeSkipPatterns(analysis.sampleLogs, ::log)
        analyzeWeaponUsage(analysis.sampleLogs, ::log)
        analyzeCardLeftBehind(analysis.sampleLogs, ::log)
        analyzeSampleGames(analysis.sampleLogs.take(5), ::log)

        outputFile.writeText(output.toString())
        println("Analysis written to: ${outputFile.absolutePath}")
    }

    private fun analyzeSkipPatterns(
        logs: List<GameLog>,
        log: (String) -> Unit,
    ) {
        log("")
        log("--- Room Skip Patterns ---")

        val skippedRooms =
            logs.flatMap { gameLog ->
                gameLog.events.filterIsInstance<GameEvent.RoomSkipped>()
            }

        if (skippedRooms.isEmpty()) {
            log("No rooms were skipped in sample games")
            return
        }

        val skipReasons = skippedRooms.groupBy { it.reason.substringBefore("(") }
        log("Skip reasons:")
        skipReasons.forEach { (reason, events) ->
            log("  ${events.size}x: $reason")
        }

        val avgHealthWhenSkipping = skippedRooms.map { it.health }.average()
        log("Average health when skipping: ${"%.1f".format(avgHealthWhenSkipping)}")

        val avgDamageWhenSkipping = skippedRooms.map { it.estimatedDamage }.average()
        log("Average estimated damage when skipping: ${"%.1f".format(avgDamageWhenSkipping)}")
    }

    private fun analyzeWeaponUsage(
        logs: List<GameLog>,
        log: (String) -> Unit,
    ) {
        log("")
        log("--- Weapon Usage Patterns ---")

        val combatEvents =
            logs.flatMap { gameLog ->
                gameLog.events.filterIsInstance<GameEvent.CombatResolved>()
            }

        val withWeapon = combatEvents.count { it.usedWeapon }
        val barehanded = combatEvents.count { !it.usedWeapon }
        val barehandedWithWeaponAvailable =
            combatEvents.count {
                !it.usedWeapon && it.weaponValue != null && it.weaponValue > 0
            }

        log("Fights with weapon: $withWeapon")
        log("Fights barehanded: $barehanded")
        log("Fights barehanded despite having weapon: $barehandedWithWeaponAvailable")

        if (barehandedWithWeaponAvailable > 0) {
            val avgDamageWhenForcedBarehanded =
                combatEvents
                    .filter { !it.usedWeapon && it.weaponValue != null && it.weaponValue > 0 }
                    .map { it.damageTaken }
                    .average()
            log("  Average damage in forced barehanded fights: ${"%.1f".format(avgDamageWhenForcedBarehanded)}")
        }

        // Weapon degradation analysis
        val weaponSkips =
            logs.flatMap { gameLog ->
                gameLog.events.filterIsInstance<GameEvent.WeaponSkipped>()
            }
        log("Weapons skipped (kept current): ${weaponSkips.size}")

        val weaponEquips =
            logs.flatMap { gameLog ->
                gameLog.events.filterIsInstance<GameEvent.WeaponEquipped>()
            }
        val upgradesFromDegraded = weaponEquips.count { it.previousWeaponDegraded }
        log("Weapon upgrades from degraded weapon: $upgradesFromDegraded")
    }

    private fun analyzeCardLeftBehind(
        logs: List<GameLog>,
        log: (String) -> Unit,
    ) {
        log("")
        log("--- Cards Left Behind Patterns ---")

        val leftBehindEvents =
            logs.flatMap { gameLog ->
                gameLog.events.filterIsInstance<GameEvent.CardLeftBehind>()
            }

        val byType = leftBehindEvents.groupBy { it.cardLeft.type }
        log("Cards left behind by type:")
        byType.forEach { (type, events) ->
            val avgValue = events.map { it.cardLeft.value }.average()
            log("  $type: ${events.size} times, avg value: ${"%.1f".format(avgValue)}")
        }

        // Look at monsters left behind
        val monstersLeft = leftBehindEvents.filter { it.cardLeft.type == CardType.MONSTER }
        if (monstersLeft.isNotEmpty()) {
            val monstersByRank =
                monstersLeft
                    .groupBy { it.cardLeft.rank.value }
                    .mapValues { it.value.size }
                    .toList()
                    .sortedByDescending { it.second }

            log("Monster ranks left behind (most common first):")
            monstersByRank.take(5).forEach { (rank, count) ->
                log("  Rank $rank: $count times")
            }
        }
    }

    private fun analyzeSampleGames(
        logs: List<GameLog>,
        log: (String) -> Unit,
    ) {
        log("")
        log("=== SAMPLE GAME TRACES ===")

        logs.forEachIndexed { index, gameLog ->
            log("")
            log("--- Game ${index + 1} (Seed: ${gameLog.seed}) ---")
            log(
                "Result: ${if (gameLog.won) "WON" else "LOST"} | Final Health: ${gameLog.finalHealth} | Score: ${gameLog.finalScore}",
            )
            log("Cards remaining in deck: ${gameLog.cardsRemainingInDeck}")
            log("")

            gameLog.events.forEach { event ->
                when (event) {
                    is GameEvent.RoomDrawn -> {
                        val cardsSummary =
                            event.cards.joinToString(", ") {
                                "${it.rank.displayName}${it.suit.symbol}"
                            }
                        log(
                            "Room ${event.roomNumber}: [$cardsSummary] (HP: ${event.health}, Deck: ${event.deckRemaining})",
                        )
                        event.weaponState?.let { ws ->
                            val degraded = ws.maxMonsterValue?.let { " (max: $it)" } ?: ""
                            log("  Weapon: ${ws.weapon.rank.displayName}${ws.weapon.suit.symbol}$degraded")
                        }
                    }
                    is GameEvent.RoomSkipped -> {
                        log("  -> SKIPPED: ${event.reason}")
                    }
                    is GameEvent.CardLeftBehind -> {
                        log("  -> Left: ${event.cardLeft.rank.displayName}${event.cardLeft.suit.symbol}")
                    }
                    is GameEvent.CombatResolved -> {
                        val weaponInfo = if (event.usedWeapon) "with weapon" else "barehanded"
                        log(
                            "  Fight ${event.monster.rank.displayName}${event.monster.suit.symbol} $weaponInfo: ${event.healthBefore} -> ${event.healthAfter} (-${event.damageTaken})",
                        )
                    }
                    is GameEvent.WeaponEquipped -> {
                        val prev = event.previousWeapon?.let { "${it.rank.displayName}${it.suit.symbol}" } ?: "none"
                        log("  Equip ${event.weapon.rank.displayName}${event.weapon.suit.symbol} (was: $prev)")
                    }
                    is GameEvent.WeaponSkipped -> {
                        log(
                            "  Skip weapon ${event.weapon.rank.displayName}${event.weapon.suit.symbol}: ${event.reason}",
                        )
                    }
                    is GameEvent.PotionUsed -> {
                        val wasted = if (event.healingWasted > 0) " (${event.healingWasted} wasted)" else ""
                        log("  Potion ${event.healthBefore} -> ${event.healthAfter}$wasted")
                    }
                    is GameEvent.PotionWasted -> {
                        log("  Potion WASTED: ${event.reason}")
                    }
                    is GameEvent.GameEnded -> {
                        // Already printed at top
                    }
                }
            }

            gameLog.deathInfo?.let { death ->
                log("")
                log("DEATH: Killed by ${death.killerMonster.rank.displayName}${death.killerMonster.suit.symbol}")
                log("  HP before hit: ${death.healthBeforeHit}, Damage: ${death.damageTaken}")
                if (death.hadWeapon) {
                    log(
                        "  Had weapon: ${death.weaponValue}${death.weaponMaxMonster?.let {
                            " (degraded to max $it)"
                        } ?: ""}",
                    )
                    if (death.couldWeaponHaveHelped) {
                        log("  WEAPON COULD HAVE HELPED if not degraded!")
                    }
                }
                log("  Rooms skipped: ${death.roomsSkipped}, Potions used: ${death.potionsUsedThisGame}")
            }
        }
    }

    @Test
    @Ignore("Long running - run manually to collect winnable seeds")
    fun `collect winnable seeds`() {
        val player = HeuristicPlayer()
        val winnableSeeds = mutableListOf<Long>()
        val totalGames = 500_000L
        val outputFile = File("build/winnable-seeds.txt")

        println("Collecting winnable seeds from $totalGames games...")
        val startTime = System.currentTimeMillis()

        for (seed in 1L..totalGames) {
            val game = GameState.newGame(Random(seed))
            val finalState = player.playGame(game)

            val won =
                finalState.health > 0 &&
                    finalState.deck.isEmpty &&
                    (finalState.currentRoom == null || finalState.currentRoom.isEmpty())

            if (won) {
                winnableSeeds.add(seed)
            }

            if (seed % 50_000 == 0L) {
                val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
                val rate = seed / elapsed
                println(
                    "Progress: $seed/$totalGames (${winnableSeeds.size} wins, " +
                        "${"%.2f".format(winnableSeeds.size * 100.0 / seed)}% win rate, " +
                        "${"%.0f".format(rate)} games/sec)",
                )
            }
        }

        val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
        val winRate = winnableSeeds.size * 100.0 / totalGames

        val output =
            buildString {
                appendLine("# Winnable Seeds for Scoundrel Heuristic Player")
                appendLine("# Generated: ${java.time.LocalDateTime.now()}")
                appendLine("# Total games: $totalGames")
                appendLine("# Winnable seeds found: ${winnableSeeds.size}")
                appendLine("# Win rate: ${"%.4f".format(winRate)}%")
                appendLine("# Time: ${"%.1f".format(elapsed)} seconds")
                appendLine("#")
                appendLine("# These seeds are confirmed winnable with the current heuristic strategy.")
                appendLine("# Use these for focused testing and optimization.")
                appendLine()
                winnableSeeds.forEach { appendLine(it) }
            }

        outputFile.writeText(output)

        println()
        println("=== COLLECTION COMPLETE ===")
        println("Total games: $totalGames")
        println("Winnable seeds: ${winnableSeeds.size}")
        println("Win rate: ${"%.4f".format(winRate)}%")
        println("Time: ${"%.1f".format(elapsed)} seconds")
        println("Output: ${outputFile.absolutePath}")
    }

    @Test
    fun `analyze winnable seed patterns`() {
        val outputFile = File("build/seed-analysis.txt")
        val output = StringBuilder()

        fun log(s: String = "") {
            output.appendLine(s)
        }

        // Load winnable seeds from file
        val winnableSeedsFile = File("build/winnable-seeds.txt")
        if (!winnableSeedsFile.exists()) {
            log("ERROR: build/winnable-seeds.txt not found. Run `collect winnable seeds` test first.")
            outputFile.writeText(output.toString())
            return
        }

        val winnableSeeds =
            winnableSeedsFile
                .readLines()
                .filter { it.isNotBlank() && !it.startsWith("#") }
                .mapNotNull { it.trim().toLongOrNull() }

        log("=== WINNABLE SEED PATTERN ANALYSIS ===")
        log("Analyzing ${winnableSeeds.size} winning seeds vs ${winnableSeeds.size} sample losing seeds")
        log()

        // Generate a set of losing seeds for comparison
        // Use seeds just after each winning seed (likely to lose)
        val losingSeedCandidates = mutableListOf<Long>()
        val player = HeuristicPlayer()

        var offset = 1L
        for (winningSeed in winnableSeeds) {
            while (losingSeedCandidates.size < winnableSeeds.size) {
                val candidateSeed = winningSeed + offset
                val game = GameState.newGame(Random(candidateSeed))
                val finalState = player.playGame(game)
                val won =
                    finalState.health > 0 &&
                        finalState.deck.isEmpty &&
                        (finalState.currentRoom == null || finalState.currentRoom.isEmpty())

                if (!won && candidateSeed !in winnableSeeds) {
                    losingSeedCandidates.add(candidateSeed)
                    break
                }
                offset++
            }
            offset = 1L
        }
        val losingSeeds = losingSeedCandidates.take(winnableSeeds.size)

        // Analyze deck patterns
        val winningDeckAnalyses =
            winnableSeeds.map { seed ->
                analyzeDeck(seed, Deck.create().shuffle(Random(seed)).cards)
            }
        val losingDeckAnalyses =
            losingSeeds.map { seed ->
                analyzeDeck(seed, Deck.create().shuffle(Random(seed)).cards)
            }

        // 1. Face Card Distribution Analysis
        log("=== FACE CARD DISTRIBUTION (J, Q, K, A monsters) ===")
        analyzeFaceCardDistribution(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 2. Early Weapon Availability
        log()
        log("=== EARLY WEAPON AVAILABILITY ===")
        analyzeEarlyWeapons(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 3. Monster Clustering
        log()
        log("=== MONSTER CLUSTERING ===")
        analyzeMonsterClustering(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 4. Difficulty Metrics
        log()
        log("=== DIFFICULTY METRICS ===")
        analyzeDifficultyMetrics(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 5. Ace Monster Positions
        log()
        log("=== ACE MONSTER POSITIONS ===")
        analyzeAcePositions(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 6. High-Value Weapon Positions
        log()
        log("=== HIGH-VALUE WEAPON POSITIONS (8, 9, 10) ===")
        analyzeHighWeaponPositions(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 7. Potion Distribution
        log()
        log("=== POTION DISTRIBUTION ===")
        analyzePotionDistribution(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 8. First Room Composition
        log()
        log("=== FIRST ROOM COMPOSITION (cards 1-4) ===")
        analyzeFirstRoom(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 9. Card Type Sequence Patterns
        log()
        log("=== CARD TYPE IN POSITIONS ===")
        analyzeCardTypeByPosition(winningDeckAnalyses, losingDeckAnalyses, ::log)

        // 10. Monster Run Analysis
        log()
        log("=== CONSECUTIVE MONSTER RUNS ===")
        analyzeMonsterRuns(winningDeckAnalyses, losingDeckAnalyses, ::log)

        outputFile.writeText(output.toString())
        println("Analysis written to: ${outputFile.absolutePath}")
        println()
        println(output.toString())
    }

    /**
     * Data class to hold deck analysis metrics
     */
    data class DeckAnalysis(
        val seed: Long,
        val cards: List<Card>,
        // Face card positions (J=11, Q=12, K=13, A=14 monsters only)
        val faceCardPositions: List<Int>,
        // Weapon positions
        val weaponPositions: List<Int>,
        // Potion positions
        val potionPositions: List<Int>,
        // High monster (10+) positions
        val highMonsterPositions: List<Int>,
        // Ace monster positions
        val acePositions: List<Int>,
        // High weapon (8+) positions
        val highWeaponPositions: List<Int>,
        // Cards in first 4 positions
        val firstRoom: List<Card>,
        // Cards in first 8 positions (first 2 rooms)
        val firstTwoRooms: List<Card>,
    )

    private fun analyzeDeck(
        seed: Long,
        cards: List<Card>,
    ): DeckAnalysis {
        val faceCardPositions = mutableListOf<Int>()
        val weaponPositions = mutableListOf<Int>()
        val potionPositions = mutableListOf<Int>()
        val highMonsterPositions = mutableListOf<Int>()
        val acePositions = mutableListOf<Int>()
        val highWeaponPositions = mutableListOf<Int>()

        cards.forEachIndexed { index, card ->
            when (card.type) {
                CardType.MONSTER -> {
                    if (card.rank in listOf(Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)) {
                        faceCardPositions.add(index)
                    }
                    if (card.value >= 10) {
                        highMonsterPositions.add(index)
                    }
                    if (card.rank == Rank.ACE) {
                        acePositions.add(index)
                    }
                }
                CardType.WEAPON -> {
                    weaponPositions.add(index)
                    if (card.value >= 8) {
                        highWeaponPositions.add(index)
                    }
                }
                CardType.POTION -> {
                    potionPositions.add(index)
                }
            }
        }

        return DeckAnalysis(
            seed = seed,
            cards = cards,
            faceCardPositions = faceCardPositions,
            weaponPositions = weaponPositions,
            potionPositions = potionPositions,
            highMonsterPositions = highMonsterPositions,
            acePositions = acePositions,
            highWeaponPositions = highWeaponPositions,
            firstRoom = cards.take(4),
            firstTwoRooms = cards.take(8),
        )
    }

    private fun analyzeFaceCardDistribution(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        // Analyze spread of face cards (standard deviation of positions)
        fun spreadMetric(positions: List<Int>): Double {
            if (positions.size <= 1) return 0.0
            val mean = positions.average()
            val variance = positions.map { (it - mean) * (it - mean) }.average()
            return kotlin.math.sqrt(variance)
        }

        // Average gap between consecutive face cards
        fun avgGap(positions: List<Int>): Double {
            if (positions.size <= 1) return 0.0
            val sorted = positions.sorted()
            val gaps = sorted.zipWithNext { a, b -> b - a }
            return gaps.average()
        }

        val winSpread = winning.map { spreadMetric(it.faceCardPositions) }.average()
        val loseSpread = losing.map { spreadMetric(it.faceCardPositions) }.average()

        val winGap = winning.map { avgGap(it.faceCardPositions) }.average()
        val loseGap = losing.map { avgGap(it.faceCardPositions) }.average()

        val winAvgPos = winning.flatMap { it.faceCardPositions }.average()
        val loseAvgPos = losing.flatMap { it.faceCardPositions }.average()

        log("Face cards = J, Q, K, A in Clubs and Spades (8 total per deck)")
        log("")
        log("Average position of face cards:")
        log("  Winning: ${"%.2f".format(winAvgPos)} (out of 44)")
        log("  Losing:  ${"%.2f".format(loseAvgPos)}")
        log(
            "  Delta:   ${"%.2f".format(
                winAvgPos - loseAvgPos,
            )} ${if (winAvgPos > loseAvgPos) "(face cards LATER in winning)" else "(face cards EARLIER in winning)"}",
        )
        log("")
        log("Spread (std dev of positions):")
        log("  Winning: ${"%.2f".format(winSpread)}")
        log("  Losing:  ${"%.2f".format(loseSpread)}")
        log(
            "  Delta:   ${"%.2f".format(
                winSpread - loseSpread,
            )} ${if (winSpread > loseSpread) "(MORE spread in winning)" else "(LESS spread in winning)"}",
        )
        log("")
        log("Average gap between consecutive face cards:")
        log("  Winning: ${"%.2f".format(winGap)} positions")
        log("  Losing:  ${"%.2f".format(loseGap)} positions")
        log("  Delta:   ${"%.2f".format(winGap - loseGap)}")
    }

    private fun analyzeEarlyWeapons(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        // Weapon in first 4 cards (first room)
        val winHasWeaponFirst4 = winning.count { it.weaponPositions.any { pos -> pos < 4 } }
        val loseHasWeaponFirst4 = losing.count { it.weaponPositions.any { pos -> pos < 4 } }

        // Weapon in first 8 cards (first 2 rooms)
        val winHasWeaponFirst8 = winning.count { it.weaponPositions.any { pos -> pos < 8 } }
        val loseHasWeaponFirst8 = losing.count { it.weaponPositions.any { pos -> pos < 8 } }

        // High weapon (8+) in first 8 cards
        val winHasHighWeaponFirst8 = winning.count { it.highWeaponPositions.any { pos -> pos < 8 } }
        val loseHasHighWeaponFirst8 = losing.count { it.highWeaponPositions.any { pos -> pos < 8 } }

        // Position of first weapon
        val winFirstWeaponPos = winning.map { it.weaponPositions.minOrNull() ?: 44 }.average()
        val loseFirstWeaponPos = losing.map { it.weaponPositions.minOrNull() ?: 44 }.average()

        // Position of first high weapon
        val winFirstHighWeaponPos = winning.map { it.highWeaponPositions.minOrNull() ?: 44 }.average()
        val loseFirstHighWeaponPos = losing.map { it.highWeaponPositions.minOrNull() ?: 44 }.average()

        log("Has weapon in first 4 cards (Room 1):")
        log(
            "  Winning: $winHasWeaponFirst4/${winning.size} (${"%.1f".format(
                winHasWeaponFirst4 * 100.0 / winning.size,
            )}%)",
        )
        log(
            "  Losing:  $loseHasWeaponFirst4/${losing.size} (${"%.1f".format(
                loseHasWeaponFirst4 * 100.0 / losing.size,
            )}%)",
        )
        log("")
        log("Has weapon in first 8 cards (Rooms 1-2):")
        log(
            "  Winning: $winHasWeaponFirst8/${winning.size} (${"%.1f".format(
                winHasWeaponFirst8 * 100.0 / winning.size,
            )}%)",
        )
        log(
            "  Losing:  $loseHasWeaponFirst8/${losing.size} (${"%.1f".format(
                loseHasWeaponFirst8 * 100.0 / losing.size,
            )}%)",
        )
        log("")
        log("Has HIGH weapon (8+) in first 8 cards:")
        log(
            "  Winning: $winHasHighWeaponFirst8/${winning.size} (${"%.1f".format(
                winHasHighWeaponFirst8 * 100.0 / winning.size,
            )}%)",
        )
        log(
            "  Losing:  $loseHasHighWeaponFirst8/${losing.size} (${"%.1f".format(
                loseHasHighWeaponFirst8 * 100.0 / losing.size,
            )}%)",
        )
        log("")
        log("Average position of FIRST weapon:")
        log("  Winning: ${"%.2f".format(winFirstWeaponPos)}")
        log("  Losing:  ${"%.2f".format(loseFirstWeaponPos)}")
        log(
            "  Delta:   ${"%.2f".format(
                loseFirstWeaponPos - winFirstWeaponPos,
            )} ${if (winFirstWeaponPos < loseFirstWeaponPos) "(first weapon EARLIER in winning)" else "(first weapon LATER in winning)"}",
        )
        log("")
        log("Average position of FIRST high weapon (8+):")
        log("  Winning: ${"%.2f".format(winFirstHighWeaponPos)}")
        log("  Losing:  ${"%.2f".format(loseFirstHighWeaponPos)}")
        log("  Delta:   ${"%.2f".format(loseFirstHighWeaponPos - winFirstHighWeaponPos)}")
    }

    private fun analyzeMonsterClustering(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        // Find max run of consecutive monster positions
        fun maxMonsterRun(cards: List<Card>): Int {
            var maxRun = 0
            var currentRun = 0
            for (card in cards) {
                if (card.type == CardType.MONSTER) {
                    currentRun++
                    maxRun = maxOf(maxRun, currentRun)
                } else {
                    currentRun = 0
                }
            }
            return maxRun
        }

        // Find runs of high monsters (10+)
        fun maxHighMonsterRun(cards: List<Card>): Int {
            var maxRun = 0
            var currentRun = 0
            for (card in cards) {
                if (card.type == CardType.MONSTER && card.value >= 10) {
                    currentRun++
                    maxRun = maxOf(maxRun, currentRun)
                } else {
                    currentRun = 0
                }
            }
            return maxRun
        }

        val winMaxRun = winning.map { maxMonsterRun(it.cards) }.average()
        val loseMaxRun = losing.map { maxMonsterRun(it.cards) }.average()

        val winMaxHighRun = winning.map { maxHighMonsterRun(it.cards) }.average()
        val loseMaxHighRun = losing.map { maxHighMonsterRun(it.cards) }.average()

        // Count how often there are 4+ monsters in a row
        val winHas4PlusRun = winning.count { maxMonsterRun(it.cards) >= 4 }
        val loseHas4PlusRun = losing.count { maxMonsterRun(it.cards) >= 4 }

        log("Max consecutive monsters in deck:")
        log("  Winning: ${"%.2f".format(winMaxRun)}")
        log("  Losing:  ${"%.2f".format(loseMaxRun)}")
        log(
            "  Delta:   ${"%.2f".format(
                loseMaxRun - winMaxRun,
            )} ${if (winMaxRun < loseMaxRun) "(FEWER consecutive monsters in winning)" else ""}",
        )
        log("")
        log("Max consecutive HIGH monsters (10+):")
        log("  Winning: ${"%.2f".format(winMaxHighRun)}")
        log("  Losing:  ${"%.2f".format(loseMaxHighRun)}")
        log("")
        log("Decks with 4+ consecutive monsters:")
        log("  Winning: $winHas4PlusRun/${winning.size} (${"%.1f".format(winHas4PlusRun * 100.0 / winning.size)}%)")
        log("  Losing:  $loseHas4PlusRun/${losing.size} (${"%.1f".format(loseHas4PlusRun * 100.0 / losing.size)}%)")
    }

    private fun analyzeDifficultyMetrics(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        fun totalMonsterDamage(cards: List<Card>): Int = cards.filter { it.type == CardType.MONSTER }.sumOf { it.value }

        fun totalWeaponValue(cards: List<Card>): Int = cards.filter { it.type == CardType.WEAPON }.sumOf { it.value }

        fun totalPotionValue(cards: List<Card>): Int = cards.filter { it.type == CardType.POTION }.sumOf { it.value }

        // These are constant for all decks, but let's verify
        val sampleWin = winning.first().cards
        val sampleLose = losing.first().cards

        log("Total monster damage: ${totalMonsterDamage(sampleWin)} (same for all decks)")
        log("Total weapon value:   ${totalWeaponValue(sampleWin)} (same for all decks)")
        log("Total potion value:   ${totalPotionValue(sampleWin)} (same for all decks)")
        log("")

        // Early difficulty: monster value in first 12 cards vs weapon/potion value
        fun earlyDifficulty(cards: List<Card>): Int {
            val first12 = cards.take(12)
            val monsterDamage = first12.filter { it.type == CardType.MONSTER }.sumOf { it.value }
            val weaponValue = first12.filter { it.type == CardType.WEAPON }.sumOf { it.value }
            val potionValue = first12.filter { it.type == CardType.POTION }.sumOf { it.value }
            return monsterDamage - weaponValue - potionValue
        }

        val winEarlyDiff = winning.map { earlyDifficulty(it.cards) }.average()
        val loseEarlyDiff = losing.map { earlyDifficulty(it.cards) }.average()

        log("Early difficulty (first 12 cards: monster dmg - weapon - potion):")
        log("  Winning: ${"%.2f".format(winEarlyDiff)}")
        log("  Losing:  ${"%.2f".format(loseEarlyDiff)}")
        log(
            "  Delta:   ${"%.2f".format(
                loseEarlyDiff - winEarlyDiff,
            )} ${if (winEarlyDiff < loseEarlyDiff) "(LOWER early difficulty in winning)" else ""}",
        )

        // Mid-game difficulty (cards 13-24)
        fun midDifficulty(cards: List<Card>): Int {
            val mid12 = cards.drop(12).take(12)
            val monsterDamage = mid12.filter { it.type == CardType.MONSTER }.sumOf { it.value }
            val weaponValue = mid12.filter { it.type == CardType.WEAPON }.sumOf { it.value }
            val potionValue = mid12.filter { it.type == CardType.POTION }.sumOf { it.value }
            return monsterDamage - weaponValue - potionValue
        }

        val winMidDiff = winning.map { midDifficulty(it.cards) }.average()
        val loseMidDiff = losing.map { midDifficulty(it.cards) }.average()

        log("")
        log("Mid-game difficulty (cards 13-24):")
        log("  Winning: ${"%.2f".format(winMidDiff)}")
        log("  Losing:  ${"%.2f".format(loseMidDiff)}")

        // Late difficulty (cards 25-44)
        fun lateDifficulty(cards: List<Card>): Int {
            val late = cards.drop(24)
            val monsterDamage = late.filter { it.type == CardType.MONSTER }.sumOf { it.value }
            val weaponValue = late.filter { it.type == CardType.WEAPON }.sumOf { it.value }
            val potionValue = late.filter { it.type == CardType.POTION }.sumOf { it.value }
            return monsterDamage - weaponValue - potionValue
        }

        val winLateDiff = winning.map { lateDifficulty(it.cards) }.average()
        val loseLateDiff = losing.map { lateDifficulty(it.cards) }.average()

        log("")
        log("Late-game difficulty (cards 25-44):")
        log("  Winning: ${"%.2f".format(winLateDiff)}")
        log("  Losing:  ${"%.2f".format(loseLateDiff)}")
    }

    private fun analyzeAcePositions(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        // There are exactly 2 Ace monsters (A of Clubs and A of Spades)
        val winAceAvgPos = winning.flatMap { it.acePositions }.average()
        val loseAceAvgPos = losing.flatMap { it.acePositions }.average()

        val winFirstAce = winning.map { it.acePositions.minOrNull() ?: 44 }.average()
        val loseFirstAce = losing.map { it.acePositions.minOrNull() ?: 44 }.average()

        val winSecondAce = winning.map { it.acePositions.maxOrNull() ?: 44 }.average()
        val loseSecondAce = losing.map { it.acePositions.maxOrNull() ?: 44 }.average()

        // Gap between aces
        val winAceGap =
            winning
                .map {
                    val sorted = it.acePositions.sorted()
                    if (sorted.size >= 2) sorted[1] - sorted[0] else 0
                }.average()
        val loseAceGap =
            losing
                .map {
                    val sorted = it.acePositions.sorted()
                    if (sorted.size >= 2) sorted[1] - sorted[0] else 0
                }.average()

        // Aces in first half vs second half
        val winAcesInFirstHalf = winning.sumOf { it.acePositions.count { pos -> pos < 22 } }
        val loseAcesInFirstHalf = losing.sumOf { it.acePositions.count { pos -> pos < 22 } }

        log("There are 2 Ace monsters (A clubs, A spades) - value 14 each")
        log("")
        log("Average position of Aces:")
        log("  Winning: ${"%.2f".format(winAceAvgPos)}")
        log("  Losing:  ${"%.2f".format(loseAceAvgPos)}")
        log(
            "  Delta:   ${"%.2f".format(
                winAceAvgPos - loseAceAvgPos,
            )} ${if (winAceAvgPos > loseAceAvgPos) "(Aces LATER in winning)" else "(Aces EARLIER in winning)"}",
        )
        log("")
        log("Position of FIRST Ace:")
        log("  Winning: ${"%.2f".format(winFirstAce)}")
        log("  Losing:  ${"%.2f".format(loseFirstAce)}")
        log("")
        log("Position of SECOND Ace:")
        log("  Winning: ${"%.2f".format(winSecondAce)}")
        log("  Losing:  ${"%.2f".format(loseSecondAce)}")
        log("")
        log("Gap between Aces:")
        log("  Winning: ${"%.2f".format(winAceGap)}")
        log("  Losing:  ${"%.2f".format(loseAceGap)}")
        log("")
        log("Aces appearing in first half (positions 0-21):")
        log(
            "  Winning: $winAcesInFirstHalf/${winning.size * 2} (${"%.1f".format(
                winAcesInFirstHalf * 100.0 / (winning.size * 2),
            )}%)",
        )
        log(
            "  Losing:  $loseAcesInFirstHalf/${losing.size * 2} (${"%.1f".format(
                loseAcesInFirstHalf * 100.0 / (losing.size * 2),
            )}%)",
        )
    }

    private fun analyzeHighWeaponPositions(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        // There are 3 high weapons: 8, 9, 10 of diamonds
        val winHighWeaponAvgPos = winning.flatMap { it.highWeaponPositions }.average()
        val loseHighWeaponAvgPos = losing.flatMap { it.highWeaponPositions }.average()

        val winFirst = winning.map { it.highWeaponPositions.minOrNull() ?: 44 }.average()
        val loseFirst = losing.map { it.highWeaponPositions.minOrNull() ?: 44 }.average()

        // High weapon before first high monster
        val winHighWeaponBeforeHighMonster =
            winning.count {
                val firstHighWeapon = it.highWeaponPositions.minOrNull() ?: Int.MAX_VALUE
                val firstHighMonster = it.highMonsterPositions.minOrNull() ?: Int.MAX_VALUE
                firstHighWeapon < firstHighMonster
            }
        val loseHighWeaponBeforeHighMonster =
            losing.count {
                val firstHighWeapon = it.highWeaponPositions.minOrNull() ?: Int.MAX_VALUE
                val firstHighMonster = it.highMonsterPositions.minOrNull() ?: Int.MAX_VALUE
                firstHighWeapon < firstHighMonster
            }

        log("High weapons: 8, 9, 10 of diamonds (3 total)")
        log("")
        log("Average position of high weapons:")
        log("  Winning: ${"%.2f".format(winHighWeaponAvgPos)}")
        log("  Losing:  ${"%.2f".format(loseHighWeaponAvgPos)}")
        log(
            "  Delta:   ${"%.2f".format(
                loseHighWeaponAvgPos - winHighWeaponAvgPos,
            )} ${if (winHighWeaponAvgPos < loseHighWeaponAvgPos) "(high weapons EARLIER in winning)" else ""}",
        )
        log("")
        log("Position of FIRST high weapon:")
        log("  Winning: ${"%.2f".format(winFirst)}")
        log("  Losing:  ${"%.2f".format(loseFirst)}")
        log("")
        log("High weapon appears BEFORE first high monster (10+):")
        log(
            "  Winning: $winHighWeaponBeforeHighMonster/${winning.size} (${"%.1f".format(
                winHighWeaponBeforeHighMonster * 100.0 / winning.size,
            )}%)",
        )
        log(
            "  Losing:  $loseHighWeaponBeforeHighMonster/${losing.size} (${"%.1f".format(
                loseHighWeaponBeforeHighMonster * 100.0 / losing.size,
            )}%)",
        )
    }

    private fun analyzePotionDistribution(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        val winPotionAvgPos = winning.flatMap { it.potionPositions }.average()
        val losePotionAvgPos = losing.flatMap { it.potionPositions }.average()

        // Potions in first half
        val winPotionsFirstHalf = winning.sumOf { it.potionPositions.count { pos -> pos < 22 } }
        val losePotionsFirstHalf = losing.sumOf { it.potionPositions.count { pos -> pos < 22 } }

        // Average position of high potions (8+)
        fun highPotionPositions(cards: List<Card>): List<Int> =
            cards.mapIndexedNotNull { index, card ->
                if (card.type == CardType.POTION && card.value >= 8) index else null
            }

        val winHighPotionAvg = winning.flatMap { highPotionPositions(it.cards) }.average()
        val loseHighPotionAvg = losing.flatMap { highPotionPositions(it.cards) }.average()

        log("Average position of all potions:")
        log("  Winning: ${"%.2f".format(winPotionAvgPos)}")
        log("  Losing:  ${"%.2f".format(losePotionAvgPos)}")
        log("")
        log("Potions in first half (positions 0-21):")
        log(
            "  Winning: $winPotionsFirstHalf/${winning.size * 9} (${"%.1f".format(
                winPotionsFirstHalf * 100.0 / (winning.size * 9),
            )}%)",
        )
        log(
            "  Losing:  $losePotionsFirstHalf/${losing.size * 9} (${"%.1f".format(
                losePotionsFirstHalf * 100.0 / (losing.size * 9),
            )}%)",
        )
        log("")
        log("Average position of HIGH potions (8, 9, 10):")
        log("  Winning: ${"%.2f".format(winHighPotionAvg)}")
        log("  Losing:  ${"%.2f".format(loseHighPotionAvg)}")
    }

    private fun analyzeFirstRoom(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        fun roomStats(room: List<Card>): Triple<Int, Int, Int> {
            val monsters = room.count { it.type == CardType.MONSTER }
            val weapons = room.count { it.type == CardType.WEAPON }
            val potions = room.count { it.type == CardType.POTION }
            return Triple(monsters, weapons, potions)
        }

        // Distribution of card type counts in first room
        val winRoomStats = winning.map { roomStats(it.firstRoom) }
        val loseRoomStats = losing.map { roomStats(it.firstRoom) }

        val winAvgMonsters = winRoomStats.map { it.first }.average()
        val winAvgWeapons = winRoomStats.map { it.second }.average()
        val winAvgPotions = winRoomStats.map { it.third }.average()

        val loseAvgMonsters = loseRoomStats.map { it.first }.average()
        val loseAvgWeapons = loseRoomStats.map { it.second }.average()
        val loseAvgPotions = loseRoomStats.map { it.third }.average()

        // Total monster value in first room
        val winFirstRoomMonsterVal =
            winning
                .map {
                    it.firstRoom.filter { c -> c.type == CardType.MONSTER }.sumOf { c -> c.value }
                }.average()
        val loseFirstRoomMonsterVal =
            losing
                .map {
                    it.firstRoom.filter { c -> c.type == CardType.MONSTER }.sumOf { c -> c.value }
                }.average()

        log("Average card count in first room:")
        log(
            "  Winning: ${"%.2f".format(
                winAvgMonsters,
            )} monsters, ${"%.2f".format(winAvgWeapons)} weapons, ${"%.2f".format(winAvgPotions)} potions",
        )
        log(
            "  Losing:  ${"%.2f".format(
                loseAvgMonsters,
            )} monsters, ${"%.2f".format(loseAvgWeapons)} weapons, ${"%.2f".format(loseAvgPotions)} potions",
        )
        log("")
        log("Total monster damage in first room:")
        log("  Winning: ${"%.2f".format(winFirstRoomMonsterVal)}")
        log("  Losing:  ${"%.2f".format(loseFirstRoomMonsterVal)}")

        // All 4 cards monster frequency
        val win4Monsters = winning.count { it.firstRoom.all { c -> c.type == CardType.MONSTER } }
        val lose4Monsters = losing.count { it.firstRoom.all { c -> c.type == CardType.MONSTER } }

        log("")
        log("First room is ALL monsters:")
        log("  Winning: $win4Monsters/${winning.size} (${"%.1f".format(win4Monsters * 100.0 / winning.size)}%)")
        log("  Losing:  $lose4Monsters/${losing.size} (${"%.1f".format(lose4Monsters * 100.0 / losing.size)}%)")
    }

    private fun analyzeCardTypeByPosition(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        // For key positions, what card type appears?
        val positions = listOf(0, 1, 2, 3, 4, 7, 11, 15, 21, 32, 43)

        log("Probability of card type at key positions:")
        log("")
        log("Position   | Win Monster | Lose Monster | Win Weapon | Lose Weapon | Win Potion | Lose Potion")
        log("-".repeat(95))

        for (pos in positions) {
            val winMonster = winning.count { it.cards[pos].type == CardType.MONSTER } * 100.0 / winning.size
            val loseMonster = losing.count { it.cards[pos].type == CardType.MONSTER } * 100.0 / losing.size
            val winWeapon = winning.count { it.cards[pos].type == CardType.WEAPON } * 100.0 / winning.size
            val loseWeapon = losing.count { it.cards[pos].type == CardType.WEAPON } * 100.0 / losing.size
            val winPotion = winning.count { it.cards[pos].type == CardType.POTION } * 100.0 / winning.size
            val losePotion = losing.count { it.cards[pos].type == CardType.POTION } * 100.0 / losing.size

            log(
                "%9d  | %10.1f%% | %11.1f%% | %9.1f%% | %10.1f%% | %9.1f%% | %10.1f%%".format(
                    pos,
                    winMonster,
                    loseMonster,
                    winWeapon,
                    loseWeapon,
                    winPotion,
                    losePotion,
                ),
            )
        }
    }

    private fun analyzeMonsterRuns(
        winning: List<DeckAnalysis>,
        losing: List<DeckAnalysis>,
        log: (String) -> Unit,
    ) {
        // Count runs of consecutive monsters of various lengths
        fun countRuns(
            cards: List<Card>,
            minLength: Int,
        ): Int {
            var count = 0
            var currentRun = 0
            for (card in cards) {
                if (card.type == CardType.MONSTER) {
                    currentRun++
                } else {
                    if (currentRun >= minLength) count++
                    currentRun = 0
                }
            }
            if (currentRun >= minLength) count++
            return count
        }

        // Total damage in longest run
        fun longestRunDamage(cards: List<Card>): Int {
            var maxDamage = 0
            var currentDamage = 0
            for (card in cards) {
                if (card.type == CardType.MONSTER) {
                    currentDamage += card.value
                } else {
                    maxDamage = maxOf(maxDamage, currentDamage)
                    currentDamage = 0
                }
            }
            return maxOf(maxDamage, currentDamage)
        }

        log("Average number of runs of 3+ consecutive monsters:")
        val win3Plus = winning.map { countRuns(it.cards, 3) }.average()
        val lose3Plus = losing.map { countRuns(it.cards, 3) }.average()
        log("  Winning: ${"%.2f".format(win3Plus)}")
        log("  Losing:  ${"%.2f".format(lose3Plus)}")

        log("")
        log("Average number of runs of 5+ consecutive monsters:")
        val win5Plus = winning.map { countRuns(it.cards, 5) }.average()
        val lose5Plus = losing.map { countRuns(it.cards, 5) }.average()
        log("  Winning: ${"%.2f".format(win5Plus)}")
        log("  Losing:  ${"%.2f".format(lose5Plus)}")

        log("")
        log("Total damage in longest consecutive monster run:")
        val winLongestDamage = winning.map { longestRunDamage(it.cards) }.average()
        val loseLongestDamage = losing.map { longestRunDamage(it.cards) }.average()
        log("  Winning: ${"%.2f".format(winLongestDamage)}")
        log("  Losing:  ${"%.2f".format(loseLongestDamage)}")
        log(
            "  Delta:   ${"%.2f".format(
                loseLongestDamage - winLongestDamage,
            )} ${if (winLongestDamage < loseLongestDamage) "(LESS concentrated damage in winning)" else ""}",
        )
    }
}
